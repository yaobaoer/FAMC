import functools
import hashlib
import inspect
import re
import os
import numpy as np

_PAIRS = {}

def _digest(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str((a.shape, a.dtype)).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def paired_fit(fn):
    signature = inspect.signature(fn)
    @functools.wraps(fn)
    def checked(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
                                                                         
        if 'seed' in values and values['seed'] is None:
            values['seed'] = 42
        datasets = []
        for name in ('train_ds', 'val_ds', 'test_ds'):
            ds = values.get(name)
            if ds is None:
                continue
            items = []
            for attr in ('labels', 'samples', 'X', 'y', 'starts'):
                if hasattr(ds, attr):
                    items.append((attr, _digest(getattr(ds, attr))))
            if not items:
                raise ValueError('Paired fit cannot fingerprint ' + name)
            datasets.append((name, tuple(items)))
        weights = tuple((k, _digest(values[k])) for k in
                        ('cw', 'sample_weight', 'class_weight') if k in values)
        g = fn.__globals__
        budget = tuple((k, repr(g[k])) for k in
                       ('BATCH', 'BATCH_SIZE', 'EPOCHS', 'LR', 'PATIENCE') if k in g)
        artifact = values.get('ckpt', values.get('ckpt_path', 'pair'))
        pair_name = re.sub(r'(front|raw)', 'paired', os.path.basename(str(artifact)), flags=re.I)
        identity = (fn.__module__, fn.__name__, values.get('kind'),
                    values.get('seed'), pair_name)
        state = (tuple(datasets), weights, budget)
        previous = _PAIRS.setdefault(identity, state)
        if previous != state:
            raise ValueError('Raw/FAMC pair has different sampling weights or training budget')
        return fn(*bound.args, **bound.kwargs)
    return checked
