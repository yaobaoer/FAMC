from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def paper_split(df, seed=42):
    pass
    buckets = [[], [], []]
    rng = np.random.default_rng(seed)
    for start in range(0, len(df), 2000):
        ix = np.arange(start, min(start + 2000, len(df)))
        key = (df.iloc[ix]['device'].astype(str) + '_' +
               df.iloc[ix]['final_type'].astype(str)).to_numpy()
        try:
            tv, te = next(StratifiedShuffleSplit(1, test_size=.2, random_state=seed).split(ix, key))
            tr, va = next(StratifiedShuffleSplit(1, test_size=.2, random_state=seed + 1).split(tv, key[tv]))
            parts = (ix[tv[tr]], ix[tv[va]], ix[te])
        except ValueError:
            parts = [[], [], []]
            for label in np.unique(key):
                rows = rng.permutation(ix[key == label])
                n = len(rows)
                a, b = int(round(.64*n)), int(round(.80*n))
                for dest, values in zip(parts, (rows[:a], rows[a:b], rows[b:])):
                    dest.extend(values.tolist())
        for dest, values in zip(buckets, parts):
            dest.extend(values)
    return tuple(df.iloc[sorted(rows)].copy() for rows in buckets)
