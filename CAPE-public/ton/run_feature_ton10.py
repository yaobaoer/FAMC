                       
pass
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import argparse
import json
import hashlib
from pathlib import Path
import sys
SUPPORT_ROOT = Path('E:/apt/测试/third')
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import silhouette_score
import run_feature_capacity as capacity
import run_feature_comparison as base

DATA_ROOT = Path('E:/apt/测试/10')
DEVICES = ['weather','modbus','thermostat','fridge','garage_door','gps','motion_light']
JOBS = [('Ours enhanced','enhanced'),('CoLD','cold'),('tFusion','tfusion'),
        ('TIPSO-Attn','tipsogan'),('FeCo','feco_small'),('AOC-IDS','aoc_small')]

def split_data(df):
                                                                                       
    df=df.copy(); df['time_group']=df.index//2000
    indices=[[],[],[]]
    for _,group in df.groupby('time_group'):
        if len(group)<3:
            indices[0].extend(group.index); continue
        try:
            tv,te=next(StratifiedShuffleSplit(1,test_size=.2,random_state=42).split(group,group.label_encoded))
        except ValueError:
            tv,te=np.arange(len(group)),np.array([],dtype=int)
        if len(tv)>1:
            sub=group.iloc[tv]
            try:
                tr,va=next(StratifiedShuffleSplit(1,test_size=.2,random_state=42).split(sub,sub.label_encoded))
                indices[0].extend(group.index[tv][tr]); indices[1].extend(group.index[tv][va])
            except ValueError:
                indices[0].extend(group.index[tv])
        else: indices[0].extend(group.index[tv])
        indices[2].extend(group.index[te])
    return [df.loc[idx].sort_index().reset_index(drop=True) for idx in indices]

class Windows(Dataset):
    def __init__(self,x,y,step,limit,seed):
        self.x=x
        self.starts=np.arange(0,len(y)-128+1,step)
        if len(self.starts)>limit:
            labels=y[self.starts+127]
            try:
                _,take=next(StratifiedShuffleSplit(1,test_size=limit,random_state=seed).split(self.starts,labels))
            except ValueError:
                take=np.random.RandomState(seed).choice(len(self.starts),limit,False)
            self.starts=np.sort(self.starts[take])
        self.labels=y[self.starts+127]
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        start=self.starts[i]
        return torch.from_numpy(self.x[start:start+128]),torch.tensor(self.labels[i])

def load_data(out):
    cache=out/'data.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved: return {key:saved[key] for key in saved.files}
    df=pd.read_csv(DATA_ROOT/'iot_fusion_weather_all.csv',low_memory=False)
    df['timestamp']=pd.to_datetime(df.timestamp)
    df=df.sort_values('timestamp').reset_index(drop=True)
    counts=df.final_type.value_counts()
    df=df[df.final_type.isin(counts[counts>10].index)].copy()
    le=LabelEncoder(); df['label_encoded']=le.fit_transform(df.final_type)
    columns=[]; dims=[]
    for device in DEVICES:
        cols=[c for c in df if c.startswith(device+'_') and not c.endswith(('_label','_type'))]
        columns.extend(cols); dims.append(len(cols))
    assert all(dims) and len(columns)==len(set(columns))
    tables=split_data(df)
    scaler=StandardScaler().fit(tables[0][columns])
    result=dict(names=le.classes_.astype(str),dims=np.asarray(dims),columns=np.asarray(columns))
    for name,table in zip(['train','val','test'],tables):
        x=np.nan_to_num(np.clip(scaler.transform(table[columns]),-8,8),nan=0.,posinf=8.,neginf=-8.).astype('float32')
        if name!='test': x[:,:dims[0]]=0.                                                
        result[name+'_x']=x; result[name+'_y']=table.label_encoded.to_numpy('int64')
    np.savez_compressed(cache,**result)
    print('DATA_READY',json.dumps(dict(rows={n:len(result[n+'_y']) for n in ['train','val','test']},
          dims=dims,classes=le.classes_.tolist())),flush=True)
    return result

class SmallFeCo(nn.Module):
    out_dim=16
    def __init__(self,d_in):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),
                               nn.Linear(64,32),nn.Linear(32,16))
    def forward(self,x): return self.net(x)

class SmallAOC(nn.Module):
    out_dim=16
    def __init__(self,d_in):
        super().__init__()
        self.enc1=nn.Linear(d_in,16); self.enc2=nn.Linear(16,16)
        self.dec1=nn.Linear(16,16); self.dec2=nn.Linear(16,d_in)
    def forward(self,x):
        z=self.enc2(F.relu(self.enc1(x)))
        self.reconstruction=self.dec2(F.relu(self.dec1(z)))
        return z

class Model(nn.Module):
    def __init__(self,dims,n_cls,kind):
        super().__init__(); self.kind=kind
        if kind=='enhanced':
            self.extractor=base.cnn.FrontV2(dims,d_model=32,dropout=.1)
            self.extractor.use_gat=False; self.extractor.set_gate_target(.3)
            self.metric_projection=nn.Sequential(nn.Linear(self.extractor.out_dim,128),nn.GELU(),
                                                 nn.LayerNorm(128),nn.Linear(128,64))
            self.prototypes=nn.Parameter(torch.randn(n_cls,64)); width=64
        elif kind in ['feco_small','aoc_small']:
            self.extractor=SmallFeCo(sum(dims)) if kind=='feco_small' else SmallAOC(sum(dims))
            width=16
        else:
            self.extractor=base.build_paper_backbone(kind,sum(dims),d_hidden=64); width=64
        self.backbone=base.cnn.CNNBackbone(width,64)
        self.classifier=nn.Linear(64,n_cls)
    def forward(self,x):
        if self.kind=='enhanced':
            seq,aux=self.extractor(x,return_features=True)
            seq=F.normalize(self.metric_projection(seq),dim=-1)
        else: seq,aux=self.extractor(x),None
        return self.classifier(self.backbone(seq)[:,-1]),seq[:,-1],aux

def smoke():
    for title,kind in JOBS:
        model=Model([3,4,2,2,2,2,2],8,kind).to(base.cnn.DEVICE)
        x=torch.randn(8,128,17,device=base.cnn.DEVICE); x[:,:,:3]=0.
        y=torch.arange(8,device=base.cnn.DEVICE)%4
        logits,z,aux=model(x); loss=F.cross_entropy(logits,y)
        if kind=='enhanced':
            assert torch.count_nonzero(aux['h_prime'][:,:,0])==0
            loss=loss+base.supcon(z,y)+F.cross_entropy(16*z@F.normalize(model.prototypes,dim=-1).T,y)
            loss=loss+model.extractor.cross_protocol_contrastive_loss(aux['h_prime'][:,-1:],y)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print('SMOKE_OK',title,tuple(z.shape),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(DATA_ROOT/'feature_comparison_capacity_v1'))
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--redraw-only',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(3)
    if args.smoke: smoke(); return
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    capacity.OUT=out; capacity.Model=Model
    if not args.redraw_only:
        manifest=dict(args=vars(args),dataset='ToN-IoT fused seven-device CSV',
            window=128,steps=[1,2,2],window_caps=[12000,6000,6000],seeds=[42,123,2024],
            split='same per-2000-row stratified split as directory 10; rows ordered within each partition; not chronological holdout',
            protocol='train/val drop weather, test all seven devices',
            feature='pre-CNN last position; enhanced Ours uses learned normalized metric projection',
            baselines='supervised core adaptations; small FeCo hidden 32/64/32 output16; small AOC hidden16 latent16; CE only, no original binary CRC objectives',
            configs=capacity.CONFIGS,selection='validation seed42 only, before test evaluation',
            source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),Path(base.__file__),Path(capacity.__file__),Path(base.cnn.__file__),
                 Path(base.cnn.FrontV2.__module__.replace('.','/')+'.py'),Path('feature_backbones.py')]})
        manifest_path=out/'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text('utf-8'))!=manifest:
            raise ValueError('Output contains a different configuration; choose a new --output.')
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    data=load_data(out)
    ds={name:Windows(data[name+'_x'],data[name+'_y'],1 if name=='train' else 2,
                     12000 if name=='train' else 6000,42+i) for i,name in enumerate(['train','val','test'])}
    print('WINDOWS',{name:len(dataset) for name,dataset in ds.items()},flush=True)
    for name,dataset in ds.items():
        if len(np.unique(dataset.labels))!=len(data['names']):
            raise ValueError(f'{name} windows miss a class: cannot silently discard it.')
    selection_path=out/'selection.json'
    if selection_path.exists(): selected=json.loads(selection_path.read_text('utf-8'))['config']
    elif args.redraw_only: raise ValueError('No completed configuration selection.')
    else:
        trials=[]
        for config in capacity.CONFIGS:
            tag='trial_'+config['name']+'_s42'; done=out/(tag+'_done.json')
            if done.exists(): info=json.loads(done.read_text('utf-8'))
            else:
                blob=capacity.fit('enhanced',42,config,data,ds,args,tag)
                info=dict(config=config,val_silhouette=blob['val_silhouette'],val_f1=blob['val_f1'],
                          epoch=blob['epoch'],checkpoint=str(out/(tag+'.pth')))
                done.write_text(json.dumps(info,indent=2),encoding='utf-8')
            trials.append(info)
        winner=max(trials,key=lambda row:row['val_silhouette']); selected=winner['config']
        selection_path.write_text(json.dumps(dict(config=selected,trials=trials),indent=2),encoding='utf-8')
        print('SELECTED',json.dumps(winner),flush=True)
    path=out/'metrics.csv'; rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    idx=base.viz.stratified_idx(ds['test'].labels,70,0)
    test_loader=DataLoader(ds['test'],batch_size=64,pin_memory=torch.cuda.is_available())
    for seed in [42,123,2024]:
        for title,kind in JOBS:
            if any(row['kind']==kind and int(row['seed'])==seed for row in rows): continue
            if args.redraw_only: raise ValueError('Missing completed method; redraw does not train.')
            if kind=='enhanced' and seed==42:
                blob=torch.load(out/f'trial_{selected["name"]}_s42.pth',map_location='cpu',weights_only=False)
                torch.save(blob,out/'enhanced_s42.pth')
            else: blob=capacity.fit(kind,seed,selected if kind=='enhanced' else {},data,ds,args,f'{kind}_s{seed}')
            model=Model(data['dims'].tolist(),len(data['names']),kind).to(base.cnn.DEVICE)
            model.load_state_dict(blob['model_state'])
            f1,z,y=base.evaluate(model,test_loader,idx)
            row=dict(method=title,kind=kind,seed=seed,epoch=blob['epoch'],dim=z.shape[-1],
                     parameters=blob['parameters'],f1=f1,**base.metrics(z,y,seed))
            rows.append(row); pd.DataFrame(rows).to_csv(path,index=False)
            np.savez_compressed(out/f'{kind}_s{seed}_features.npz',features=z,labels=y,indices=idx)
            print('RESULT_TON10',json.dumps(row),flush=True)
            del model; torch.cuda.empty_cache()
        panels=[]; reference=None
        for title,kind in JOBS:
            with np.load(out/f'{kind}_s{seed}_features.npz') as saved:
                z,y=saved['features'],saved['labels']; assert np.array_equal(saved['indices'],idx)
            if reference is None: reference=y
            assert np.array_equal(reference,y)
            panels.append((title,base.viz.project(z),float(silhouette_score(normalize(z),y))))
        base.viz.draw(panels,y,[str(n) for n in data['names']],str(out/f'capacity_clusters_s{seed}.png'),
            str(out/f'capacity_clusters_s{seed}.pdf'),
            'ToN-IoT feature comparison. Same test windows; pre-CNN last-position features. Number: original-space silhouette.')
        pd.DataFrame(rows).groupby('method')[['f1','silhouette','davies','ari','nmi']].agg(['mean','std','count']).to_csv(out/'summary.csv')
    print('TON10_FEATURE_COMPARISON_DONE',flush=True)

if __name__=='__main__': main()
