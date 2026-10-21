pass
from pathlib import Path
import os, sys, json, hashlib, importlib.util, types, gc, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'device_semantics_analysis_v1'
OUT.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'_tmp'/'matplotlib')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(3)
sys.stdout.reconfigure(encoding='utf-8')
SEEDS=[42,123,2024]
TON=ROOT.parent/'10'
records={k:[] for k in ['coverage','quality','conflicts','geometry','probes','weights','classification','class_gains','checks','interventions']}
manifest={'created':time.strftime('%Y-%m-%d %H:%M:%S'), 'mode':'Retained selected checkpoint diagnostics; not a three-seed experiment. Original multi-seed folders were removed during preparation.',
 'probe_sampling':'Up to 240 training / 180 test windows per aggregate class, deterministic seed 7108; nodes use local labels and true availability; no test fitting.',
 'stages':['projected','consensus','output'], 'checkpoints':[], 'inputs':[]}

def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write():
 for k,v in records.items():
  if v:pd.DataFrame(v).to_csv(OUT/(k+'.csv'),index=False,encoding='utf-8-sig')
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m

class LazyWindows(torch.utils.data.Dataset):
 def __init__(self,X,y,dims,window,step,cap=None,seed=42,local=None,valid=None):
  self.X=X;self.device_dims=list(dims);self.window=window
  self.starts=np.arange(0,len(X)-window+1,step,dtype=np.int64)
  if cap and len(self.starts)>cap:self.starts=np.sort(np.random.RandomState(seed).choice(self.starts,cap,replace=False))
  self.ends=self.starts+window-1;self.labels=np.asarray(y,dtype=np.int64)[self.ends]
  self.local=local;self.valid=valid
 def __len__(self):return len(self.labels)
 def __getitem__(self,i):return torch.from_numpy(self.X[self.starts[i]:self.starts[i]+self.window]),torch.tensor(self.labels[i])
 def batch(self,ids):return np.stack([self.X[self.starts[i]:self.starts[i]+self.window] for i in ids])

def verify(ds,expected,name,split):
 hl=hashlib.sha256(ds.labels.tobytes()).hexdigest();h=hashlib.sha256()
 for i in range(0,len(ds),32):h.update(ds.batch(range(i,min(i+32,len(ds)))).tobytes())
 hs=h.hexdigest();ok=(hl==expected['labels_sha256'] and hs==expected['samples_sha256'] and len(ds)==expected['windows'])
 records['checks'].append(dict(dataset=name,split=split,windows=len(ds),labels_sha256=hl,samples_sha256=hs,exact_match=ok))
 write();assert ok,f'{name} {split}: historical input hash mismatch'
 log(f'{name} {split}: historical window/label hashes verified ({len(ds)} windows)')

def ton_pack():
 name='ToN-IoT';csv=TON/'iot_fusion_weather_all.csv';log('Reading ToN-IoT and checking local labels')
 df=pd.read_csv(csv,low_memory=False)
 devs=['weather','modbus','thermostat','fridge','garage_door','gps','motion_light']
 classes=sorted(df.final_type.dropna().unique().tolist())
 lc=[('type' if d=='weather' else d+'_type') for d in devs]
 fc=[[c for c in df if c.startswith(d+'_') and not c.endswith(('_label','_type'))] for d in devs]
 for d,cols,label in zip(devs,fc,lc):
  s=df[label];valid=s.notna() & ~s.astype(str).str.lower().isin(['','nan','none'])
  for cl in classes:records['coverage'].append(dict(dataset=name,device=d,attack=cl,rows=int((valid & (s==cl)).sum())))
  vals=df[cols].to_numpy(float);finite=np.isfinite(vals)
  records['quality'].append(dict(dataset=name,device=d,rows=len(df),available_rows=int(valid.sum()),missing_cell_fraction=float(np.isnan(vals).mean()),nonfinite_cell_fraction=float((~finite).mean()),zero_cell_fraction=float((vals==0).mean()),feature_dim=len(cols)))
  records['conflicts'].append(dict(dataset=name,device=d,available_rows=int(valid.sum()),local_global_mismatch=int((valid & (s!=df.final_type)).sum()),mismatch_fraction=float((s[valid]!=df.loc[valid,'final_type']).mean())))
  ct=pd.crosstab(s[valid],df.loc[valid,'final_type']);ct.to_csv(OUT/f'ton_local_global_{d}.csv',encoding='utf-8-sig')
 load_module('analysis_ton_v1',TON/'activity_gated_gat.py');load_module('analysis_ton_v2',TON/'activity_gated_gat_v2.py')
 source=(TON/'_run_cnn.py').read_text(encoding='utf-8-sig')
 source=source.replace("_ROOT = r'E:\\apt\\yaoyao\\second'",f'_ROOT = {str(TON)!r}')
 source=source.replace("_TMP = os.path.join(_ROOT, '_tmp')",f'_TMP = {str(OUT / "_tmp")!r}')
 source=source.replace('from activity_gated_gat import','from analysis_ton_v1 import').replace('from activity_gated_gat_v2 import','from analysis_ton_v2 import')
 m=types.ModuleType('analysis_ton_runner');m.__file__=str(TON/'_run_cnn.py');exec(compile(source,m.__file__,'exec'),m.__dict__);os.chdir(ROOT)
 df.timestamp=pd.to_datetime(df.timestamp);df=df.sort_values('timestamp').reset_index(drop=True)
 valid_types=df.final_type.value_counts();df=df[df.final_type.isin(valid_types[valid_types>10].index)].copy()
 le=LabelEncoder();df['label_encoded']=le.fit_transform(df.final_type)
 parts=m.stratified_timegroup_split(df)
 dims=list(map(len,fc));sc=StandardScaler();datasets=[]
 for i,part in enumerate(parts):
  X=part[sum(fc,[])].to_numpy(np.float32)
  X=sc.fit_transform(X) if i==0 else sc.transform(X)
  X=np.nan_to_num(np.clip(X,-8,8),nan=0).astype(np.float32)
  local=np.stack([part[c].map({x:j for j,x in enumerate(le.classes_)}).fillna(-1).to_numpy(np.int64) for c in lc],axis=1)
  valid=local>=0
  datasets.append(LazyWindows(X,part.label_encoded.to_numpy(),dims,128,1 if i==0 else 2,local=local,valid=valid))
 folder=TON/'cnn_ablation_selected';expected=json.loads((folder/'provenance'/'cnn_feature_ablation_v2_contrastive_dataset.json').read_text(encoding='utf-8-sig'))
 for split,ds in zip(['train','val','test'],datasets):verify(ds,expected[split],name,split)
 manifest['inputs'].append(dict(dataset=name,path=str(csv),sha256=digest(csv),protocol='Existing within-2000-row stratified split; sorted rows then windows. Local label diagnostics; not a strict temporal or unseen-device evaluation.',raw_rows=len(df),classes=le.classes_.tolist(),dims=dims))
 del df;gc.collect();write()
 return name,m,datasets,devs,le.classes_.tolist(),folder

def nb_pack():
 name='N-BaIoT';m=load_module('analysis_nb_runner',ROOT/'run_cnn_align.py');csv=ROOT/'nba_iot_11class.csv'
 log('Reading N-BaIoT metadata; raw statistics stream in bounded chunks')
 meta=pd.read_csv(csv,usecols=['device','final_type','final_label']);le=LabelEncoder();meta['label_encoded']=le.fit_transform(meta.final_type.astype(str))
 devs=list(m.DEVICE_NAMES);classes=le.classes_.tolist()
 tab=pd.crosstab(meta.device,meta.final_type).reindex(index=devs,columns=classes,fill_value=0)
 for d in devs:
  for c in classes:records['coverage'].append(dict(dataset=name,device=d,attack=c,rows=int(tab.loc[d,c])))
 parts=m.stratified_split(meta,42)
 features=m.public_feat_cols(pd.read_csv(csv,nrows=0));allfeatures=m.all_feat_cols(pd.read_csv(csv,nrows=0));cols=len(features)
 cache=OUT/'_cache';cache.mkdir(exist_ok=True);path=cache/'nb_public_float32.dat'
 cached=path.exists() and path.stat().st_size==len(meta)*cols*4
 raw=np.memmap(path,dtype=np.float32,mode='r' if cached else 'w+',shape=(len(meta),cols))
 count={d:dict(cells=0,missing=0,nonfinite=0,zero=0) for d in devs};offset=0
 prior_quality=pd.read_csv(OUT/'quality.csv') if cached and (OUT/'quality.csv').exists() else pd.DataFrame()
 cached=cached and len(prior_quality[prior_quality.dataset==name])==len(devs) if len(prior_quality) else False
 if not cached and raw.mode=='r':
  del raw;raw=np.memmap(path,dtype=np.float32,mode='w+',shape=(len(meta),cols))
 for chunk in ([] if cached else pd.read_csv(csv,chunksize=40000,usecols=allfeatures+['device'])):
  raw[offset:offset+len(chunk)]=chunk[features].to_numpy(np.float32)
  for d,g in chunk.groupby('device'):
   v=g[allfeatures].to_numpy(float);r=count[d];r['cells']+=v.size;r['missing']+=np.isnan(v).sum();r['nonfinite']+=(~np.isfinite(v)).sum();r['zero']+=(v==0).sum()
  offset+=len(chunk)
  if offset%400000==0:log(f'N-BaIoT streamed {offset:,}/{len(meta):,} raw rows')
 if cached:
  records['quality'].extend(prior_quality[prior_quality.dataset==name].to_dict('records'));log('Reusing raw-statistics cache; historical input hashes will still be verified')
 else:
  raw.flush();assert offset==len(meta)
  for d,r in count.items():records['quality'].append(dict(dataset=name,device=d,rows=int((meta.device==d).sum()),available_rows=int((meta.device==d).sum()),missing_cell_fraction=float(r['missing']/r['cells']),nonfinite_cell_fraction=float(r['nonfinite']/r['cells']),zero_cell_fraction=float(r['zero']/r['cells']),feature_dim=len(allfeatures)))
 def aligned(part,seed):
  rng=np.random.RandomState(seed);xs=[];ys=[]
  for lab in sorted(part.label_encoded.unique().tolist()):
   streams=[part.index[(part.label_encoded==lab)&(part.device==d)].to_numpy() for d in devs]
   n=min(min(len(s) for s in streams if len(s)),15000);block=np.zeros((n,len(devs)*cols),np.float32)
   for di,s in enumerate(streams):
    if not len(s):continue
    ix=np.linspace(0,len(s)-1,n).astype(int)
    jitter=max(len(s)//max(n,1),1)
    ix=np.clip(ix+rng.randint(-jitter,jitter+1,size=n),0,len(s)-1)
    block[:,di*cols:(di+1)*cols]=raw[s[ix]]
   xs.append(block);ys.append(np.full(n,lab,np.int64))
  X=np.concatenate(xs);y=np.concatenate(ys);perm=rng.permutation(len(y));return X[perm],y[perm]
                                                                                  
 rng=np.random.RandomState(7108);train_ids=[];test_ids=[]
 for part,ids in [(parts[0],train_ids),(parts[2],test_ids)]:
  for (_, _),g in part.groupby(['device','label_encoded']):ids.extend(rng.choice(g.index,min(240,len(g)),replace=False).tolist())
 train_ids=np.array(train_ids);test_ids=np.array(test_ids)
 scraw=StandardScaler().fit(raw[train_ids]);rtest=scraw.transform(raw[test_ids]);ry=meta.label_encoded.to_numpy()[test_ids];rd=meta.device.map({d:i for i,d in enumerate(devs)}).to_numpy()[test_ids]
 geometry(name,0,'raw_public_100',rtest,ry,rd)
 for prefix in ['MI_dir_','H_','HH_','HpHp_']:
  ix=[i for i,c in enumerate(features) if c.startswith(prefix)];geometry(name,0,'raw_'+prefix.rstrip('_'),rtest[:,ix],ry,rd)
 datasets=[];sc=StandardScaler();dims=[cols]*len(devs)
 present=(tab.to_numpy()>0).T
 for i,part in enumerate(parts):
  X,y=aligned(part,42+i);X=sc.fit_transform(X) if i==0 else sc.transform(X)
  X=np.nan_to_num(np.clip(X,-8,8),nan=0).astype(np.float32)
  valid=present[y];local=np.where(valid,y[:,None],-1)
  datasets.append(LazyWindows(X,y,dims,64,4 if i==0 else 8,12000 if i==0 else 6000,42+i,local,valid))
 folder=ROOT/'cnn_ablation_selected';expected=json.loads((folder/'provenance'/'cnn_feature_ablation_v2_contrastive_dataset.json').read_text(encoding='utf-8-sig'))
 for split,ds in zip(['train','val','test'],datasets):verify(ds,expected[split],name,split)
 manifest['inputs'].append(dict(dataset=name,path=str(csv),sha256=digest(csv),protocol='Existing source-row stratified split before label-assisted alignment; same-category pairing and random row order before windows; not synchronized traffic or chronological windows. Absent blocks standardized exactly as historical model.',raw_rows=len(meta),classes=classes,dims=dims))
 del raw,meta,parts;gc.collect();write()
 return name,m,datasets,devs,classes,folder

def geometry(dataset,seed,stage,X,y,d):
 centers={}
 for cl in np.unique(y):
  for dev in np.unique(d):
   ix=(y==cl)&(d==dev)
   if ix.sum()>=5:
    c=X[ix].mean(0);norm=np.linalg.norm(c)
    if norm>1e-8:centers[(int(cl),int(dev))]=c/norm
 same=[];different=[]
 keys=list(centers)
 for i,a in enumerate(keys):
  for b in keys[i+1:]:
   if a[1]==b[1]:continue
   (same if a[0]==b[0] else different).append(float(centers[a]@centers[b]))
 if same and different:records['geometry'].append(dict(dataset=dataset,seed=seed,stage=stage,same_class_cross_device_cosine=float(np.mean(same)),different_class_cross_device_cosine=float(np.mean(different)),separation_gap=float(np.mean(same)-np.mean(different)),same_pairs=len(same),different_pairs=len(different),min_cell_samples=5))

def sample_windows(ds,n):
 rng=np.random.RandomState(7108);ids=[]
 for c in np.unique(ds.labels):
  pool=np.flatnonzero(ds.labels==c);ids.extend(rng.choice(pool,min(n,len(pool)),False).tolist())
 return np.array(sorted(ids),np.int64)

@torch.inference_mode()
def extract(model,ds,ids,nb):
 arrays={k:[] for k in ['projected','consensus','output','weight','gate','pred']}
 maskmiss=[]
 for st in range(0,len(ids),32):
  ix=ids[st:st+32];x=torch.from_numpy(ds.batch(ix)).to('cuda' if torch.cuda.is_available() else 'cpu');f=model.front
  h=f.project(x);z,hp,shared,g=f.align_nodes(h)
  if nb:
   vote=f.node_mask(h);nv=vote.float().sum(2,keepdim=True).clamp_min(1);hm=h*vote.unsqueeze(-1);mean=hm.sum(2,keepdim=True)/nv.unsqueeze(-1)
   others=(mean*nv.unsqueeze(-1)-hm)/(nv-vote.float()).unsqueeze(-1).clamp_min(1)
   co=F.cosine_similarity(h,others,dim=-1).masked_fill(~vote,-1e9);w=F.softmax(co/f.act_temp.clamp(.2,5),dim=2)*vote
   true=torch.from_numpy(ds.valid[ds.ends[ix]]).to(x.device);maskmiss.append((vote[:,-1]!=true).cpu().numpy())
  else:
   others=(h.sum(2,keepdim=True)-h)/max(1,h.shape[2]-1);co=F.cosine_similarity(h,others,dim=-1);w=F.softmax(co/f.act_temp.clamp(.2,5),dim=2)
  logits=model.head(model.backbone(z.flatten(2)))
  for k,a in [('projected',h[:,-1]),('consensus',hp[:,-1]),('output',z[:,-1]),('weight',w[:,-1]),('gate',g[:,-1].mean(-1)),('pred',logits.argmax(1))]:arrays[k].append(a.cpu().numpy())
 arrays={k:np.concatenate(v) for k,v in arrays.items()}
 assert all(np.isfinite(a).all() for a in arrays.values())
 assert np.allclose(arrays['weight'].sum(1),1,atol=1e-5)
 arrays['local']=ds.local[ds.ends[ids]];arrays['valid']=ds.valid[ds.ends[ids]];arrays['labels']=ds.labels[ids];arrays['window_ids']=ids
 if nb:arrays['model_mask_mismatch']=np.concatenate(maskmiss)
 return arrays

def balanced_node_indices(local,valid,classes,devs,cap,seed):
 rng=np.random.RandomState(seed);indices=[]
 for c in classes:
  pools=[np.flatnonzero((local.reshape(-1)==c)&(np.tile(np.arange(local.shape[1]),len(local))==d)&valid.reshape(-1)) for d in devs]
  n=min(cap,min(map(len,pools)))
  for p in pools:indices.extend(rng.choice(p,n,False).tolist())
 return np.array(indices,np.int64)

def analyze_features(name,seed,train,test,devs):
                                                                                
 ndev=len(devs);common=[]
 for c in np.unique(train['local'][train['valid']]):
  if all(np.sum((a['local'][:,d]==c)&a['valid'][:,d])>=5 for a in [train,test] for d in range(ndev)):common.append(int(c))
 manifest.setdefault('device_probe_common_classes',{})[name]=common
 pi=balanced_node_indices(train['local'],train['valid'],common,range(ndev),120,seed)
 qi=balanced_node_indices(test['local'],test['valid'],common,range(ndev),80,seed)
 for stage in ['projected','consensus','output']:
  xt=train[stage].reshape(-1,train[stage].shape[-1]);xe=test[stage].reshape(-1,test[stage].shape[-1]);yt=train['local'].reshape(-1);ye=test['local'].reshape(-1)
  goodtr=np.flatnonzero(train['valid'].reshape(-1));goodte=np.flatnonzero(test['valid'].reshape(-1))
                                                                                
  supported=sorted(map(int,set(yt[goodtr])&set(ye[goodte])));rng=np.random.RandomState(seed)
  tr=[];te=[]
  nt=min(1000,min(np.sum(yt[goodtr]==c) for c in supported));ne=min(800,min(np.sum(ye[goodte]==c) for c in supported))
  for c in supported:
   tr.extend(rng.choice(goodtr[yt[goodtr]==c],nt,False));te.extend(rng.choice(goodte[ye[goodte]==c],ne,False))
  for target,a,b,ya,yb,chance in [('local_attack',np.array(tr),np.array(te),yt,ye,1/len(supported)),('device_class_balanced',pi,qi,np.tile(np.arange(ndev),len(train['local'])),np.tile(np.arange(ndev),len(test['local'])),1/ndev)]:
   if len(a)==0 or len(np.unique(ya[a]))<2:continue
   clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=800,random_state=seed));clf.fit(xt[a],ya[a]);pred=clf.predict(xe[b])
   records['probes'].append(dict(dataset=name,seed=seed,stage=stage,target=target,macro_f1=float(f1_score(yb[b],pred,average='macro',zero_division=0)),accuracy=float(accuracy_score(yb[b],pred)),chance_accuracy=chance,train_nodes=len(a),test_nodes=len(b),classes=json.dumps(common if target.startswith('device') else supported),converged=bool(clf[-1].n_iter_.max()<800)))
  d=np.tile(np.arange(ndev),len(test['local']));geometry(name,seed,stage,xe[goodte],ye[goodte],d[goodte])
 for di,dev in enumerate(devs):
  for cl in np.unique(test['local'][:,di]):
   valid=(test['local'][:,di]==cl)&test['valid'][:,di]
   if not valid.any():continue
   records['weights'].append(dict(dataset=name,seed=seed,device=dev,local_class=int(cl),nodes=int(valid.sum()),mean_weight=float(test['weight'][valid,di].mean()),mean_gate=float(test['gate'][valid,di].mean())))
 records['checks'].append(dict(dataset=name,seed=seed,split='extracted_test',windows=len(test['labels']),exact_match=True,unavailable_weight=float((test['weight']*(~test['valid'])).sum(1).mean()),model_mask_mismatch=float(test.get('model_mask_mismatch',np.zeros(1)).mean())))

def historical_predictions(name,folder,ds,classes):
 selection=json.loads((folder/'selection.json').read_text(encoding='utf-8-sig'));config={r['variant']:r for r in selection}
 for seed in [config['front']['seed']]:
  full=folder/'front_predictions.npz'
  if not full.exists():continue
  p=np.load(full);y=p['labels'];assert np.array_equal(y,ds.labels)
  fp=p['predictions'];fb=f1_score(y,fp,average=None,labels=np.arange(len(classes)),zero_division=0)
  for v in ['front','no_domain','no_contrastive','uniform_weights','fixed_fusion']:
   path=folder/f'{v}_predictions.npz'
   if not path.exists():continue
   q=np.load(path);assert np.array_equal(q['labels'],y);yp=q['predictions']
   vs=config[v]['seed'];paired=vs==seed;gate_same=config[v]['gate_target']==config['front']['gate_target']
   records['classification'].append(dict(dataset=name,seed=vs,variant=v,accuracy=float(accuracy_score(y,yp)),macro_f1=float(f1_score(y,yp,average='macro',zero_division=0)),same_seed_as_full=paired,same_gate_target=gate_same,source=str(path)))
   if v!='front':
    score=f1_score(y,yp,average=None,labels=np.arange(len(classes)),zero_division=0)
    for i,c in enumerate(classes):records['class_gains'].append(dict(dataset=name,seed=seed,ablation_seed=vs,same_seed=paired,same_gate_target=gate_same,attack=c,comparison=v,full_f1=float(fb[i]),ablation_f1=float(score[i]),gain=float(fb[i]-score[i]),test_windows=int((y==i).sum())))

def run(pack):
 name,m,datasets,devs,classes,folder=pack;train,val,test=datasets
 historical_predictions(name,folder,test,classes)
 ti=sample_windows(train,240);ei=sample_windows(test,180)
 config=json.loads((folder/'selection.json').read_text(encoding='utf-8-sig'));seed=next(r['seed'] for r in config if r['variant']=='front')
 for seed in [seed]:
  cp=folder/'front.pth';log(f'{name} retained seed {seed}: extracting component stages and fitting independent probes')
  saved=torch.load(cp,map_location='cpu',weights_only=False)
  model=m.PlugCNN(train.device_dims,len(classes),use_front=True).to('cuda' if torch.cuda.is_available() else 'cpu');model.load_state_dict(saved['model_state'],strict=True);model.eval()
  manifest['checkpoints'].append(dict(dataset=name,seed=seed,path=str(cp),sha256=digest(cp),strict_state_load=True))
  a=extract(model,train,ti,name=='N-BaIoT');b=extract(model,test,ei,name=='N-BaIoT')
  prediction_file=folder/'front_predictions.npz'
  if prediction_file.exists():
   old=np.load(prediction_file)['predictions'][ei];assert np.array_equal(b['pred'],old),f'{name}: checkpoint predictions differ from historical outputs'
  np.savez_compressed(OUT/f'{name.replace("-","").lower()}_s{seed}_features.npz',**{'train_'+k:v for k,v in a.items()},**{'test_'+k:v for k,v in b.items()})
  analyze_features(name,seed,a,b,devs);interventions(name,seed,model,test,ei,b);write();del model,a,b;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 log(f'{name}: retained model diagnostic completed')

@torch.inference_mode()
def interventions(name,seed,model,ds,ids,baseline):
 f=model.front;oldcal=f.calibrate_cross_protocol;oldalign=f.align_nodes
 def no_consensus(h):return h,torch.zeros_like(h[:,:,:1]),torch.zeros_like(h)
 def no_graph(h):
  hp,shared,g=oldcal(h);return hp,hp,shared,g
 def available_only(h):
  valid=f._analysis_available.to(h.device);nv=valid.float().sum(2,keepdim=True).clamp_min(1);hm=h*valid.unsqueeze(-1)
  others=(hm.sum(2,keepdim=True)-hm)/(nv-valid.float()).unsqueeze(-1).clamp_min(1)
  co=F.cosine_similarity(h,others,dim=-1).masked_fill(~valid,-1e9)
  w=F.softmax(co/f.act_temp.clamp(.2,5),dim=2)*valid;shared=(h*w.unsqueeze(-1)).sum(2,keepdim=True)
  g=f.dim_gate(h)*valid.unsqueeze(-1);hp=(1-g)*h+g*shared
  return hp,shared,g
 for variant in ['full','bypass_consensus','bypass_graph']+(['availability_masked_consensus'] if name=='ToN-IoT' else []):
  f.calibrate_cross_protocol=no_consensus if variant=='bypass_consensus' else available_only if variant=='availability_masked_consensus' else oldcal
  f.align_nodes=no_graph if variant=='bypass_graph' else oldalign
  if variant=='full':pred=baseline['pred']
  else:
   p=[]
   for st in range(0,len(ids),32):
    ix=ids[st:st+32];x=torch.from_numpy(ds.batch(ix)).to(next(model.parameters()).device)
    f._analysis_available=torch.from_numpy(np.stack([ds.valid[ds.starts[i]:ds.starts[i]+ds.window] for i in ix]))
    p.append(model(x).argmax(1).cpu().numpy())
   pred=np.concatenate(p)
  y=ds.labels[ids];records['interventions'].append(dict(dataset=name,seed=seed,variant=variant,windows=len(ids),accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),prediction_changes=int((pred!=baseline['pred']).sum()),protocol='Inference-only intervention on same fixed checkpoint, class-stratified test subset; not a retrained ablation'))
 f.calibrate_cross_protocol=oldcal;f.align_nodes=oldalign

def figures():
 coverage=pd.DataFrame(records['coverage']);fig,axes=plt.subplots(1,2,figsize=(17,6))
 for ax,name in zip(axes,['ToN-IoT','N-BaIoT']):
  p=coverage[coverage.dataset==name].pivot(index='device',columns='attack',values='rows');im=ax.imshow(np.log10(p.values+1),aspect='auto',cmap='Blues');ax.set_xticks(range(len(p.columns)),p.columns,rotation=55,ha='right');ax.set_yticks(range(len(p.index)),p.index);ax.set_xlabel('Local class');ax.set_ylabel('Device');ax.set_title(name+' original device / class coverage')
  for i in range(len(p)):
   for j in range(len(p.columns)):ax.text(j,i,str(int(p.iloc[i,j])),ha='center',va='center',fontsize=7,color='white' if im.get_array()[i,j]>im.get_clim()[1]*.7 else 'black')
  fig.colorbar(im,ax=ax,label='log10(original rows + 1)',shrink=.7)
 fig.tight_layout();fig.savefig(OUT/'01_device_attack_coverage.png',dpi=180);fig.savefig(OUT/'01_device_attack_coverage.pdf');plt.close(fig)
 probes=pd.DataFrame(records['probes']);geo=pd.DataFrame(records['geometry']);stages=['projected','consensus','output']
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for col,name in enumerate(['ToN-IoT','N-BaIoT']):
  for target,label in [('local_attack','Local attack macro-F1'),('device_class_balanced','Class-balanced device macro-F1')]:
   p=probes[(probes.dataset==name)&(probes.target==target)].groupby('stage').macro_f1.agg(['mean','std']).reindex(stages)
   axes[0,col].plot(range(3),p['mean'],marker='o',label=label)
  axes[0,col].set_title(name+' linear probes (retained full checkpoint)');axes[0,col].set_xticks(range(3),stages);axes[0,col].set_ylim(0,1.05);axes[0,col].set_ylabel('Macro-F1 (0-1)');axes[0,col].set_xlabel('Component stage');axes[0,col].legend(fontsize=8)
  for metric,label in [('same_class_cross_device_cosine','Same local class'),('different_class_cross_device_cosine','Different local classes'),('separation_gap','Same-minus-different gap')]:
   p=geo[(geo.dataset==name)&(geo.seed!=0)].groupby('stage')[metric].agg(['mean','std']).reindex(stages)
   axes[1,col].plot(range(3),p['mean'],marker='o',label=label)
  axes[1,col].set_xticks(range(3),stages);axes[1,col].set_ylabel('Centroid cosine / gap (unitless)');axes[1,col].set_xlabel('Component stage');axes[1,col].legend(fontsize=8);axes[1,col].set_title('Equal weight per supported cross-device centroid pair')
 fig.tight_layout();fig.savefig(OUT/'02_component_stage_diagnostics.png',dpi=180);fig.savefig(OUT/'02_component_stage_diagnostics.pdf');plt.close(fig)
 w=pd.DataFrame(records['weights']);fig,axes=plt.subplots(1,2,figsize=(15,5))
 for ax,name in zip(axes,['ToN-IoT','N-BaIoT']):
  p=w[w.dataset==name].groupby(['device','local_class']).mean(numeric_only=True).mean_weight.unstack();im=ax.imshow(p,aspect='auto',cmap='Blues');ax.set_xticks(range(len(p.columns)),p.columns);ax.set_yticks(range(len(p)),p.index);ax.set_xlabel('Local class ID (see manifest for names)');ax.set_ylabel('Device');ax.set_title(name+' consensus weights (retained model)');fig.colorbar(im,ax=ax,label='Mean normalized vote weight')
 fig.tight_layout();fig.savefig(OUT/'03_consensus_weights.png',dpi=180);fig.savefig(OUT/'03_consensus_weights.pdf');plt.close(fig)

if __name__=='__main__':
 log('Starting reproducible two-dataset diagnostics; existing models only')
 run(ton_pack());gc.collect();run(nb_pack());figures();write()
 log('COMPLETE: '+str(OUT))
