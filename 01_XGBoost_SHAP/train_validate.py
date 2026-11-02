import os,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,xgboost as xgb
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.model_selection import KFold,GroupKFold,GroupShuffleSplit,train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
ROOT=Path('validation');OUT=ROOT/'data';MODELS=ROOT/'models';SEED=42
D=pd.read_csv('validation_inputs/modeling_data_12_factor_comparison.csv');F=['Temperature','DEM','Slope','GDP','POP','NL','PD','SHDI','CONTAG','Area_mn'];S=['GDP','PD','NL','POP','CONTAG','Area_mn','SHDI'];X=D[F].to_numpy(np.float32);y=D['GP-WY TOP'].to_numpy();N=len(y)
BASE=dict(learning_rate=.04,subsample=.85,colsample_bytree=.9,min_child_weight=5,reg_lambda=5,objective='reg:squarederror',tree_method='hist',n_jobs=4)
# Models and all configuration choices are compared on identical outer partitions.
CAND=[dict(max_depth=3,min_child_weight=10,reg_lambda=10),dict(max_depth=5,min_child_weight=5,reg_lambda=5),dict(max_depth=7,min_child_weight=10,reg_lambda=10)]
def groups(size):
 bx=np.floor((D.x-D.x.min())/(size*1000)).astype(int);by=np.floor((D.y_coord-D.y_coord.min())/(size*1000)).astype(int)
 return pd.factorize(list(zip(bx,by)))[0]
def metric(a,p):return dict(R2=float(r2_score(a,p)),RMSE=float(np.sqrt(mean_squared_error(a,p))),MAE=float(mean_absolute_error(a,p)))
def fit_select(xx,yy,tr,g,seed=42,tune=True):
 if g is None:a,b=train_test_split(tr,test_size=.2,random_state=seed)
 else:
  aa,bb=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(tr,groups=g[tr]));a,b=tr[aa],tr[bb]
 logs=[]
 for k,c in enumerate(CAND if tune else [CAND[1]]):
  m=xgb.XGBRegressor(**(BASE|c),n_estimators=1000,early_stopping_rounds=40,random_state=seed);m.fit(xx[a],yy[a],eval_set=[(xx[b],yy[b])],verbose=False)
  logs.append(dict(candidate=k,params=c,trees=int(m.best_iteration+1),validation=metric(yy[b],m.predict(xx[b]))))
 chosen=min(logs,key=lambda a:a['validation']['RMSE']);m=xgb.XGBRegressor(**(BASE|chosen['params']),n_estimators=chosen['trees'],random_state=seed);m.fit(xx[tr],yy[tr]);return m,chosen,logs
# audit + dependency proxies
D[F].corr(method='spearman').to_csv(OUT/'predictor_spearman.csv');D[F].describe(percentiles=[.01,.05,.5,.95,.99]).to_csv(OUT/'input_summary.csv')
metrics=[];tuning=[];predictions={};g50=groups(50);pd.DataFrame({'ID':D.ID,'block_30km':groups(30),'block_50km':g50,'block_80km':groups(80)}).to_csv(OUT/'spatial_blocks.csv',index=False)
for kind,g in [('random',None),('block50',g50)]:
 for repeat,seed in enumerate([42,137]):
  splits=KFold(5,shuffle=True,random_state=seed).split(X) if g is None else GroupKFold(5,shuffle=True,random_state=seed).split(X,groups=g)
  pp=np.full(N,np.nan);ff=np.zeros(N,int)
  for fold,(tr,te) in enumerate(splits):
   m,b,logs=fit_select(X,y,tr,g,seed+fold,tune=True);p=m.predict(X[te]);pp[te]=p;ff[te]=fold
   metrics.append(dict(design=kind,repeat=repeat,fold=fold,train_n=len(tr),test_n=len(te),trees=b['trees'],**b['params'],**metric(y[te],p)))
   tuning.append(dict(design=kind,repeat=repeat,fold=fold,search=logs))
   m.save_model(MODELS/f'{kind}_r{repeat}_f{fold}.ubj');print('CV',kind,repeat,fold,metrics[-1]['R2'],flush=True)
  predictions[f'{kind}_r{repeat}']=pp;pd.DataFrame({'ID':D.ID,'fold':ff,'observed':y,'predicted':pp,'residual':y-pp}).to_csv(OUT/f'oof_{kind}_r{repeat}.csv',index=False)
# Spatial scale sensitivity, validation-driven single central specification, no tuning on test.
for size in [30,80]:
 g=groups(size);pp=np.full(N,np.nan)
 for fold,(tr,te) in enumerate(GroupKFold(5,shuffle=True,random_state=42).split(X,groups=g)):
  m,b,logs=fit_select(X,y,tr,g,42+fold,tune=False);p=m.predict(X[te]);pp[te]=p
  metrics.append(dict(design=f'block{size}_fixed_spec',repeat=0,fold=fold,train_n=len(tr),test_n=len(te),trees=b['trees'],**b['params'],**metric(y[te],p)))
 predictions[f'block{size}']=pp;print('SCALE',size,metric(y,pp),flush=True)
# Same spatial folds for factor-set sensitivity. Removing direct PRE/NDVI is not full independence.
variants={'12factor':list(F)+['NDVI','Precipitation'],'socio_landscape':[f for f in F if f not in ['Temperature','DEM','Slope']],'socio_only':['GDP','POP','NL'],'landscape_only':['PD','SHDI','CONTAG','Area_mn']}
for name,features in variants.items():
 xx=D[features].to_numpy(np.float32);pp=np.full(N,np.nan)
 for fold,(tr,te) in enumerate(GroupKFold(5,shuffle=True,random_state=42).split(xx,groups=g50)):
  # Match outer-fold main-model configuration to isolate variable-set changes.
  b=next(a for a in metrics if a['design']=='block50' and a['repeat']==0 and a['fold']==fold)
  par=BASE|{k:b[k] for k in ['max_depth','min_child_weight','reg_lambda']};m=xgb.XGBRegressor(**par,n_estimators=b['trees'],random_state=42+fold);m.fit(xx[tr],y[tr]);pp[te]=m.predict(xx[te])
  metrics.append(dict(design=name,repeat=0,fold=fold,train_n=len(tr),test_n=len(te),trees=b['trees'],**metric(y[te],pp[te])))
  m.save_model(MODELS/f'{name}_f{fold}.ubj')
 predictions[name]=pp;print('VARIANT',name,metric(y,pp),flush=True)
pd.DataFrame(metrics).to_csv(OUT/'cv_metrics.csv',index=False);json.dump(tuning,open(OUT/'inner_tuning.json','w'),indent=2)
# Out-of-fold residual Moran's I; symmetrized binary 8-nearest-neighbor weights.
xy=D[['x','y_coord']].to_numpy();nei=cKDTree(xy).query(xy,k=9)[1][:,1:];a=np.repeat(np.arange(N),8);b=nei.ravel();edges=np.unique(np.sort(np.c_[a,b],axis=1),axis=0);ii,jj=edges.T
rng=np.random.default_rng(42);moran=[]
for name,p in predictions.items():
 e=y-p;e=e-e.mean();den=e@e;I=float(N/len(ii)*np.sum(e[ii]*e[jj])/den);null=[]
 for k in range(199):
  v=rng.permutation(e);null.append(float(N/len(ii)*np.sum(v[ii]*v[jj])/den))
 pv=(1+sum(abs(np.array(null)+1/(N-1))>=abs(I+1/(N-1))))/200
 moran.append(dict(design=name,Moran_I=I,permutation_p_two_sided=pv,permutations=199,weights='symmetrized binary 8NN',**metric(y,p)))
pd.DataFrame(moran).to_csv(OUT/'oof_residual_moran.csv',index=False)
# Choose a representative training configuration using validation performance, not shapes.
main=pd.DataFrame(metrics).query("design=='block50'");counts=main.groupby(['max_depth','min_child_weight','reg_lambda']).size();config=counts.idxmax();trees=int(main.query('max_depth==@config[0]').trees.median()) if False else int(main[main.max_depth==config[0]].trees.median())
params=BASE|dict(max_depth=int(config[0]),min_child_weight=int(config[1]),reg_lambda=float(config[2]),n_estimators=trees,random_state=42)
model=xgb.XGBRegressor(**params);model.fit(X,y);model.save_model(MODELS/'final_spatial_validated.ubj');json.dump(params,open(OUT/'final_parameters.json','w'),indent=2)
print('PHASE1_DONE',params,flush=True)
