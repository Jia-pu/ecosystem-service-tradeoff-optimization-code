"""Core PDP/support/zoning calculations extracted from archived workflow.
Uses the same feature order, background seeds, grids and Q75 definitions.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/data'
FEATURES=['Temperature','DEM','Slope','GDP','POP','NL','PD','SHDI','CONTAG','Area_mn']
SINGLE=['GDP','PD','NL','POP','CONTAG','Area_mn','SHDI']
PAIRS=[('GDP','PD'),('POP','NL'),('PD','SHDI'),('SHDI','CONTAG'),('PD','CONTAG')]
TRIPLES=[('PD','SHDI','CONTAG'),('GDP','PD','CONTAG')]

def load():
    d=pd.read_csv(ROOT/'validation_inputs/modeling_data_12_factor_comparison.csv')
    x=d[FEATURES].to_numpy(np.float32)
    model=xgb.XGBRegressor()
    model.load_model(ROOT/'validation/models/final_spatial_validated.ubj')
    model.set_params(n_jobs=4)
    return d,x,d['GP-WY TOP'].to_numpy(),model

def shap_values(model,x):
    values=model.get_booster().predict(xgb.DMatrix(x),pred_contribs=True)[:,:-1]
    np.savez_compressed(OUT/'final_SHAP.npz',values=values)
    imp=np.abs(values).mean(0)
    pd.DataFrame({'feature':FEATURES,'mean_abs_SHAP':imp,'percent':imp/imp.sum()*100}).sort_values('mean_abs_SHAP',ascending=False).to_csv(OUT/'full_sample_SHAP_importance.csv',index=False)
    return values

def support(d,comb,points):
    bins=20 if len(comb)==2 else 6
    ed=[np.unique(np.quantile(d[f],np.linspace(0,1,bins+1))) for f in comb]
    idx=[np.clip(np.searchsorted(e,d[f],side='right')-1,0,len(e)-2) for e,f in zip(ed,comb)]
    counts=np.zeros(tuple(len(e)-1 for e in ed),int)
    np.add.at(counts,tuple(idx),1)
    pi=[np.clip(np.searchsorted(e,points[:,j],side='right')-1,0,len(e)-2) for j,e in enumerate(ed)]
    mass=counts[tuple(pi)]
    inside=np.ones(len(points),bool)
    for j,e in enumerate(ed):
        inside&=(points[:,j]>=e[0])&(points[:,j]<=e[-1])
    return inside&(mass>=30),mass

def surface(d,model,comb,bg,ngrid):
    axes=[np.linspace(*np.quantile(d[f],[.01,.99]),ngrid) for f in comb]
    pts=np.array(list(itertools.product(*axes)),np.float32)
    out=[]
    for chunk in np.array_split(pts,max(1,len(pts)//150+1)):
        a=np.tile(bg,(len(chunk),1))
        for j,f in enumerate(comb):
            a[:,FEATURES.index(f)]=np.repeat(chunk[:,j],len(bg))
        out.extend(model.predict(a).reshape(len(chunk),len(bg)).mean(1))
    return axes,np.array(out).reshape([ngrid]*len(comb))

def classify(score,ok,pos,y,q=.75):
    if not ok.any():
        return np.full(len(score),-1,dtype=int)
    high=(score>=np.quantile(score[ok],q))&ok
    positive=pos[(pos>0)&ok]
    hp=(pos>=np.quantile(positive,q))&(pos>0)&ok if len(positive) else np.zeros(len(pos),bool)
    hy=y>=np.quantile(y,q)
    k=high.astype(int)+hp+hy
    return np.where(ok,np.where(k==3,2,np.where(k==2,1,0)),-1)

def calculate_priority(dimension):
    d,x,y,model=load()
    shap=np.load(OUT/'final_SHAP.npz')['values']
    if shap.shape != x.shape:
        raise ValueError('SHAP shape does not match input feature matrix')
    rng=np.random.default_rng(992)
    bg=x[rng.choice(len(d),128,replace=False)]
    combos=[(f,) for f in SINGLE] if dimension==1 else PAIRS if dimension==2 else TRIPLES
    arrays={};tables=[];coverage=[]
    out={'ID':d.ID,'x':d.x,'y_coord':d.y_coord,'TOP':y}
    curves=np.load(OUT/'main_response_curves.npz') if dimension==1 else None
    for comb in combos:
        name='__'.join(comb)
        obs=d[list(comb)].to_numpy()
        if dimension==1:
            grid=curves[comb[0]+'_grid']
            v=np.interp(obs[:,0],grid,curves[comb[0]+'_PDP'])
            ok=(obs[:,0]>=grid.min())&(obs[:,0]<=grid.max())
        else:
            axes,vals=surface(d,model,comb,bg,25 if dimension==2 else 13)
            v=RegularGridInterpolator(axes,vals,bounds_error=False,fill_value=np.nan)(obs)
            ok,mass=support(d,comb,obs)
            ok&=np.isfinite(v)
            pts=np.array(list(itertools.product(*axes)))
            pok,pmass=support(d,comb,pts)
            tables.append(pd.DataFrame({**{f:pts[:,j] for j,f in enumerate(comb)},'PDP':vals.ravel(),'supported':pok,'local_bin_n':pmass,'combination':name}))
            arrays.update({name+'_axis'+str(j):a for j,a in enumerate(axes)})
            arrays[name+'_PDP']=vals
        pos=np.maximum(shap[:,[FEATURES.index(f) for f in comb]],0).sum(1)
        out[name+'_zone']=classify(v,ok,pos,y)
        out[name+'_PDP_score']=v
        out[name+'_SHAPpos']=pos
        out[name+'_supported']=ok
        coverage.append({'combination':name,'dimension':dimension,'supported_cells':int(ok.sum()),'supported_fraction':float(ok.mean())})
    pd.DataFrame(out).to_csv(OUT/f'priority_zones_{dimension}d.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/f'joint_support_{dimension}d.csv',index=False)
    if arrays:
        np.savez_compressed(OUT/f'joint_surfaces_{dimension}d.npz',**arrays)
        pd.concat(tables).to_csv(OUT/f'joint_PDP_surfaces_{dimension}d.csv',index=False)
    return d,pd.DataFrame(out),arrays
