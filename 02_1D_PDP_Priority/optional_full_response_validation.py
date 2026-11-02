import json,os,time
from pathlib import Path
import numpy as np,pandas as pd,xgboost as xgb
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr
OUT=Path('validation/data');MD=Path('validation/models');D=pd.read_csv('validation_inputs/modeling_data_12_factor_comparison.csv');F=['Temperature','DEM','Slope','GDP','POP','NL','PD','SHDI','CONTAG','Area_mn'];S=['GDP','PD','NL','POP','CONTAG','Area_mn','SHDI'];X=D[F].to_numpy(np.float32);y=D['GP-WY TOP'].to_numpy();N=len(y)
rng=np.random.default_rng(773);ref=rng.choice(N,2048,replace=False);background=rng.choice(N,384,replace=False);bg=X[background];xx=X[ref];g=pd.read_csv(OUT/'spatial_blocks.csv').block_50km.to_numpy();G=int(g.max()+1)
np.savez_compressed(OUT/'fixed_reference_samples.npz',reference=ref,background=background)
params=json.load(open(OUT/'final_parameters.json'));model=xgb.XGBRegressor();model.load_model(MD/'final_spatial_validated.ubj');model.set_params(n_jobs=4)
# ALE implementation uses quantile bins and observed samples, centered by bin mass.
def ale(m,j,edges,weights=None):
 idx=np.clip(np.searchsorted(edges,xx[:,j],side='right')-1,0,len(edges)-2);lo=xx.copy();hi=xx.copy();lo[:,j]=edges[idx];hi[:,j]=edges[idx+1];diff=m.predict(hi)-m.predict(lo);w=np.ones(len(xx)) if weights is None else weights
 mass=np.bincount(idx,weights=w,minlength=len(edges)-1);eff=np.bincount(idx,weights=w*diff,minlength=len(edges)-1)/np.maximum(mass,1e-12);curve=np.cumsum(eff)-eff/2;curve-=np.sum(curve*mass)/mass.sum();return (edges[:-1]+edges[1:])/2,curve,mass
# Breakpoint algorithm: piecewise linear approximation with optional discontinuities.
# Grid-interleaved CV assesses reconstruction, not an independent statistical test.
class Breaks:
 def __init__(self,obs,lo,hi):
  self.lo,self.hi=lo,hi;self.span=hi-lo;self.u=np.linspace(0,1,61);u=self.u;self.models=[]
  base=[(np.c_[np.ones(61),u],()),(np.c_[np.ones(61),u,u*u,u*u*u],())]
  knots=np.arange(3,58,2)/60
  valid=[t for t in knots if min(np.mean(obs<lo+t*self.span),np.mean(obs>=lo+t*self.span))>=.05 and np.sum(abs(obs-(lo+t*self.span))<=.02*self.span)>=50]
  for t in valid:base.append((np.c_[np.ones(61),u,(u>t),np.maximum(u-t,0)],(t,)))
  for i,t in enumerate(valid):
   for q in valid[i+1:]:
    if q-t<.10 or np.mean((obs>=lo+t*self.span)&(obs<lo+q*self.span))<.05:continue
    base.append((np.c_[np.ones(61),u,(u>t),np.maximum(u-t,0),(u>q),np.maximum(u-q,0)],(t,q)))
  self.cvops=[]
  for A,ts in base:
   H=np.zeros((61,61))
   for k in range(5):
    te=np.arange(61)%5==k;tr=~te;H[np.ix_(te,tr)]=A[te]@np.linalg.pinv(A[tr])
   self.cvops.append(H-np.eye(61));self.models.append((A,ts,np.linalg.pinv(A)))
  self.cvops=np.stack(self.cvops)
 def select(self,x,curve,min_jump=.005,gain=.20):
  v=np.interp(self.lo+self.span*self.u,x,curve);err=np.mean(np.einsum('mij,j->mi',self.cvops,v)**2,axis=1);best0=int(np.argmin(err[:2]));baseerr=err[best0];eligible=[]
  for k,(A,ts,pinv) in enumerate(self.models[2:],2):
   coef=pinv@v;jumps=abs(coef[[2] if len(ts)==1 else [2,4]])
   if min(jumps)<min_jump:continue
   if err[k] < (1-gain)*baseerr:eligible.append(k)
  if not eligible:return [],float(baseerr),float(baseerr),best0
  # Favor fewer breaks unless a second improves reconstruction RMSE by >=10%.
  one=[k for k in eligible if len(self.models[k][1])==1];two=[k for k in eligible if len(self.models[k][1])==2]
  b1=min(one,key=lambda k:err[k]) if one else None;b2=min(two,key=lambda k:err[k]) if two else None
  best=b2 if b1 is None else b1
  if b1 is not None and b2 is not None and err[b2]<.81*err[b1]:best=b2
  return [self.lo+t*self.span for t in self.models[best][1]],float(baseerr),float(err[best]),best
curves={};detectors={};edges={};main=[];sensitivity=[]
for f in S:
 j=F.index(f);lo,hi=np.quantile(X[:,j],[.01,.99]);grid=np.linspace(lo,hi,81);mat=np.tile(bg,(len(grid),1));mat[:,j]=np.repeat(grid,len(bg));ice=model.predict(mat).reshape(len(grid),len(bg));pdp=ice.mean(1);ed=np.unique(np.quantile(X[:,j],np.linspace(0,1,21)));edges[f]=ed;ac,ay,mass=ale(model,j,ed)
 det=Breaks(X[:,j],lo,hi);detectors[f]=det;knots,b0,b1,_=det.select(grid,pdp);curves[f]=(grid,pdp,ice,ac,ay,mass,knots)
 centered=pdp-np.mean(np.interp(X[:,j],grid,pdp));ai=np.interp(grid,ac,ay);rho=float(spearmanr(pdp,ai).statistic)
 main.append(dict(feature=f,PDP_endpoint_change=float(pdp[-1]-pdp[0]),ALE_endpoint_change=float(ay[-1]-ay[0]),shape_spearman=rho,break_count=len(knots),breaks=';'.join(map(str,knots)),no_break_CV_MSE=b0,selected_CV_MSE=b1))
 for bins in [10,20,30]:
  ec=np.unique(np.quantile(X[:,j],np.linspace(0,1,bins+1)));aaa,bbb,cc=ale(model,j,ec);sensitivity.append(dict(feature=f,setting='ALE_bins',value=bins,endpoint_change=float(bbb[-1]-bbb[0]),shape_vs_PDP=float(spearmanr(np.interp(aaa,grid,pdp),bbb).statistic)))
 for ngrid in [41,81,121]:
  # Interpolation sensitivity only; explicitly separate from fresh model predictions.
  gg=np.linspace(lo,hi,ngrid);kk,*_=det.select(gg,np.interp(gg,grid,pdp));sensitivity.append(dict(feature=f,setting='PDP_grid_resampling',value=ngrid,breaks=';'.join(map(str,kk))))
 print('RESP',f,'breaks',knots,'ALErho',rho,flush=True)
pd.DataFrame(main).to_csv(OUT/'PDP_ALE_comparison.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'curve_settings_sensitivity.csv',index=False)
np.savez_compressed(OUT/'main_response_curves.npz',**{f+'_'+k:v for f,(grid,pdp,ice,ac,ay,mass,knots) in curves.items() for k,v in [('grid',grid),('PDP',pdp),('ICE',ice),('ALE_x',ac),('ALE',ay),('ALE_mass',mass),('breaks',np.array(knots))]})
# Full-data SHAP for explicitly defined zoning scores; no claim of independent validation.
sh=model.get_booster().predict(xgb.DMatrix(X),pred_contribs=True)[:,:-1];np.savez_compressed(OUT/'final_SHAP.npz',values=sh)
imp=abs(sh).mean(0);pd.DataFrame({'feature':F,'mean_abs_SHAP':imp,'percent':imp/imp.sum()*100}).sort_values('mean_abs_SHAP',ascending=False).to_csv(OUT/'full_sample_SHAP_importance.csv',index=False)
# Block-bootstrap models: no early stopping/tuning on evaluated response curves.
bootrows=[];bootP={f:[] for f in S};bootA={f:[] for f in S};bootarea=[];stability=np.zeros((len(ref),len(S)),int);eligible=np.zeros((len(ref),len(S)),int)
for b in range(100):
 brng=np.random.default_rng(3100+b);counts=np.bincount(brng.integers(0,G,G),minlength=G);w=counts[g];tr=w>0
 m=xgb.XGBRegressor(**(params|{'random_state':3100+b}));m.fit(X[tr],y[tr],sample_weight=w[tr]);m.save_model(MD/f'bootstrap_{b:03}.ubj')
 shp=m.get_booster().predict(xgb.DMatrix(xx),pred_contribs=True)[:,:-1];refw=w[ref].astype(float)
 for k,f in enumerate(S):
  j=F.index(f);grid=curves[f][0];mat=np.tile(bg,(len(grid),1));mat[:,j]=np.repeat(grid,len(bg));p=m.predict(mat).reshape(len(grid),len(bg)).mean(1);ac,ay,mass=ale(m,j,edges[f],refw);knots,b0,b1,_=detectors[f].select(grid,p)
  bootP[f].append(p);bootA[f].append(ay);bootrows.append(dict(replicate=b,feature=f,break_count=len(knots),breaks=';'.join(map(str,knots)),PDP_change=float(p[-1]-p[0]),ALE_change=float(ay[-1]-ay[0])))
  # Model-resampling zoning check on the fixed 2048-cell reference sample only.
  score=np.interp(xx[:,j],grid,p);pos=np.maximum(shp[:,j],0);posv=pos[pos>0];high=score>=np.quantile(score,.75);hp=(pos>=np.quantile(posv,.75))&(pos>0) if len(posv) else np.zeros(len(pos),bool);hy=y[ref]>=np.quantile(y,.75);z=high.astype(int)+hp+hy
  stability[:,k]+=z==3;eligible[:,k]+=1;bootarea.append(dict(replicate=b,feature=f,reference_n=len(ref),priority_I_fraction=float(np.mean(z==3)),priority_II_fraction=float(np.mean(z==2))))
 if b%10==0:
  pd.DataFrame(bootrows).to_csv(OUT/'bootstrap_thresholds.csv',index=False);print('BOOT',b,flush=True)
pd.DataFrame(bootrows).to_csv(OUT/'bootstrap_thresholds.csv',index=False);pd.DataFrame(bootarea).to_csv(OUT/'bootstrap_zoning_reference_sample.csv',index=False)
pd.DataFrame({'ID':D.ID.iloc[ref].to_numpy(),**{f+'_PriorityI_frequency':stability[:,k]/eligible[:,k] for k,f in enumerate(S)}}).to_csv(OUT/'reference_sample_priority_frequencies.csv',index=False)
np.savez_compressed(OUT/'bootstrap_response_curves.npz',**{f+'_PDP':np.array(bootP[f]) for f in S},**{f+'_ALE':np.array(bootA[f]) for f in S})
# Match bootstrap breakpoints to each reference breakpoint; absent breaks remain absent.
summary=[];bt=pd.DataFrame(bootrows)
for f in S:
 grid=curves[f][0];span=np.ptp(grid);sub=bt[bt.feature==f];mainknots=curves[f][-1]
 if not mainknots:summary.append(dict(feature=f,reference_break=np.nan,detection_rate=float(np.mean(sub.break_count>0)),status='no reference breakpoint selected'));continue
 for t in mainknots:
  hits=[]
  for row in sub.itertuples():
   ts=[] if not row.breaks else [float(a) for a in str(row.breaks).split(';')]
   if ts:
    closest=min(ts,key=lambda a:abs(a-t))
    if abs(closest-t)<=.10*span:hits.append(closest)
  rate=len(hits)/100;ci=np.quantile(hits,[.025,.5,.975]) if hits else [np.nan]*3;stable=rate>=.8 and (ci[2]-ci[0])<=.1*span
  summary.append(dict(feature=f,reference_break=t,detection_rate=rate,conditional_low=ci[0],conditional_median=ci[1],conditional_high=ci[2],status='operationally stable candidate' if stable else 'unstable candidate',rule='>=80% detection; conditional interval <=10% displayed range'))
pd.DataFrame(summary).to_csv(OUT/'threshold_stability_summary.csv',index=False)
print('PHASE2_DONE',flush=True)
