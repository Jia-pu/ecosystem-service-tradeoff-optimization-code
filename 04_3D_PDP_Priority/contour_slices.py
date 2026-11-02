from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,xgboost as xgb
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,font_manager
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from matplotlib.patches import Patch
from common.plot_helpers import setup
import os
setup(os.environ.get('TNR_FONT'),False)
O=Path('validation/slices');O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':23,'axes.labelsize':26,'axes.titlesize':25,'xtick.labelsize':21,'ytick.labelsize':21,'axes.linewidth':1.1,'hatch.linewidth':.55})
d=pd.read_csv('validation_inputs/modeling_data_12_factor_comparison.csv')
f=['Temperature','DEM','Slope','GDP','POP','NL','PD','SHDI','CONTAG','Area_mn']
x=d[f].to_numpy(np.float32);rng=np.random.default_rng(992);bg=x[rng.choice(len(d),128,replace=False)]
m=xgb.XGBRegressor();m.load_model('validation/models/final_spatial_validated.ubj');m.set_params(n_jobs=4)
quantiles=[.25,.5,.75];cuts=d.CONTAG.quantile(quantiles).to_numpy()
combos=[['PD','SHDI','CONTAG'],['GDP','PD','CONTAG']]
fig=plt.figure(figsize=(24,15),facecolor='white');rows=[];summ=[];arrays={}
for row,comb in enumerate(combos):
    name='__'.join(comb);aa=[np.linspace(*d[v].quantile([.01,.99]),35) for v in comb[:2]]
    xx,yy=np.meshgrid(*aa,indexing='ij');flat=np.c_[xx.ravel(),yy.ravel()]
    ed=[np.unique(d[v].quantile(np.linspace(0,1,7))) for v in comb]
    shape=tuple(len(e)-1 for e in ed);counts=np.zeros(shape,int)
    ii=[np.clip(np.searchsorted(e,d[v],side='right')-1,0,len(e)-2) for e,v in zip(ed,comb)];np.add.at(counts,tuple(ii),1)
    surfaces=[];supports=[];masses=[]
    for q,c in zip(quantiles,cuts):
        vals=[]
        for pts in np.array_split(flat,9):
            mat=np.tile(bg,(len(pts),1))
            mat[:,f.index(comb[0])]=np.repeat(pts[:,0],len(bg));mat[:,f.index(comb[1])]=np.repeat(pts[:,1],len(bg));mat[:,f.index('CONTAG')]=c
            vals.extend(m.predict(mat).reshape(len(pts),len(bg)).mean(1))
        v=np.array(vals).reshape(xx.shape)
        pts=np.c_[flat,np.full(len(flat),c)]
        jj=[np.clip(np.searchsorted(e,pts[:,j],side='right')-1,0,len(e)-2) for j,e in enumerate(ed)]
        mass=counts[tuple(jj)].reshape(xx.shape);ok=mass>=30
        surfaces.append(v);supports.append(ok);masses.append(mass)
        rows.append(pd.DataFrame({'combination':name,'CONTAG_quantile':q,'CONTAG_value':c,comb[0]:flat[:,0],comb[1]:flat[:,1],'PDP':v.ravel(),'local_3D_bin_n':mass.ravel(),'supported':ok.ravel()}))
        summ.append({'combination':name,'quantile':q,'CONTAG':float(c),'PDP_min':float(v.min()),'PDP_max':float(v.max()),'grid_supported_fraction':float(ok.mean())})
        arrays[name+f'_Q{int(q*100)}_PDP']=v
    lo=np.floor(min(v.min() for v in surfaces)*100)/100;hi=np.ceil(max(v.max() for v in surfaces)*100)/100
    levels=np.linspace(lo,hi,25);cm=plt.get_cmap('viridis',24);norm=colors.BoundaryNorm(levels,cm.N)
    titley=.962 if row==0 else .492;py=.59 if row==0 else .12
    fig.text(.07,titley,' × '.join(comb),fontsize=30,weight='bold',ha='left')
    for col,(q,c,v,ok) in enumerate(zip(quantiles,cuts,surfaces,supports)):
        ax=fig.add_axes([.07+.276*col,py,.215,.29]);im=ax.contourf(aa[0],aa[1],v.T,levels=levels,cmap=cm,norm=norm)
        # Shared contour levels within each row, with numeric labels.
        lines=np.linspace(lo,hi,9)[1:-1]
        ct=ax.contour(aa[0],aa[1],v.T,levels=lines,colors='white',linewidths=.9)
        ax.clabel(ct,inline=True,fontsize=17,fmt='%.3f',inline_spacing=6)
        if (~ok).any():
            ax.contourf(aa[0],aa[1],np.where(ok,0,1).T,levels=[.5,1.5],colors='none',hatches=['////'])
        ax.set(xlabel=comb[0],ylabel=comb[1],xlim=(aa[0].min(),aa[0].max()),ylim=(aa[1].min(),aa[1].max()))
        ax.set_title(f'Q{int(q*100)}: CONTAG = {c:.3f}',pad=14)
        ax.text(-.10,1.04,f'({chr(97+row*3+col)})',transform=ax.transAxes,fontsize=25,ha='right')
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5));ax.tick_params(pad=7)
    cb=fig.colorbar(im,cax=fig.add_axes([.91,py,.012,.29]),ticks=np.linspace(lo,hi,6))
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'));cb.ax.tick_params(labelsize=21,pad=5)
    cb.set_label('Predicted TOP (PDP)',fontsize=25,labelpad=17)
    arrays[name+'_axis0']=aa[0];arrays[name+'_axis1']=aa[1]
fig.legend(handles=[Patch(facecolor='white',edgecolor='#555',hatch='////',label='Sparse support: fewer than 30 observations in the joint 3D bin')],loc='lower center',bbox_to_anchor=(.5,.035),frameon=False,fontsize=21)
fig.text(.5,.010,'Axes show transformed, dimensionless feature values. Each row shares its color scale; slices use actual CONTAG quartiles.',ha='center',fontsize=21,color='#405654')
fig.savefig(O/'Three_factor_PDP_2D_slices_preview.jpg',dpi=350,bbox_inches='tight',pad_inches=.30,pil_kwargs={'quality':97,'subsampling':0})
fig.savefig(O/'layout_preview.jpg',dpi=70,bbox_inches='tight',pad_inches=.25)
pd.concat(rows,ignore_index=True).to_csv(O/'slice_values.csv',index=False)
pd.DataFrame(summ).to_csv(O/'slice_summary.csv',index=False)
np.savez_compressed(O/'slice_arrays.npz',**arrays)
meta={'model':'validation/models/final_spatial_validated.ubj','model_sha256':hashlib.sha256(Path('validation/models/final_spatial_validated.ubj').read_bytes()).hexdigest(),'xgboost':xgb.__version__,'background_seed':992,'background_n':128,'grid_n':35,'support':'6 marginal quantile bins per variable; joint bin count >=30','mode':'fresh model predictions, not interpolated from archived 13^3 surface','font':'Times New Roman (required by this entry)','feature_scale':'transformed dimensionless; CONTAG quartile values are not original percentages','unchanged':'No retraining, no threshold inference, no modification of priority zoning'}
(O/'method.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(pd.DataFrame(summ).to_string(index=False),flush=True)
