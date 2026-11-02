from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from common.plot_helpers import setup
import os
setup(os.environ.get('TNR_FONT'),False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':24,'axes.labelsize':30,'axes.titlesize':31,'xtick.labelsize':23,'ytick.labelsize':23,'axes.linewidth':1.1})
D=pd.read_csv('validation_inputs/modeling_data_12_factor_comparison.csv');Z=pd.read_csv('validation/data/priority_zones_all_cells.csv');S=np.load('validation/data/joint_surfaces.npz');T=np.load('validation/slices/slice_arrays.npz')
cuts=D.CONTAG.quantile([.25,.5,.75]).to_numpy();names=['PD__SHDI__CONTAG','GDP__PD__CONTAG']
fig=plt.figure(figsize=(25,22),facecolor='white');support_stats=[]
def support(comb,pts):
 edges=[np.unique(D[f].quantile(np.linspace(0,1,7))) for f in comb]
 counts=np.zeros(tuple(len(e)-1 for e in edges),int)
 ids=[np.clip(np.searchsorted(e,D[f],side='right')-1,0,len(e)-2) for e,f in zip(edges,comb)]
 np.add.at(counts,tuple(ids),1)
 ids=[np.clip(np.searchsorted(e,pts[:,j],side='right')-1,0,len(e)-2) for j,e in enumerate(edges)]
 return counts[tuple(ids)]>=30
for row,(name,py) in enumerate(zip(names,[.55,.08])):
 fs=name.split('__');aa=[S[name+'_axis'+str(i)] for i in range(3)];v=S[name+'_PDP'];xyz=np.meshgrid(*aa,indexing='ij');pts=np.c_[*(a.ravel() for a in xyz)];ok=support(fs,pts)
 lo=np.floor(v.min()*100)/100;hi=np.ceil(v.max()*100)/100;norm=colors.Normalize(lo,hi);cm=plt.get_cmap('viridis')
 ax=fig.add_axes([.005,py,.47,.39],projection='3d')
 ax.scatter(*pts[ok].T,c=v.ravel()[ok],cmap=cm,norm=norm,s=14,alpha=.38,edgecolors='none',depthshade=False)
 ax.view_init(elev=25,azim=-60);ax.set_box_aspect((1.16,1,.9),zoom=1.08);ax.set_proj_type('persp',focal_length=1.15)
 x=T[name+'_axis0'];y=T[name+'_axis1'];xx,yy=np.meshgrid(x,y,indexing='ij');base=np.c_[xx.ravel(),yy.ravel()]
 levels=np.linspace(lo,hi,25);linelevels=np.linspace(lo,hi,8)[1:-1]
 for j,(q,c) in enumerate(zip([25,50,75],cuts)):
  vals=T[name+f'_Q{q}_PDP'];mask=support(fs,np.c_[base,np.full(len(base),c)]).reshape(vals.shape);zz=np.ma.masked_where(~mask,vals).T
  ax.contourf(x,y,zz,zdir='z',offset=c,levels=levels,cmap=cm,norm=norm,alpha=.83,corner_mask=False)
  # Extract contour geometry on a temporary flat axis, then place lines at the fixed CONTAG level.
  tmp,tax=plt.subplots();cs=tax.contour(x,y,zz,levels=linelevels);segments=cs.allsegs
  for i,segs in enumerate(segments):
   for seg in segs:
    if len(seg)>1:ax.plot(seg[:,0],seg[:,1],np.full(len(seg),c),color='white',lw=.7,alpha=.9)
  # Two numeric labels per slice, avoiding unsupported locations and edge labels.
  eligible=[(i,max([a for a in segs if len(a)>5],key=len)) for i,segs in enumerate(segments) if any(len(a)>5 for a in segs)]
  if eligible:
   chosen=[eligible[len(eligible)//3],eligible[(2*len(eligible))//3]]
   used=[]
   for i,seg in chosen:
    inside=(seg[:,0]>x.min()+.10*np.ptp(x))&(seg[:,0]<x.max()-.10*np.ptp(x))&(seg[:,1]>y.min()+.10*np.ptp(y))&(seg[:,1]<y.max()-.10*np.ptp(y));cand=seg[inside]
    if not len(cand):continue
    point=cand[len(cand)//2]
    if any(np.linalg.norm((point-a)/[np.ptp(x),np.ptp(y)])<.2 for a in used):continue
    ax.text(*point,c+.003,f'{linelevels[i]:.3f}',fontsize=19,zorder=100,ha='center',va='center',color='#172f30',bbox=dict(facecolor='white',edgecolor='none',alpha=.90,pad=.7))
    used.append(point)
  plt.close(tmp)
 ax.xaxis.set_rotate_label(False);ax.yaxis.set_rotate_label(False)
 ax.set_xlabel(fs[0],labelpad=23,rotation=-17);ax.set_ylabel(fs[1],labelpad=29,rotation=42);ax.set_zlabel('CONTAG',labelpad=23)
 ax.set(xlim=(aa[0].min(),aa[0].max()),ylim=(aa[1].min(),aa[1].max()),zlim=(aa[2].min(),aa[2].max()))
 ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5));ax.set_zticks(cuts,[f'{c:.3f}' for c in cuts])
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
  axis.set_tick_params(pad=6,labelsize=23);axis.pane.set_facecolor((.97,.97,.97,1));axis.pane.set_edgecolor('#a5a5a5');axis._axinfo['grid'].update(color='#b2b2b2',linewidth=.6)
 ax.set_title(' × '.join(fs),pad=22);ax.text2D(-.02,1.01,f'({chr(97+row*2)})',transform=ax.transAxes,fontsize=29)
 cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=fig.add_axes([.50,py+.075,.011,.245]),ticks=np.linspace(lo,hi,6));cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'));cb.ax.tick_params(labelsize=23);cb.set_label('Predicted TOP (PDP)',rotation=270,labelpad=29,fontsize=27)
 mp=fig.add_axes([.64,py+.035,.31,.33]);mp.axis('off');mp.set_aspect('equal');mp.text(-.01,1.01,f'({chr(98+row*2)})',transform=mp.transAxes,fontsize=29)
 r=(D.row-D.row.min()).to_numpy();cc=(D.col-D.col.min()).to_numpy();zone=Z[name+'_zone'].to_numpy();g=np.full((int(r.max()+1),int(cc.max()+1)),np.nan);g[r,cc]=np.where(zone==2,2,np.where(zone==1,1,0))
 mp.imshow(g,origin='upper',cmap=colors.ListedColormap(['#edf0ec','#ffe17a','#ed9b00']),norm=colors.BoundaryNorm([-.5,.5,1.5,2.5],3),interpolation='nearest',extent=[D.x.min()-500,D.x.max()+500,D.y_coord.min()-500,D.y_coord.max()+500]);mp.set(xlim=(D.x.min()-10000,D.x.max()+10000),ylim=(D.y_coord.min()-10000,D.y_coord.max()+10000))
 fig.legend(handles=[Patch(facecolor='#ed9b00',label='High-priority optimization areas'),Patch(facecolor='#ffe17a',label='Secondary-priority optimization areas'),Patch(facecolor='#edf0ec',label='Urban-fringe areas')],loc='upper left',bbox_to_anchor=(.64,py+.015),fontsize=23,frameon=False,labelspacing=.4)
fig.text(.5,-.012,f'Horizontal slices: CONTAG Q25 = {cuts[0]:.3f}, Q50 = {cuts[1]:.3f}, Q75 = {cuts[2]:.3f}. Colors and contour labels show predicted TOP.',ha='center',fontsize=22,color='#405654')
fig.text(.5,-.031,'Sparse joint bins (<30 observations) are omitted. Axes show transformed, dimensionless features; GIS zoning retains the Q75 rules.',ha='center',fontsize=21,color='#405654')
out=Path('replotted_figures');out.mkdir(exist_ok=True)
fig.savefig(out/'Three_dimensional_PDP_with_contour_slices.jpg',dpi=350,bbox_inches='tight',pad_inches=.30,pil_kwargs={'quality':97,'subsampling':0})
fig.savefig(out/'layout_preview.jpg',dpi=70,bbox_inches='tight',pad_inches=.25)
print('DONE')
