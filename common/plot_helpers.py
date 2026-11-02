"""Plotting helpers reorganized from the archived portable replot script.
The archived portable script was reconstructed, not the lost original layout file.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
ROOT=Path(__file__).resolve().parents[1]
def setup(font, allow_fallback):
    if font:
        font_manager.fontManager.addfont(str(Path(font).expanduser()))
    try:
        font_manager.findfont('Times New Roman',fallback_to_default=False)
        family='Times New Roman'
    except ValueError:
        if not allow_fallback:
            raise SystemExit('Supply --font PATH_TO_TIMES_TTF or TNR_FONT. For a diagnostic preview only, use --allow-font-fallback.')
        family='DejaVu Serif'
    plt.rcParams.update({'font.family':family,'font.size':24,'axes.labelsize':27,
                         'axes.titlesize':30,'xtick.labelsize':23,'ytick.labelsize':23})

def map_panel(fig,rect,d,z,name,letter,palette,legend_xy,fontsize=23):
    ax=fig.add_axes(rect);ax.axis('off');ax.set_aspect('equal')
    ax.text(-.015,1.01,letter,transform=ax.transAxes,fontsize=28)
    r=(d.row-d.row.min()).to_numpy();c=(d.col-d.col.min()).to_numpy()
    zz=z[name+'_zone'].to_numpy()
    g=np.full((int(r.max()+1),int(c.max()+1)),np.nan)
    g[r,c]=np.where(zz==2,2,np.where(zz==1,1,0))
    cm=colors.ListedColormap([palette[0],palette[1],palette[2]])
    ax.imshow(g,origin='upper',cmap=cm,norm=colors.BoundaryNorm([-.5,.5,1.5,2.5],3),interpolation='nearest',
              extent=[d.x.min()-500,d.x.max()+500,d.y_coord.min()-500,d.y_coord.max()+500])
    ax.set(xlim=(d.x.min()-10000,d.x.max()+10000),ylim=(d.y_coord.min()-10000,d.y_coord.max()+10000))
    handles=[Patch(facecolor=palette[2],label='High-priority optimization areas'),
             Patch(facecolor=palette[1],label='Secondary-priority optimization areas'),
             Patch(facecolor=palette[0],edgecolor='#dbe1dc',label='Urban-fringe areas')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=legend_xy,frameon=False,
               fontsize=fontsize,handlelength=1.35,labelspacing=.42,borderaxespad=0)
    return ax

def save(fig,name,args):
    target=ROOT/'replotted_figures';target.mkdir(exist_ok=True)
    fig.savefig(target/(name+'.jpg'),dpi=args.dpi,bbox_inches='tight',pad_inches=.35,
                pil_kwargs={'quality':97,'subsampling':0})
    plt.close(fig)
