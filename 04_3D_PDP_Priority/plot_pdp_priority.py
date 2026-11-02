from common.plot_helpers import *

def plot3(d,z,s,args):
    plt.rcParams.update({'axes.labelsize':33,'axes.titlesize':35})
    fig=plt.figure(figsize=(24,20),facecolor='white')
    for k,(name,yy) in enumerate(zip(['PD__SHDI__CONTAG','GDP__PD__CONTAG'],[.55,.075])):
        fs=name.split('__');aa=[s[name+'_axis'+str(j)] for j in range(3)];v=s[name+'_PDP']
        lo=np.floor(v.min()*1000)/1000;hi=np.ceil(v.max()*1000)/1000
        ax=fig.add_axes([.005,yy,.47,.40],projection='3d')
        xyz=np.meshgrid(*aa,indexing='ij')
        sc=ax.scatter(*(a.ravel() for a in xyz),c=v.ravel(),cmap='viridis',vmin=lo,vmax=hi,
                      s=27,alpha=1,edgecolors='none',depthshade=False)
        ax.view_init(elev=30,azim=-60);ax.set_box_aspect((1.16,1,.86),zoom=1.10)
        ax.set_proj_type('persp',focal_length=1.15)
        ax.xaxis.set_rotate_label(False);ax.yaxis.set_rotate_label(False)
        ax.set_xlabel(fs[0],labelpad=24,rotation=-17)
        ax.set_ylabel(fs[1],labelpad=31,rotation=43);ax.set_zlabel(fs[2],labelpad=23)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.set_major_locator(MaxNLocator(5));axis.set_major_formatter(FormatStrFormatter('%.1f'))
            axis.set_tick_params(pad=6,labelsize=26)
            axis.pane.set_facecolor((.965,.97,.97,1));axis.pane.set_edgecolor('#a5a5a5')
            axis.pane.set_linewidth(.6);axis.line.set_color('#404040');axis.line.set_linewidth(.9)
            axis._axinfo['grid'].update(color='#a6a6a6',linewidth=.65)
        ax.set(xlim=(aa[0].min(),aa[0].max()),ylim=(aa[1].min(),aa[1].max()),zlim=(aa[2].min(),aa[2].max()))
        ax.set_title(' × '.join(fs),pad=19)
        ax.text2D(-.025,1.015,f'({chr(97+2*k)})',transform=ax.transAxes,fontsize=28)
        cb=fig.colorbar(sc,cax=fig.add_axes([.502,yy+.073,.009,.245]),ticks=np.linspace(lo,hi,6))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'));cb.ax.tick_params(labelsize=26,pad=6)
        cb.set_label('Partial dependence',rotation=270,labelpad=30,fontsize=29)
        map_panel(fig,[.625,yy+.025,.34,.35],d,z,name,f'({chr(98+2*k)})',
                  ['#edf0ec','#ffe17a','#ed9b00'],(.625,yy-.003))
    fig.text(.5,-.035,'Predictor axes show transformed, dimensionless values. Each panel uses its own color scale; compare numeric PDP values across panels.',ha='center',fontsize=21,color='#405654')
    fig.text(.5,-.055,'Full modeled factor grids include sparsely sampled combinations. Priority zoning uses supported observations and Q75 criteria.',ha='center',fontsize=20,color='#405654')
    save(fig,'Three_factor_PDP_priority_replot',args)

plot = plot3
