from common.plot_helpers import *

def plot2(d,z,s,args):
    plt.rcParams.update({'axes.labelsize':27,'axes.titlesize':30})
    fig=plt.figure(figsize=(28,22),facecolor='white')
    names=['GDP__PD','POP__NL','PD__SHDI','SHDI__CONTAG','PD__CONTAG']
    positions=[(.045,.31,.73),(.56,.825,.73),(.045,.31,.41),(.56,.825,.41),(.3025,.5675,.09)]
    for k,(name,(px,mx,py)) in enumerate(zip(names,positions)):
        a,b=name.split('__');x=s[name+'_axis0'];y=s[name+'_axis1'];v=s[name+'_PDP']
        lo=np.floor(float(v.min())*1000)/1000;hi=np.ceil(float(v.max())*1000)/1000
        levels=np.linspace(lo,hi,27);cm=plt.get_cmap('GnBu',26);norm=colors.BoundaryNorm(levels,cm.N)
        ax=fig.add_axes([px,py,.18,.23])
        cf=ax.contourf(x,y,v.T,levels=levels,cmap=cm,norm=norm,extend='both',antialiased=True)
        lev=np.linspace(float(v.min()),float(v.max()),9)[1:-1]
        ct=ax.contour(x,y,v.T,levels=lev,colors='white',linewidths=1.15)
        used=[]
        for i,t in enumerate(lev):
            ss=[p for p in ct.allsegs[i] if len(p)>2]
            if not ss:continue
            p=max(ss,key=len);ok=(p[:,0]>x.min()+.06*np.ptp(x))&(p[:,0]<x.max()-.06*np.ptp(x))&(p[:,1]>y.min()+.06*np.ptp(y))&(p[:,1]<y.max()-.06*np.ptp(y));p=p[ok]
            if not len(p):continue
            q=p[len(p)//2];u=(q-[x.min(),y.min()])/[np.ptp(x),np.ptp(y)]
            if any(np.linalg.norm(u-o)<.14 for o in used):continue
            ax.text(*q,f'{t:.2f}',fontsize=20,color='#193b3b',ha='center',va='center',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.18))
            used.append(u)
        ax.set(xlabel=a,ylabel=b,xlim=(x.min(),x.max()),ylim=(y.min(),y.max()))
        ax.set_title(f'{a} × {b}',pad=17);ax.text(-.10,1.015,f'({chr(97+2*k)})',transform=ax.transAxes,ha='right',va='bottom',fontsize=28)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5));ax.tick_params(pad=8)
        cb=fig.colorbar(cf,cax=fig.add_axes([px+.198,py+.009,.007,.212]),ticks=np.linspace(lo,hi,6))
        cb.ax.tick_params(labelsize=19,pad=3);cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        cb.set_label('Partial dependence',rotation=270,labelpad=23,fontsize=23)
        map_panel(fig,[mx,py,.165,.23],d,z,name,f'({chr(98+2*k)})',['#edf0ec','#7fcac4','#144792'],(mx,py-.013),18.5)
    fig.text(.5,-.003,'Panel-specific color scales; transformed, dimensionless axes. PDP includes sparse combinations; Q75 zoning uses supported observations.',ha='center',fontsize=20,color='#405654')
    save(fig,'Two_factor_PDP_priority_replot',args)

plot = plot2
