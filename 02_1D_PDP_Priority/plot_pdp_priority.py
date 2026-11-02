from common.plot_helpers import *

def plot1(d,z,s,args):
    """Uses archived 81-point PDP/ICE. Historical dense-grid illustration may differ slightly."""
    c=np.load(ROOT/'validation/data/main_response_curves.npz')
    fig=plt.figure(figsize=(28,23),facecolor='white')
    names=['GDP','PD','POP','CONTAG','SHDI']
    positions=[(.045,.31,.73),(.56,.825,.73),(.045,.31,.41),(.56,.825,.41),(.3025,.5675,.09)]
    # These C annotations are archived exploratory local-change candidates, not robust thresholds.
    annotation={'GDP':[('C1',.504044),('C2',.852160)],'PD':[('T',.5563729364)],'POP':[('C1',.570454),('C2',.783045)],'CONTAG':[('C1',.017417),('C2',.771927)],'SHDI':[('C1',.572233),('C2',.864271)]}
    for k,(f,(px,mx,py)) in enumerate(zip(names,positions)):
        ax=fig.add_axes([px,py,.18,.23]);x=c[f+'_grid'];p=c[f+'_PDP'];ice=c[f+'_ICE']
        ax.plot(x,ice[:,::4],color='#9bd6cc',lw=.6,alpha=.16)
        ax.plot(x,p,color='#164f52',lw=3,label='PDP')
        for i,(lab,t) in enumerate(annotation[f]):
            ax.axvline(t,color='#aa5936',ls='--',lw=1.7)
            ax.text(t,.95-i*.12,f'{lab} = {t:.3f}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=18,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
        ax.set(xlabel=f,ylabel='Partial dependence',xlim=(x.min(),x.max()));ax.set_title(f,pad=17)
        ax.text(-.10,1.015,f'({chr(97+2*k)})',transform=ax.transAxes,ha='right',va='bottom',fontsize=28)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.legend(frameon=False,fontsize=18,loc='lower right')
        dens=ax.inset_axes([0,-.18,1,.09]);dens.hist(d[f],bins=40,color='#8bbcb0');dens.set_xlim(ax.get_xlim());dens.axis('off')
        map_panel(fig,[mx,py,.165,.23],d,z,f,f'({chr(98+2*k)})',['#edf0ec','#9bd088','#00742d'],(mx,py-.035),18)
    fig.text(.5,-.015,'C: exploratory local-change candidate; T: supported PD response-turning candidate. Axes show transformed, dimensionless feature values; zoning uses Q75 criteria.',ha='center',fontsize=20,color='#405654')
    save(fig,'Five_factor_PDP_priority_replot',args)

plot = plot1
