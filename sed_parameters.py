from utils import *

interp_lims   = {'AV':[0,4], 'LOG_TAU':[7.0,10.], 'LOG_AGE':[7.5,10]}
interp_labels = ['AV', 'LOG_TAU', 'LOG_AGE']
interp_smooth = [0.0001,0.04,0.02]

def read_3dhst_catalog():

    fields = ["aegis","cosmos","goodsn","goodss"]
    catalog = fitsio.getdata("3dhst/{:s}_params.fits".format(fields[0]))
    for field in fields[1:]:
        _catalog = fitsio.getdata("3dhst/{:s}_params.fits".format(field))
        catalog  = rfn.stack_arrays((catalog,_catalog),asrecarray=True,usemask=False,autoconvert=False)
    return catalog

catalog_3dhst = read_3dhst_catalog()

def cut_3dhst(data,z0,z1):

    data = data[(z0 <= data['Z_PHOT']) & (data['Z_PHOT'] <= z1)]
    return data

def gen_func_sedpars(catalog,z0,z1):

    catalog = cut_3dhst(data=catalog,z0=z0,z1=z1)

    interp_fns = {}
    for label,s in zip(interp_labels,interp_smooth):

        lim, data = interp_lims[label], catalog[label]
        data = data[(lim[0] <= data) & (data <= lim[1])]

        x = np.arange(len(data))/float(len(data))
        y = np.sort(data)
        yb = np.sort(np.array(list(set(y))))
        xb = np.array([np.mean(x[y==i]) for i in yb])
        yb = np.insert(yb,[0,len(yb)],[y[0],y[-1]])
        xb = np.insert(xb,[0,len(xb)],[0,1])
        fn = scipy.interpolate.UnivariateSpline(xb,yb,k=3,s=s)
        interp_fns[label] = fn

    return interp_fns

def plot_sedpars():

    def plot_jointdist(data,z0,z1,axis,labelx,labely,color):

        catalog = cut_3dhst(catalog_3dhst,z0,z1)
        axis.scatter(data[labelx],data[labely],s=10,color=color,lw=0,alpha=0.5)
        axis.set_xlim(lims[labelx])
        axis.set_ylim(lims[labely])

    def plot_hist(data,z0,z1,bins,axis,label,color):

        catalog = cut_3dhst(catalog_3dhst,z0,z1)
        data = catalog[label]

        hist, bins = np.histogram(data, bins=bins)
        binc = 0.5*(bins[1:]+bins[:-1])
        hist = hist/float(sum(hist))
        axis.plot(binc,hist,color=color,lw=1.5,drawstyle='steps-mid')
        axis.set_xlim(lims[label])
        axis.set_ylim(0,max(1.1*np.max(hist),axis.get_ylim()[1]))

    zrange = np.clip(np.arange(0,survey_zmax,1.0),survey_zmin,np.inf)
    colormap = plt.cm.gist_rainbow_r(np.linspace(0,1,len(zrange)))

    bbins = [('AV'      ,np.arange(-1,5,0.1)),
             ('LOG_TAU' ,np.arange(5,15,0.25)),
             ('LOG_AGE' ,np.arange(5,15,0.2)),
             ('LOG_MASS',np.arange(0,15,0.2))]
    bbins = OrderedDict(bbins)
    lims  = [('AV'      ,[-0.1, 4.0]),
             ('LOG_TAU' ,[ 6.8,10.2]),
             ('LOG_AGE' ,[ 7.3,10.2]),
             ('LOG_MASS',[ 3.0,13.0])]
    lims  = OrderedDict(lims)

    fig, axes = plt.subplots(len(bbins.keys()),len(bbins.keys()),figsize=(12,12),dpi=75)
    fig.subplots_adjust(left=0.08,right=0.98,bottom=0.08,top=0.98,hspace=0,wspace=0)

    for j,labely in enumerate(bbins.keys()):
        for i,labelx in enumerate(bbins.keys()):
            axes[j,i].xaxis.set_visible(False)
            axes[j,i].yaxis.set_visible(False)
            if i > j: axes[j,i].set_visible(False)
            if i == 0 and j != 0:
                axes[j,i].yaxis.set_visible(True)
                axes[j,i].set_ylabel(labely)
            if j == axes.shape[0]-1:
                axes[j,i].xaxis.set_visible(True)
                axes[j,i].set_xlabel(labelx)
            if i != j:
                for k,(zmin,zmax) in enumerate(zip(zrange[:-1],zrange[1:])):
                    plot_jointdist(data=catalog_3dhst,z0=zmin,z1=zmax,axis=axes[j,i],
                                   labelx=labelx,labely=labely,
                                   color=colormap[k])
            if i == j:
                for k,(zmin,zmax) in enumerate(zip(zrange[:-1],zrange[1:])):
                    plot_hist(data=catalog_3dhst,z0=zmin,z1=zmax,bins=bbins[labelx],
                               axis=axes[j,i],label=labelx,color=colormap[k])

    for i,label in enumerate(bbins.keys()):
        for k,(zmin,zmax) in enumerate(zip(zrange[:-1],zrange[1:])):
            plot_hist(data=catalog_3dhst,z0=zmin,z1=zmax,bins=bbins[label],
                       axis=axes[i,i],label=label,color=colormap[k])


    for k,(zmin,zmax) in enumerate(zip(zrange[:-1],zrange[1:])):
        axes[0,0].scatter(-99, -99, s=10, color=colormap[k],
                            alpha=0.8, label='{:.1f}<z<{:.1f}'.format(zmin,zmax))

    leg = axes[0,0].legend(fontsize=16,ncol=1,loc="center left",framealpha=0,
                           handlelength=0,handletextpad=0,
                           bbox_to_anchor=[3,0.5])

    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_fontweight(600)
        txt.set_color(hndl.get_facecolor()[0])
        hndl.set_visible(False)

    return fig, axes

def plot_func_sedpars():

    zrange = np.array([survey_zmin,1,2,3,survey_zmax])
    colormap = plt.cm.gist_rainbow_r(np.linspace(0,1,len(zrange)))

    fig,axes = plt.subplots(1,3,figsize=(18,6),dpi=75,sharey=True)
    fig.subplots_adjust(left=0.05,right=0.98,bottom=0.12,top=0.98,wspace=0.1,hspace=0.1)

    for k,(zmin,zmax) in enumerate(zip(zrange[:-1],zrange[1:])):
        interp_fns = gen_func_sedpars(catalog=catalog_3dhst,z0=zmin,z1=zmax)
        for i,label in enumerate(interp_labels):
            _y = np.linspace(0,1,1000)
            _x = interp_fns[label](_y)
            axes[i].plot(_x,_y,color=colormap[k],lw=1.5,alpha=0.8,label='{:.1f}<z<{:.1f}'.format(zmin,zmax))

    for i,label in enumerate(interp_labels):
        axes[i].set_xlabel(label,fontsize=18)
        axes[i].set_ylim(0,1)
        axes[i].set_xlim(interp_lims[label])

    axes[0].set_ylabel("CDF",fontsize=18)
    axes[0].legend(loc=4,fontsize=18)

if __name__ == '__main__':

    # plot_sedpars()
    plot_func_sedpars()
    plt.show()
