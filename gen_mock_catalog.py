from utils import *
from define import *
from sed_parameters import *
from extract_bc03 import TemplateSED_BC03

def get_correct_LF(z):

    cond = np.array([_["zmin"]<=z<_["zmax"] for _ in LFs],dtype=bool)
    if not np.any(cond):
        raise Exception("Invalid redshift.")
    if sum(cond)>1:
        raise Exception("Multiple LFs defined for z={:f}.".format(z))
    return LFs[cond][0]

def calc_Ngal(zmin,zmax,mlim,area):

    def integrand(M,z,LF):
        return schechter_func(M,**LF) * \
               Planck15.differential_comoving_volume(z=z).value * \
               get_solid_angle(area)

    def Mlim(z,*extras):
        return [-25, get_abs_from_app(app_mag=mlim,z=z)]

    LF = get_correct_LF(0.5*(zmin+zmax))
    return scipy.integrate.nquad(integrand,[Mlim,[zmin,zmax]],opts=[quad_args,quad_args],args=(LF,))[0]

def gen_dNdz(mlim,area,plot=False):

    zrange = np.arange(survey_zmin,survey_zmax,0.1)
    Ngals  = Parallel(n_jobs=15,verbose=0,backend="multiprocessing")(
                delayed(calc_Ngal)(zmin=zmin,zmax=zmax,mlim=mlim,area=area)
                     for zmin,zmax in zip(zrange[:-1],zrange[1:]))

    if plot:

        fig,ax = plt.subplots(1,1,figsize=(10,7),dpi=75,tight_layout=True)

        dNdz = Ngals / np.diff(zrange)
        ax.plot(0.5*(zrange[:-1]+zrange[1:]),dNdz,c='k',lw=2,alpha=0.8)
        for i,LF in enumerate(LFs):
            ax.axvspan(LF["zmin"],LF["zmax"],
                        color=plt.cm.gist_rainbow_r(np.linspace(0.2,1.0,len(LFs)))[i],
                        lw=0,alpha=0.2)
        ax.set_title("dN/dz for {:.1f} arcmin$^2$ and m<{:.1f}".format(survey_area*3600.,survey_mlim),fontsize=18,fontweight=600)
        ax.set_ylabel("Ngals per $\\Delta z=0.1$",fontsize=18)
        ax.set_xlabel("Redshift",fontsize=18)
        ax.set_xlim(0.0,4.1)
        ax.set_ylim(0,ax.get_ylim()[1])
        [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    return zrange,Ngals

def get_dtype():

    dtype = [("ID",int),("redshift",float),("absM",float),("norm_wave",float)]
    dtype += [_dtype for camera in hst_filters \
                     for instr  in hst_filters[camera] \
                     for filt   in hst_filters[camera][instr] \
                     for _dtype in [("flux_{:s}_{:s}".format(camera,filt),float),
                                    ( "mag_{:s}_{:s}".format(camera,filt),float)]]
    dtype += [("lmass",float),("lage",float),("ltau",float),("Av",float),("metal",float)]
    return dtype

def gen_Ngals():

    zrange,Ngals = gen_dNdz(mlim=survey_mlim,area=survey_area)
    Ngals = np.ceil(Ngals).astype(int)

    catalog = np.recarray(sum(Ngals),dtype=get_dtype())
    for _ in catalog.dtype.names: catalog[_] = -99

    idx = 0
    for zmin,zmax,Ngal in zip(zrange[:-1],zrange[1:],Ngals):
        print("--- Ngal --- {:3d} galaxies @ {:.2f}<z<{:.2f} [total:{:5d}]".format(Ngal,zmin,zmax,len(catalog)))
        catalog["redshift"][idx:idx+Ngal] = gen_random_z(z0=zmin,z1=zmax,N=Ngal)
        idx = idx+Ngal

    catalog["ID"] = np.arange(len(catalog))+1
    return catalog

def gen_random_z(z0,z1,N):

    return np.random.rand(N) * (z1-z0) + z0

def gen_random_absM(catalog):

    for LF in LFs:

        idx = np.where((LF["zmin"]<=catalog["redshift"]) & (catalog["redshift"]<LF["zmax"]))[0]

        Mlim = get_abs_from_app(survey_mlim,z=LF["zmin"])
        icdf = gen_LF_iCDF(alpha=LF["alpha"],Mst=LF["Mst"],mag_lim=Mlim)

        _Mlim = get_abs_from_app(survey_mlim,z=catalog["redshift"][idx])
        ntry, cond = 0, np.ones_like(idx,dtype=bool)
        while np.any(cond):
            print("\r--- absM --- {2:5d} {3[zmin]:.2f}<z<{3[zmax]:.2f} galaxies with {3[ref]:12s} -- try#{0:d}".format(ntry,sum(cond),len(idx),LF),end="")
            cond = (catalog["absM"][idx] > _Mlim) | (catalog["absM"][idx]==-99.)
            catalog["absM"][idx[cond]] = icdf(np.random.rand(sum(cond)))
            catalog["norm_wave"][idx[cond]] = LF["wave"]
            ntry+=1
        print()

    return catalog

def gen_random_sedpars(catalog):

    zrange = np.array([survey_zmin,1,2,3,survey_zmax])
    maxage = Planck15.age(catalog["redshift"]).value-0.5

    for zmin,zmax in zip(zrange[:-1],zrange[1:]):

        interp_fns = gen_func_sedpars(catalog=catalog_3dhst,z0=zmin,z1=zmax)
        idx = np.where((zmin<=catalog["redshift"]) & (catalog["redshift"]<zmax))[0]

        catalog[   "Av"][idx] = interp_fns[     "AV"](np.random.rand(len(idx)))
        catalog[ "ltau"][idx] = interp_fns["LOG_TAU"](np.random.rand(len(idx)))
        catalog[ "lage"][idx] = interp_fns["LOG_AGE"](np.random.rand(len(idx)))
        catalog["metal"][idx] = 0.02

        ntry, redo_ages = 0, True
        while np.any(redo_ages):
            ntry += 1
            print("\r--- sedP --- {:5d} {:.2f}<z<{:.2f} galaxies -- try#{:d}".format(len(idx),zmin,zmax,ntry),end="")
            redo_ages = (10**(catalog["lage"][idx]-9) > maxage[idx])
            catalog["lage"][idx[redo_ages]] = interp_fns["LOG_AGE"](np.random.rand(len(idx[redo_ages])))
        print()

    return catalog

def normalize_sed(template,abs_mag,norm_wave):

    app_mag   = get_app_from_abs(abs_mag,z=template.redshift)
    init_flux = calc_window_flux(wave=template.sed['waves'],spec=template.sed['spec1'],
                                 wave0=norm_wave*(1+template.redshift),window=100*(1+template.redshift))
    true_flux = 10**(-(app_mag+48.6)/2.5)
    flux_norm = true_flux / init_flux
    template.sed["spec1"] *= flux_norm

    lum_dist = Planck15.luminosity_distance(z=template.redshift).cgs.value
    mass_norm = flux_norm * (4.*np.pi*lum_dist**2) * (1+template.redshift)
    mass = np.log10(template.M_unnorm["spec1"] * mass_norm)

    return template, mass

def calc_magnitudes(entry):

    dtype  = [("ID",int),("lmass",float)]
    dtype += [_dtype for camera in hst_filters \
                     for instr  in hst_filters[camera] \
                     for filt   in hst_filters[camera][instr] \
                     for _dtype in [("flux_{:s}_{:s}".format(camera,filt),float),
                                    ( "mag_{:s}_{:s}".format(camera,filt),float)]]
    result = np.recarray(1,dtype=dtype)
    result["ID"] = entry["ID"]

    sfh = "constant" if entry["ltau"]<0 else "exp"
    template = TemplateSED_BC03(age=10**(entry["lage"]-9),
                                metallicity=entry["metal"],
                                sfh="exp",tau=10**(entry["ltau"]-9),
                                dust="calzetti",Av=entry['Av'],
                                redshift=entry["redshift"],
                                igm=True,lya_esc=0.2,lyc_esc=0,
                                emlines=True,
                                res="lr",imf="chab",units="flambda",
                                rootdir='/data/highzgal/mehta/Software/galaxev12/',library_version=2012,
                                cleanup=True,workdir="tmp/")

    template.generate_sed()
    template, result["lmass"] = normalize_sed(template,abs_mag=entry["absM"],norm_wave=entry["norm_wave"])

    for camera in hst_filters:
        for instr in hst_filters[camera]:
            for filt in hst_filters[camera][instr]:
                flux_cgs = calc_filter_flux(wave=template.sed["waves"],spec=template.sed["spec1"],camera=camera,instr=instr,filt=filt)
                result["flux_{:s}_{:s}".format(camera,filt)] = flux_cgs * calc_zeropoint_factor(init_zp=-48.6,outp_zp=23.9)
                result[ "mag_{:s}_{:s}".format(camera,filt)] = -2.5*np.log10(result["flux_{:s}_{:s}".format(camera,filt)]) + 23.9

    return result

def gen_magnitudes(catalog):

    result = Parallel(n_jobs=15,verbose=10,backend="multiprocessing")(delayed(calc_magnitudes)(entry) for entry in catalog)
    result = rfn.stack_arrays(result,asrecarray=True,usemask=False,autoconvert=False)
    assert np.all(result["ID"]==catalog["ID"])
    for x in result.dtype.names[1:]: catalog[x] = result[x]
    return catalog

def main():

    catalog = gen_Ngals()
    catalog = gen_random_absM(catalog)
    catalog = gen_random_sedpars(catalog)
    catalog = gen_magnitudes(catalog)

    fitsio.writeto("mock_catalog.fits",catalog,overwrite=True)

if __name__ == '__main__':

    # gen_dNdz(mlim=survey_mlim,area=survey_area,plot=True)
    main()
    plt.show()
