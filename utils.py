import os
import time
import numpy as np
import scipy.integrate
import scipy.interpolate
import astropy.units as u
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import pysynphot as S
import numpy.lib.recfunctions as rfn
from astropy.cosmology import Planck15
from collections import OrderedDict

from joblib import Parallel, delayed
from define import *

light = 3e18 # Angs / s
quad_args = {'limit':250,'epsrel':1e-4,'epsabs':1e-4}

def get_abs_from_app(app_mag,z):
    """
    Returns the absolute magnitude for a given apparent magnitude at redshift z.
    """
    dist = Planck15.luminosity_distance(z=z).to(u.pc).value
    if isinstance(app_mag,np.ndarray):
        cond = (np.abs(app_mag)!=99.)
        abs_mag = np.zeros(len(app_mag)) + 99.
        abs_mag[cond] = app_mag[cond] - 5*(np.log10(dist[cond]) - 1) + 2.5*np.log10(1+z[cond])
    else:
        abs_mag = app_mag - 5*(np.log10(dist) - 1) + 2.5*np.log10(1+z) if np.abs(app_mag)!=99. else app_mag
    return abs_mag

def get_app_from_abs(abs_mag,z):
    """
    Returns the apparent magnitude for a given absolute magnitude at redshift z.
    """
    dist = Planck15.luminosity_distance(z=z).to(u.pc).value
    app_mag = abs_mag + 5*(np.log10(dist) - 1) - 2.5*np.log10(1+z)
    return app_mag

def calc_zeropoint_factor(init_zp,outp_zp=23.9):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((outp_zp - init_zp) / 2.5)
    return fscale

def schechter_shape(M,alpha,Mst,**extras):

    x = np.power(10., -0.4*(M - Mst))
    return 0.4 * np.log(10) * np.power(x,alpha+1.) * np.exp(-x)

def schechter_func(M,alpha,Mst,phi,**extras):

    return np.power(10,phi) * schechter_shape(M,alpha=alpha,Mst=Mst)

def gen_LF_iCDF(alpha,Mst,mag_lim):

    M  = np.arange(-30,mag_lim,0.0005)
    LF = schechter_shape(M,alpha=alpha,Mst=Mst)
    int_LF  = scipy.integrate.cumtrapz(LF,M,initial=0)
    nint_LF = int_LF / int_LF[-1]
    icdf    = scipy.interpolate.interp1d(nint_LF, M)
    return icdf

def get_solid_angle(area):

    sr_in_deg2 = (np.pi/180.)**2
    return area * sr_in_deg2

def calc_window_flux(wave,spec,wave0,window=100):

    filt_sens = np.zeros_like(spec)
    filt_sens[np.abs(wave-wave0)<window] = 1
    flux = scipy.integrate.simps(spec*filt_sens*wave, wave) / scipy.integrate.simps(filt_sens*wave, wave)
    flux = (wave0**2/light) * flux
    return flux

def calc_filter_flux(wave,spec,camera,instr,filt):

    bandpass_label = "{:s},{:s},{:s}".format(camera,instr,filt)
    bandpass = S.ObsBandpass(bandpass_label)
    filt_wave,filt_sens = bandpass.wave, bandpass.throughput
    filt_interp = scipy.interpolate.interp1d(filt_wave, filt_sens, bounds_error=False, fill_value=0, kind='linear')

    filt_sens = filt_interp(wave)
    if np.all(filt_sens==0): return 0
    flux = scipy.integrate.simps(spec*filt_sens*wave, wave) / scipy.integrate.simps(filt_sens*wave, wave)
    flux = (bandpass.pivot()**2/light) * flux
    return flux

def show_available_filters():

    naxes = len([_ for camera in hst_filters for _ in hst_filters[camera].keys()])
    fig,axes = plt.subplots(naxes,1,figsize=(12,3.5*naxes),dpi=75,tight_layout=True)
    iaxes = 0

    for camera in hst_filters:
        for instr in hst_filters[camera]:
            xmin,xmax = 1e5,1e1
            for i,filt in enumerate(hst_filters[camera][instr]):

                bandpass_label = "{:s},{:s},{:s}".format(camera,instr,filt)
                bandpass = S.ObsBandpass(bandpass_label)

                xmin  = min(xmin,min(bandpass.wave[bandpass.throughput>1e-3]))
                xmax  = max(xmax,max(bandpass.wave[bandpass.throughput>1e-3]))
                ytext1= bandpass.throughput[np.argmin(np.abs(bandpass.wave-bandpass.pivot()))]*100*1.1
                ytext2= max(bandpass.throughput)*100*1.05
                ytext = ytext2 if np.abs(np.log10(ytext1/ytext2))<0.12 else ytext1
                color = plt.cm.gist_rainbow_r(np.linspace(0,1,len(hst_filters[camera][instr])))[i]

                axes[iaxes].plot(bandpass.wave,bandpass.throughput*100,
                                    color=color,lw=2,alpha=0.8,label=filt)
                axes[iaxes].text(bandpass.pivot(),ytext,filt,
                                    color=color,fontsize=12,fontweight=600,va="bottom",ha="center")
            axes[iaxes].text(0.005,0.98,"{:s} {:s}".format(camera,instr),
                                    fontsize=20,fontweight=600,va='top',ha='left',transform=axes[iaxes].transAxes)
            axes[iaxes].set_xlim(xmin-250,xmax+500)
            axes[iaxes].set_ylim(0,axes[iaxes].get_ylim()[1]*1.15)
            axes[iaxes].set_xlabel("Observed Wavelength [$\\AA$]",fontsize=18)
            axes[iaxes].set_ylabel("Throughput [%]",fontsize=18)
            [_.set_fontsize(14) for _ in axes[iaxes].get_xticklabels()+axes[iaxes].get_yticklabels()]
            iaxes += 1

def show_available_LFs():

    fig,ax = plt.subplots(1,1,figsize=(10,7),dpi=75,tight_layout=True)

    mags = np.arange(-50,0,0.05)

    for i,LF in enumerate(LFs):
        ax.plot(mags,schechter_func(mags,alpha=LF["alpha"],Mst=LF["Mst"],phi=LF["phi"]),
                    color=plt.cm.gist_rainbow_r(np.linspace(0.2,1.0,len(LFs)))[i],
                    lw=2,alpha=0.8,label="{0[ref]:s} [{0[wave]:s}; {0[zmin]:.2f}<z<{0[zmax]:.2f}]".format(LF))

    ax.set_xlabel("Absolute UV mag [AB]",fontsize=18)
    ax.set_ylabel("$\\phi$ [Mpc$^{-3}$ mag$^{-1}$]",fontsize=18)
    ax.set_yscale("log")
    ax.set_xlim(-15,-23.5)
    ax.set_ylim(1e-5,1e-1)
    [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    leg = ax.legend(fontsize=16,ncol=1,loc=3,framealpha=0,
                     handlelength=0,handletextpad=0)

    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_fontweight(600)
        txt.set_color(hndl.get_color())
        hndl.set_visible(False)

if __name__ == '__main__':

    show_available_filters()
    show_available_LFs()

    plt.show()
