# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2026-08-05 16:38:35
# @Last Modified by:   lshuns
# @Last Modified time: 2026-10-10 23:32:42

### Compare the intrinsic alignment amplitude A_IA with and without the 3D misalignment randomization.
###    A_IA vs halo mass, one point per mass bin for: un-randomized, sigma_MF = 0.37, sigma_MF = 0.64
###    single redshift, 2D `sim' normalization (NormEach), RIAC shapes (the only randomized shape type)

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

## I/O
DATA_DIR = "/disks/shear10/ssli/DarkQuest_IA"
outdir = './plots'

## plotting config
## A&A page geometry (aa.cls): text width 184 mm = two 90 mm columns + 4 mm gap;
##    main text 10 pt, captions 9 pt. Figures are made at their printed size.
COLUMN_WIDTH = 90/25.4   # inch
TEXT_WIDTH = 184/25.4    # inch
ms=3
ec=None
ew=0.8
cs=1.5
plt.rcParams['font.size'] = 9
plt.rcParams["figure.dpi"] = 150      # display DPI
plt.rcParams["savefig.dpi"] = 300     # saved figures

## general sample info
mbin = ['mbin0','mbin1','mbin2','mbin3']
mass_bins = ['12-12.5','12.5-13','13-13.5','13.5-14']
mass_centers = np.array([12.25,12.75,13.25,13.75]) 
shape = ["RIAC","RIVC","RnonI","SIAC","SIVC","SnonI"]
snapshot = ['_S000_','_S004_','_S020_']
redshifts = ['z=1.5','z=0.5','z=0.0']
z_values = np.array([1.5,0.5,0.0]) 
real = [1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23,24] #12-15 not available
nreal = len(real)

## the three cases compared: (output directory, misalignment sigma or None, legend label)
label_norm = '2D normalization'
cases = [("NormEach",             None,   r'un-randomized'),
         ("NormEach_3DRandomized", '0p37', r'$\sigma_{\rm MF}=0.37$'),
         ("NormEach_3DRandomized", '0p64', r'$\sigma_{\rm MF}=0.64$')]

## linear regime cut in h/Mpc
KMAX_LIN = 0.05 
## cosmology, for the b_1^g -> A_IA conversion
OMEGA_M = 0.3089 # Planck15
C1_RHO_CR = 0.0134

## helpers
trapezoid = getattr(np,'trapezoid',None) or np.trapz # np.trapz is removed in numpy 2
def growth_factor(z,omega_m=OMEGA_M,npts=4096):
    """Linear growth factor for flat LCDM, normalized to D(z=0) = 1.

    D(a) propto (5/2) Omega_m E(a) * int_0^a da' / (a'^3 E(a')^3).
    """
    def unnormalized(a):
        ap = np.linspace(1e-8,a,npts) 
        e = np.sqrt(omega_m/ap**3 + (1.0-omega_m))
        return 2.5*omega_m*np.sqrt(omega_m/a**3 + (1.0-omega_m)) \
               * trapezoid(1.0/(ap**3 * e**3),ap)
    a = 1.0/(1.0+np.atleast_1d(np.asarray(z,dtype=float)))
    return np.array([unnormalized(ai) for ai in a])/unnormalized(1.0)


def b1g_to_AIA(b1g,z):
    """Invert b_1^g = -2 C1 A_IA Omega_m rho_cr,0 / D(z)."""
    return -b1g*growth_factor(z)/(2.0*C1_RHO_CR*OMEGA_M)


def get_fname(nml,sigma,idx_mbin,idx_shape,idx_snapshot,spec,ireal):
    fname = f"{DATA_DIR}/output_" + nml + "/pk_all/" + spec \
    + '_R' + str(ireal).zfill(3) + snapshot[idx_snapshot] + "CIC1024_kmin0.0_kmax3_nbin200"
    if spec != 'Pdd': 
        fname += "_" + mbin[idx_mbin] + "_shape" + shape[idx_shape]
        if sigma is not None:
            fname += "_3DmisalignSigmaMF" + sigma + "_seed12345"
    return fname + ".dat"


def get_AIA(nml,sigma,idx_mbin,idx_shape,idx_snapshot):
    """A_IA per realization, from b_1^g averaged over the k < KMAX_LIN bins."""
    b1g = np.zeros(nreal)
    for re in enumerate(real):
        idx = re[0]
        dd = np.loadtxt(get_fname(nml,sigma,idx_mbin,idx_shape,idx_snapshot,'Pdd',re[1]))
        dE = np.loadtxt(get_fname(nml,sigma,idx_mbin,idx_shape,idx_snapshot,'PdE0',re[1]))
        assert np.allclose(dd[:,0],dE[:,0]), 'k grids of Pdd and PdE0 differ'
        lin = dd[:,0] < KMAX_LIN
        # ratio per bin first, then averaged over bins
        b1g[idx] = 1.5*np.mean(dE[lin,1]/dd[lin,1])
    return b1g_to_AIA(b1g,z_values[idx_snapshot])


def plotter_AIA_randomization(idx_snapshot=1,idx_shape=0,outdir=None):
    """A_IA vs halo mass, one series per randomization case."""
    AIA = np.full((len(cases),len(mbin),nreal),np.nan)
    for c, (nml,sigma,label) in enumerate(cases):
        for m in range(len(mbin)):
            if not os.path.exists(get_fname(nml,sigma,m,idx_shape,idx_snapshot,'PdE0',real[0])):
                print('no data, skipped: ' + label + ' / ' + mass_bins[m])
                continue
            AIA[c,m,:] = get_AIA(nml,sigma,m,idx_shape,idx_snapshot)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH,0.82*COLUMN_WIDTH))
    xs, tops = [], []
    for c, (nml,sigma,label) in enumerate(cases):
        # mean over realizations, error on the mean from the scatter across the 20 sims
        ## small offset so the three cases do not overlap within a mass bin
        x = mass_centers + 0.06*(c-1)
        mean = np.mean(AIA[c],axis=1)
        err = np.std(AIA[c],axis=1,ddof=1)/np.sqrt(nreal)
        ax.errorbar(x, mean, yerr = err,
                    fmt='o', markersize=ms, markeredgecolor=ec, label=label,
                    color='C'+str(c), elinewidth=ew, capsize=cs)
        xs.append(x)
        tops.append(mean + err)
    ax.set_xlabel(r'$\log_{10}(M_{\rm h}\,/\,h^{-1}M_{\odot})$')
    ax.set_xticks(mass_centers)
    ax.set_ylabel(r'$A_{\rm IA}$')
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto',steps=[1,2,5,10]))
    txt = ax.text(0.04,0.96,shape[idx_shape] + f', {label_norm}'
            + '\n' + r'$' + redshifts[idx_snapshot] + '$, '
             + r'$k < %g\,h\,{\rm Mpc}^{-1}$'%KMAX_LIN,
            transform=ax.transAxes,ha='left',va='top')
    fig.tight_layout()

    fig.canvas.draw()
    to_axes = ax.transAxes.inverted()
    txt_box = txt.get_window_extent().transformed(to_axes)
    leg = ax.legend(loc='upper left',bbox_to_anchor=(0.0,txt_box.y0))
    fig.canvas.draw()
    leg_box = leg.get_window_extent().transformed(to_axes)
    xs, tops = np.concatenate(xs), np.concatenate(tops)
    x_axes = (ax.transData + to_axes).transform(np.column_stack([xs,np.zeros_like(xs)]))[:,0]
    under = (x_axes < max(txt_box.x1,leg_box.x1) + 0.03) & np.isfinite(tops)
    lo, hi = ax.get_ylim()
    if np.any(under):
        hi = max(hi, lo + (np.max(tops[under]) - lo)/(leg_box.y0 - 0.04))
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    if outdir is None:
        plt.show()
    else:
        os.makedirs(outdir,exist_ok=True)
        fbase = 'AIA_randomization_' + shape[idx_shape] + snapshot[idx_snapshot].rstrip('_')
        fig.savefig(os.path.join(outdir,fbase + '.png'))
        print('saved ' + os.path.join(outdir,fbase + '.png'))
    plt.close(fig)


## run
for idx_snapshot in range(len(snapshot)):
    plotter_AIA_randomization(idx_snapshot=idx_snapshot,
                            idx_shape=0, # RIAC, the only randomized shape type
                            outdir=outdir) 