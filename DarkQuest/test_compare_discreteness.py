# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2026-09-30 12:00:00
# @Last Modified by:   lshuns
# @Last Modified time: 2026-10-11 21:39:08

### Test whether the residual symmetry-relation signal for the th (3D) normalization is a discreteness effect.
###    Here the estimator's sum over the discrete FFT modes is done exactly for the (1 - mu^2) template
###    (whose continuum mu integral gives zero), on the same grid and k binning as the measurement,
###    and compared with the measured th ratios.
###    First plot (full, averaged k): measured ratios against the mode sum and the integral, in the coarse k bins
###        rows = ratio, columns = mass bin, all redshifts per panel, single shape type
###    Second plot (zoom-in, averaged k): the mode sum alone in the same coarse bins, y axis in units of 1e-4
###    Third plot (individual k): the mode sum in the fine bins below k = 2 h/Mpc, before averaging (the same for all samples)

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
plt.rcParams['axes.titlesize'] = 'medium'

## general sample info
mbin = ['mbin0','mbin1','mbin2','mbin3']
mass_bins = ['12-12.5','12.5-13','13-13.5','13.5-14']
shape = ["RIAC","RIVC","RnonI","SIAC","SIVC","SnonI"]
snapshot = ['_S000_','_S004_','_S020_']
redshifts = ['z=1.5','z=0.5','z=0.0']
real = [1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23,24] #12-15 not available
nreal = len(real)

## th only: 3D normalization, where the symmetry relations hold exactly in the continuum
NML = "NormEach3D"
cmap = plt.cm.Blues

## redshifts: shade of the colour map (darker = higher z) and marker symbol
z_shades = [0.95,0.7,0.45]
z_markers = ['o','s','^']

## estimator grid, as in the measurement files
LBOX = 1000.0                          
KF = 2*np.pi/LBOX                      
FINE_EDGES = np.linspace(0.0,3.0,201)  # the kmin0.0_kmax3_nbin200 binning

## coarse k binning
K_EDGES = np.linspace(0.0,2.0,7)
nk = len(K_EDGES) - 1

## the two ratios: (axis label, file tag)
quantities = [(r'$(P_{hE}^{\ell=0}+P_{hE}^{\ell=2})\,/\,P_{hE}^{\ell=0}$', 'P0pP2_over_P0'),
              (r'$P_{hE}^{\ell=4}\,/\,P_{hE}^{\ell=0}$',                  'P4_over_P0')]

## zoom-in figure: the ratios in units of 1e-4
SCALE = 1e4 

## helpers
def get_fname(idx_mbin,idx_shape,idx_snapshot,ell,ireal):
    return f"{DATA_DIR}/output_" + NML + "/pk_all/PhE" + str(ell) \
    + '_R' + str(ireal).zfill(3) + snapshot[idx_snapshot] + "CIC1024_kmin0.0_kmax3_nbin200" \
    + "_" + mbin[idx_mbin] + "_shape" + shape[idx_shape] + ".dat"


def lattice_mode_sums(nfine):
    """Sum the multipole-estimator weights of the (1 - mu^2) template over the grid modes."""
    ## k binning info
    kcut = FINE_EDGES[nfine]
    dk = FINE_EDGES[1] - FINE_EDGES[0]

    ## squared xy length in units of KF
    nmax = int(np.ceil(kcut/KF))
    n1 = np.arange(-nmax,nmax+1)
    nx, ny = np.meshgrid(n1,n1,indexing='ij')
    r2xy = (nx**2 + ny**2).astype(float)

    ## loop over the nz planes
    sums = np.zeros((5,nfine))
    for nz in range(-nmax,nmax+1):
        ## |k| of every mode in this plane
        k = KF*np.sqrt(r2xy + nz**2)
        s = (k > 0) & (k < kcut)
        k = k[s]
        ## angular part of the th template P_hE(k) = (1 - mu^2) g(|k|)
        mu2 = (KF*nz/k)**2
        w = 1.0 - mu2
        ## fine bin of each mode
        idx = (k/dk).astype(int)
        sums[0] += np.bincount(idx,minlength=nfine) # number of modes
        sums[1] += np.bincount(idx,k,minlength=nfine) # sum of |k|
        sums[2] += np.bincount(idx,w,minlength=nfine) # l=0 estimator weight (1 - mu^2) L_0
        sums[3] += np.bincount(idx,5*w*0.5*(3*mu2-1),minlength=nfine) # l=2 estimator weight 5 (1 - mu^2) L_2(mu)
        sums[4] += np.bincount(idx,9*w*(35*mu2*mu2-30*mu2+3)/8.,minlength=nfine) # l=4 estimator weight 9 (1 - mu^2) L_4(mu)
    
    sums[:2] /= 2.
    sums[2:] /= 2.
    return sums


def rebin(k,p,nmodes):
    """Mode-weighted average of fine k bins to the coarse bins."""
    idx = np.digitize(k,K_EDGES) - 1
    kc = np.array([np.sum((nmodes*k)[idx==j])/np.sum(nmodes[idx==j]) for j in range(nk)])
    pc = np.array([np.sum((nmodes*p)[idx==j])/np.sum(nmodes[idx==j]) for j in range(nk)])
    return kc, pc


def get_measured(idx_mbin,idx_shape,idx_snapshot):
    """Measured (P0+P2)/P0 and P4/P0 per realization, each (nreal, nk), and the
    realization-averaged fine-binned P0 (for the weighting of the prediction)."""
    r = np.zeros((2,nreal,nk))
    p0_fine = 0.
    for re in enumerate(real):
        idx = re[0]
        ar = [np.loadtxt(get_fname(idx_mbin,idx_shape,idx_snapshot,ell,re[1])) for ell in (0,2,4)]
        k, nmodes = ar[0][:,0], ar[0][:,3]
        kc, p0 = rebin(k,ar[0][:,1],nmodes)
        _, p2 = rebin(k,ar[1][:,1],nmodes)
        _, p4 = rebin(k,ar[2][:,1],nmodes)
        r[0,idx] = (p0 + p2)/p0
        r[1,idx] = p4/p0
        p0_fine = p0_fine + ar[0][:,1]/nreal
    return kc, r, k, nmodes, p0_fine


def get_predicted(sums,p0_fine):
    """Mode-sum prediction of the two ratios in the coarse bins, for the (1-mu^2) g(|k|)
    template. g is taken constant within a fine bin and set by the measured P0 there, so
    the coarse bins weight the fine bins as the measurement does."""
    nf = sums.shape[1]
    nmodes, kfine = sums[0], sums[1]/sums[0]
    d0, d2, d4 = sums[2]/nmodes, sums[3]/nmodes, sums[4]/nmodes
    g = p0_fine[:nf]/d0
    _, p0 = rebin(kfine,g*d0,nmodes)
    _, p2 = rebin(kfine,g*d2,nmodes)
    _, p4 = rebin(kfine,g*d4,nmodes)
    return (p0 + p2)/p0, p4/p0


def get_all(idx_shape=0):
    """Lattice mode sums, plus the measured th ratios (meas) and their mode-sum prediction
    (pred) in the coarse k bins, for every redshift and mass bin of one shape type."""
    ## number of fine bins that feed the coarse bins (centre below the last coarse edge)
    nfine = int(np.sum(0.5*(FINE_EDGES[:-1] + FINE_EDGES[1:]) < K_EDGES[-1]))
    ## the lattice mode sums of the (1 - mu^2) template
    sums = lattice_mode_sums(nfine)

    meas = np.full((2,len(snapshot),len(mbin),nreal,nk),np.nan)
    pred = np.full((2,len(snapshot),len(mbin),nk),np.nan)
    kc = None
    for j in range(len(snapshot)):
        for m in range(len(mbin)):
            ## load the measurement and check that its fine-bin mode counts and mean k match the lattice
            kc, r, kfile, nfile, p0_fine = get_measured(m,idx_shape,j)
            assert np.allclose(nfile[:nfine],sums[0],rtol=1e-5,atol=0) \
                and np.allclose(kfile[:nfine],sums[1]/sums[0],rtol=1e-5), \
                'lattice modes differ from the estimator ones'
            meas[:,j,m] = r
            pred[:,j,m] = get_predicted(sums,p0_fine)
    return kc, meas, pred, sums


def plotter_discreteness(kc,meas,pred,idx_shape=0,outdir=None):
    """Full view, averaged k: measured th ratios (points) against the exact mode-sum
    prediction (lines) and the continuum integral (zero, dashed), in the coarse bins."""

    fig, axes = plt.subplots(len(quantities),len(mbin),figsize=(TEXT_WIDTH,0.6*TEXT_WIDTH),sharex=True,sharey='row')
    for q, (ylabel,_) in enumerate(quantities):
        for m in range(len(mbin)):
            ax = axes[q,m]
            for j in range(len(snapshot)):
                color = cmap(z_shades[j])
                vals = meas[q,j,m]
                ## measured results
                ax.errorbar(kc + 0.035*(j-1), np.mean(vals,axis=0),
                            yerr = np.std(vals,axis=0,ddof=1)/np.sqrt(nreal),
                            fmt=z_markers[j], markersize=ms, markeredgecolor=ec,
                            label='$' + redshifts[j] + '$',
                            color=color, elinewidth=ew, capsize=cs)
                ## exact mode-sum prediction
                ax.plot(kc, pred[q,j,m], '-', color=color, lw=1.2)
            ## perfect integral (zero)
            ax.axhline(0,linestyle='dashed',color='black',lw=1) 
            if q == 0:
                lo, hi = mass_bins[m].split('-')
                ax.set_title(r'$\log_{10}M_{\rm h}\in[%s, %s]$'%(lo,hi))
            if q == len(quantities)-1:
                ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        axes[q,0].set_ylabel(ylabel)
    ax.set_xlim(K_EDGES[0],K_EDGES[-1])
    ax.set_xticks([0.5,1.0,1.5]) 
    axes[1,0].text(0.04,0.96,'th, ' + shape[idx_shape],transform=axes[1,0].transAxes,ha='left',va='top')
    handles, labels = axes[1,0].get_legend_handles_labels()
    handles += [Line2D([],[],color='grey',lw=1.2), Line2D([],[],color='black',lw=1,ls='dashed')]
    labels += ['mode sum', 'integral']
    leg = axes[1,0].legend(handles, labels, ncol=2, columnspacing=0.6, handletextpad=0.3,
                           handlelength=1.5, borderaxespad=0.3,
                           loc='upper left', bbox_to_anchor=(0.0,0.88)) 
    leg.set_in_layout(False) 
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.04,hspace=0.04) 
    if outdir is None:
        plt.show()
    else:
        os.makedirs(outdir,exist_ok=True)
        fname = os.path.join(outdir,'test_th_modesum_' + shape[idx_shape] + '.png')
        fig.savefig(fname)
        print('saved ' + fname)
    plt.close(fig)


def plotter_zoom(kc,pred,idx_shape=0,outdir=None):
    """Zoom-in, averaged k: the mode-sum prediction alone (the solid lines of the full
    figure) in the coarse bins, with the y axis in units of 1/SCALE."""
    fig, axes = plt.subplots(len(quantities),len(mbin),figsize=(TEXT_WIDTH,0.6*TEXT_WIDTH),sharex=True,sharey='row')
    for q, (ylabel,_) in enumerate(quantities):
        for m in range(len(mbin)):
            ax = axes[q,m]
            for j in range(len(snapshot)):
                ax.plot(kc + 0.035*(j-1), SCALE*pred[q,j,m], '-', marker=z_markers[j], markersize=ms,
                        color=cmap(z_shades[j]), lw=1, label='$' + redshifts[j] + '$')
            ax.axhline(0,linestyle='dashed',color='black',lw=1)
            if q == 0:
                lo, hi = mass_bins[m].split('-')
                ax.set_title(r'$\log_{10}M_{\rm h}\in[%s, %s]$'%(lo,hi))
            else:
                ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        axes[q,0].set_ylabel(r'$10^{%d}\times$'%round(np.log10(SCALE)) + ylabel)
    ax.set_xlim(K_EDGES[0],K_EDGES[-1])
    ax.set_xticks([0.5,1.0,1.5])
    axes[0,0].legend(loc='lower right',title='th, ' + shape[idx_shape] + ': mode sum')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.04,hspace=0.04)
    if outdir is None:
        plt.show()
    else:
        os.makedirs(outdir,exist_ok=True)
        fname = os.path.join(outdir,'test_th_modesum_zoom_' + shape[idx_shape] + '.png')
        fig.savefig(fname)
        print('saved ' + fname)
    plt.close(fig)


def plotter_fine(sums,outdir=None):
    """the mode sum in each fine k bin, before averaging into the coarse bins."""
    nmodes, kfine = sums[0], sums[1]/sums[0]
    d0, d2, d4 = sums[2]/nmodes, sums[3]/nmodes, sums[4]/nmodes
    ratios = [(d0 + d2)/d0, d4/d0]

    fig, axes = plt.subplots(1,len(quantities),figsize=(TEXT_WIDTH,0.35*TEXT_WIDTH))
    for q, (ylabel,_) in enumerate(quantities):
        ax = axes[q]
        ax.plot(kfine, ratios[q], '.-', color='grey', lw=0.8, markersize=3, label='mode sum (fine bins)')
        ax.axhline(0,linestyle='dashed',color='black',lw=1,label='integral')
        for e in K_EDGES[1:-1]: #coarse bin edges, for reference
            ax.axvline(e,color='black',lw=0.5,alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('symlog',linthresh=1e-3)
        ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0,1,0.88))
    if outdir is None:
        plt.show()
    else:
        os.makedirs(outdir,exist_ok=True)
        fname = os.path.join(outdir,'test_th_modesum_fine.png')
        fig.savefig(fname)
        print('saved ' + fname)
    plt.close(fig)


## run
idx_shape = 0 # RIAC
kc, meas, pred, sums = get_all(idx_shape)
# averaged k with measurements and predictions
plotter_discreteness(kc,meas,pred,idx_shape=idx_shape,outdir=outdir)
# averaged k with only predictions (zoomed in)
plotter_zoom(kc,pred,idx_shape=idx_shape,outdir=outdir)
# fine k bins before averaging
plotter_fine(sums,outdir=outdir)
