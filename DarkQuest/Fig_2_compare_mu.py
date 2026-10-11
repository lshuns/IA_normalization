# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2026-10-01 12:00:00
# @Last Modified by:   lshuns
# @Last Modified time: 2026-10-10 22:34:57

### Test the angular structure of P_hE(k,mu) directly, for the th (3D) and obs (2D) normalizations.
###    For th, P_hE(k,mu) = (1 - mu^2) g(k), so after dividing out (1 - mu^2) every mu bin should match the first:
###       r_i = < [P_hE(k,mu_i)/(1 - mu_i^2)] / [P_hE(k,mu_1)/(1 - mu_1^2)] >  = 1   for A = th
###    rows = th / obs, columns = mass bin, colour = mu bin (selectable), single shape type and redshift
###    ratios are formed per realization, then averaged; errors from the scatter across realizations

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

## Which mu bins to plot
# mu_plot = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95] 
mu_plot = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]

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
shape = ["RIAC","RIVC","RnonI","SIAC","SIVC","SnonI"] #only RIAC has P(k,mu) on disk
snapshot = ['_S000_','_S004_','_S020_']
redshifts = ['z=1.5','z=0.5','z=0.0']
real = [1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23,24] #12-15 not available
nreal = len(real)

## the two cases compared: 
cases = [("pk_all_kmu_NormEach3D", r'{\rm th}'),
         ("pk_all_kmu_NormEach2D", r'{\rm obs}')]

## mu binning of the files: 10 equal bins in [0, 1], referred to by their centres
MU_EDGES = np.linspace(0.0,1.0,11)
mu_c = 0.5*(MU_EDGES[:-1] + MU_EDGES[1:])
nmu = len(mu_c)

## coarse k binning, as in B_compare_ratio.py: 6 linearly spaced bins in [0, 2] h/Mpc
K_EDGES = np.linspace(0.0,2.0,7)
nk = len(K_EDGES) - 1

## y range: the last mu bin (1 - mu^2 ~ 0.1) is noisy at high mass and would set the scale;
##    points below the range are marked with a triangle at the bottom edge
YLIM = (0.7,1.2)

## mu bins coloured from dark (low mu) to light (high mu)
mu_cmap = plt.cm.viridis
mu_colors = mu_cmap(np.linspace(0.0,0.9,nmu-1))

## helpers
def get_fname(case_dir,idx_mbin,idx_shape,idx_snapshot,ireal):
    return f"{DATA_DIR}/" + case_dir + "/PhE2D_R" + str(ireal).zfill(3) + snapshot[idx_snapshot] \
    + "CIC1024_kmin0.0_kmax3_nbin200_" + mbin[idx_mbin] + "_shape" + shape[idx_shape] + ".dat"


def load_kmu(fname):
    """The P(k,mu) files hold 5 blocks of (200 k bins) x (10 mu bins), separated by blank
    lines: mean k, mean |mu|, P, error, number of modes, in each (k, mu) cell."""
    k, mu, p, err, nmodes = np.loadtxt(fname).reshape(5,-1,nmu)
    assert np.all((mu >= 0) & (mu <= 1)) and np.all(nmodes == np.round(nmodes)), \
        'unexpected block layout in ' + fname
    return k, p, nmodes


def rebin(k,p,nmodes):
    """Mode-weighted average onto K_EDGES, separately in each mu bin."""
    kfine = np.sum(nmodes*k,axis=1)/np.sum(nmodes,axis=1)
    idx = np.digitize(kfine,K_EDGES) - 1
    kc = np.zeros(nk)
    pc = np.zeros((nk,nmu))
    for j in range(nk):
        s = idx == j
        kc[j] = np.sum(nmodes[s]*k[s])/np.sum(nmodes[s])
        pc[j] = np.sum(nmodes[s]*p[s],axis=0)/np.sum(nmodes[s],axis=0)
    return kc, pc


def get_ratios(case_dir,idx_mbin,idx_shape,idx_snapshot):
    """r_i per realization, shape (nreal, nk, nmu); the first mu bin is 1 by construction."""
    r = np.zeros((nreal,nk,nmu))
    for re in enumerate(real):
        idx = re[0]
        kc, pc = rebin(*load_kmu(get_fname(case_dir,idx_mbin,idx_shape,idx_snapshot,re[1])))
        g = pc/(1.0 - mu_c**2) # divide out the th angular dependence
        r[idx] = g/g[:,:1]
    return kc, r


def plotter_mu(idx_snapshot=1,idx_shape=0,mu_plot=None,outdir=None):
    """One figure for a single shape type and redshift: r_i against k, one row per
    case (th, obs), one column per mass bin, one colour per mu bin.
    mu_plot: centres of the mu bins to show (any subset of 0.15, ..., 0.95); None shows all.
    Each mu bin keeps its colour whatever the selection, so figures stay comparable."""
    if mu_plot is None:
        i_plot = list(range(1,nmu))
    else:
        i_plot = [int(np.argmin(np.abs(mu_c - x))) for x in mu_plot]
        assert all(i > 0 and np.isclose(mu_c[i],x) for i, x in zip(i_plot,mu_plot)), \
            'mu_plot must be bin centres from %s (the first bin is the reference)'%mu_c[1:]
        i_plot = sorted(set(i_plot))
    # gather first: res has shape (n_case, n_mbin, nreal, nk, nmu)
    res = np.full((len(cases),len(mbin),nreal,nk,nmu),np.nan)
    for c, (case_dir,label) in enumerate(cases):
        for m in range(len(mbin)):
            kc, res[c,m] = get_ratios(case_dir,m,idx_shape,idx_snapshot)

    fig, axes = plt.subplots(len(cases),len(mbin),figsize=(TEXT_WIDTH,0.6*TEXT_WIDTH),
                             sharex=True,sharey=True)
    for c, (case_dir,label) in enumerate(cases):
        for m in range(len(mbin)):
            ax = axes[c,m]
            for j, i in enumerate(i_plot):
                vals = res[c,m,:,:,i]
                # mean over realizations, error on the mean from the scatter across the 20 sims
                ## small offset so the mu bins do not overlap within a k bin
                kk = kc + 0.025*(j - (len(i_plot)-1)/2.)
                mean = np.mean(vals,axis=0)
                ax.errorbar(kk, mean, yerr = np.std(vals,axis=0,ddof=1)/np.sqrt(nreal),
                            fmt='o', markersize=ms, markeredgecolor=ec,
                            color=mu_colors[i-1], elinewidth=ew, capsize=cs)
                low = mean < YLIM[0] 
                if np.any(low): 
                    ax.plot(kk[low], np.full(np.sum(low),YLIM[0] + 0.02*(YLIM[1]-YLIM[0])), 'v',
                            markersize=ms+1, color=mu_colors[i-1], clip_on=False)
            ax.axhline(1,linestyle='dashed',color='black',lw=1)
            if c == 0:
                lo, hi = mass_bins[m].split('-')
                ax.set_title(r'$\log_{10}M_{\rm h}\in[%s, %s]$'%(lo,hi))
            if c == len(cases)-1:
                ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        axes[c,0].set_ylabel(r'$r_i^{%s}$'%label)
    ax.set_xlim(K_EDGES[0],K_EDGES[-1])
    ax.set_xticks([0.5,1.0,1.5]) 
    ax.set_ylim(*YLIM)
    ax.set_yticks([0.8,0.9,1.0,1.1]) 
    axes[1,0].text(0.04,0.04,shape[idx_shape] + ', $' + redshifts[idx_snapshot] + '$',
                   transform=axes[1,0].transAxes,ha='left',va='bottom')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.04,hspace=0.04,right=0.88) 

    pos0, pos1 = axes[0,-1].get_position(), axes[-1,-1].get_position()
    cax = fig.add_axes([0.895,pos1.y0,0.015,pos0.y1-pos1.y0])
    nplot = len(i_plot)
    norm = mpl.colors.BoundaryNorm(np.arange(nplot+1),nplot)
    cmap_plot = mpl.colors.ListedColormap([mu_colors[i-1] for i in i_plot])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap_plot),cax=cax,
                      ticks=np.arange(nplot)+0.5)
    cb.set_ticklabels(['%.2f'%mu_c[i] for i in i_plot])
    cb.set_label(r'$\bar{\mu}_i$')
    cb.ax.tick_params(direction='out',right=True)
    if outdir is None:
        plt.show()
    else:
        os.makedirs(outdir,exist_ok=True)
        fname = os.path.join(outdir,'r_mu_' + shape[idx_shape] + snapshot[idx_snapshot].rstrip('_') + '.png')
        fig.savefig(fname)
        print('saved ' + fname)
    plt.close(fig)


## run
for idx_snapshot in range(len(snapshot)):
    plotter_mu(idx_snapshot=idx_snapshot,
                idx_shape=0, # RIAC, the only shape type with P(k,mu)
                mu_plot=mu_plot, # mu bin centres to show, None = all
                outdir=outdir)