# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2026-10-11 22:05:04
# @Last Modified by:   lshuns
# @Last Modified time: 2026-10-11 22:17:05

### Shape-definition dependence of the residual P_hE symmetry relations, th - obs.
###    (P_hE^0 + P_hE^2) / P_hE^0   and   P_hE^4 / P_hE^0, th minus obs, for the six shape types
###    one figure per redshift; rows = ratio, columns = mass bin, one colour per shape type

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
plt.rcParams['axes.titlesize'] = 'medium'

## general sample info
mbin = ['mbin0','mbin1','mbin2','mbin3']
mass_bins = ['12-12.5','12.5-13','13-13.5','13.5-14']
shape = ["RIAC","RIVC","RnonI","SIAC","SIVC","SnonI"]
snapshot = ['_S000_','_S004_','_S020_']
redshifts = ['z=1.5','z=0.5','z=0.0']
real = [1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23,24] #12-15 not available
nreal = len(real)

## the two normalizations, differenced per realization (same halos and shapes), so most of
##    the sample variance common to th and obs cancels
##    th  = 3D normalization, projected tensor divided by the 3D trace, linear in the 3D shape
##    obs = 2D normalization, divided by the projected trace, as for observed ellipticities
norms = [('th',  'NormEach3D'),
         ('obs', 'NormEach')]

## coarse k binning
K_EDGES = np.linspace(0.0,2.0,7)
nk = len(K_EDGES) - 1

## helpers
def get_fname(nml,idx_mbin,idx_shape,idx_snapshot,ell,ireal):
    return f"{DATA_DIR}/output_" + nml + "/pk_all/PhE" + str(ell) \
    + '_R' + str(ireal).zfill(3) + snapshot[idx_snapshot] + "CIC1024_kmin0.0_kmax3_nbin200" \
    + "_" + mbin[idx_mbin] + "_shape" + shape[idx_shape] + ".dat"


def rebin(ar):
    """Mode-weighted average of the nbin200 spectrum onto K_EDGES (fine bins
    assigned by their centre). Returns the coarse k centres and P."""
    k, p, nmodes = ar[:,0], ar[:,1], ar[:,3]
    idx = np.digitize(k,K_EDGES) - 1
    kc = np.zeros(nk)
    pc = np.zeros(nk)
    for j in range(nk):
        s = idx == j
        kc[j] = np.sum(nmodes[s]*k[s])/np.sum(nmodes[s])
        pc[j] = np.sum(nmodes[s]*p[s])/np.sum(nmodes[s])
    return kc, pc


def get_ratios(nml,idx_mbin,idx_shape,idx_snapshot):
    """(P0+P2)/P0 and P4/P0 per realization, each of shape (nreal, nk)."""
    r02 = np.zeros((nreal,nk))
    r4 = np.zeros((nreal,nk))
    for re in enumerate(real):
        idx = re[0]
        ar = [np.loadtxt(get_fname(nml,idx_mbin,idx_shape,idx_snapshot,ell,re[1])) for ell in (0,2,4)]
        assert np.allclose(ar[0][:,0],ar[1][:,0]) and np.allclose(ar[0][:,0],ar[2][:,0]), \
            'k grids of the PhE multipoles differ'
        kc, p0 = rebin(ar[0])
        _, p2 = rebin(ar[1])
        _, p4 = rebin(ar[2])
        r02[idx] = (p0 + p2)/p0
        r4[idx] = p4/p0
    return kc, r02, r4


def plotter_shapes(idx_snapshot=1,outdir=None):
    """One figure (th - obs) for a single redshift: one row per ratio, one column
    per mass bin, each panel showing all shape types."""
    # gather first: res has shape (n_norm, n_shape, n_ratio, n_mbin, nreal, nk)
    res = np.full((len(norms),len(shape),2,len(mbin),nreal,nk),np.nan)
    for n, (_,nml) in enumerate(norms):
        for s in range(len(shape)):
            for m in range(len(mbin)):
                kc, res[n,s,0,m], res[n,s,1,m] = get_ratios(nml,m,s,idx_snapshot)
    vals_all = res[0] - res[1] # th - obs, (n_shape, n_ratio, n_mbin, nreal, nk)

    # (y label, short name), one per row
    quantities = [(r'$(P_{hE}^{\ell=0}+P_{hE}^{\ell=2})\,/\,P_{hE}^{\ell=0}$', 'P0pP2_over_P0'),
                  (r'$P_{hE}^{\ell=4}\,/\,P_{hE}^{\ell=0}$',                  'P4_over_P0')]

    # mean over realizations, error on the mean from the scatter across the 20 sims
    mean = np.mean(vals_all,axis=3)                          # (n_shape, n_ratio, n_mbin, nk)
    err = np.std(vals_all,axis=3,ddof=1)/np.sqrt(nreal)
    # y range per row from the points with typical errors, so a few very noisy ones do not set it
    ylims = []
    for q in range(len(quantities)):
        good = err[:,q] < 10*np.median(err[:,q])
        lo = min(np.min((mean[:,q] - err[:,q])[good]), 0.)
        hi = max(np.max((mean[:,q] + err[:,q])[good]), 0.)
        ylims.append((lo - 0.08*(hi - lo), hi + 0.08*(hi - lo)))
    fig, axes = plt.subplots(len(quantities),len(mbin),figsize=(TEXT_WIDTH,0.6*TEXT_WIDTH),
                             sharex=True,sharey='row')
    for q, (ylabel,_) in enumerate(quantities):
        for m in range(len(mbin)):
            ax = axes[q,m]
            y0, y1 = ylims[q]
            for s in range(len(shape)):
                # small offset so the six shapes do not overlap within a k bin
                kk = kc + 0.035*(s - (len(shape)-1)/2.)
                ax.errorbar(kk, mean[s,q,m], yerr = err[s,q,m],
                            fmt='o', markersize=ms, markeredgecolor=ec, label=shape[s],
                            color='C'+str(s), elinewidth=ew, capsize=cs)
                # arrows at the panel edge for points outside the y range
                for off, ypos, mk in [(mean[s,q,m] < y0, y0 + 0.05*(y1 - y0), r'$\downarrow$'),
                                      (mean[s,q,m] > y1, y1 - 0.05*(y1 - y0), r'$\uparrow$')]:
                    if np.any(off):
                        ax.plot(kk[off], np.full(np.sum(off),ypos), linestyle='none', marker=mk,
                                markersize=ms+4, color='C'+str(s), clip_on=False)
            ax.axhline(0,linestyle='dashed',color='black',lw=1)
            if q == 0:
                lo, hi = mass_bins[m].split('-')
                ax.set_title(r'$\log_{10}M_{\rm h}\in[%s, %s]$'%(lo,hi))
            if q == len(quantities)-1:
                ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        axes[q,0].set_ylabel(ylabel)
        axes[q,0].set_ylim(*ylims[q])
    ax.set_xlim(K_EDGES[0],K_EDGES[-1])
    ax.set_xticks([0.5,1.0,1.5])
    leg = axes[1,0].legend(loc='best', ncol=2, columnspacing=0.6, handletextpad=0.2,
                           borderpad=0.3, borderaxespad=0.3,
                           title=r'th $-$ obs, $' + redshifts[idx_snapshot] + '$')
    leg.set_in_layout(False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.04,hspace=0.04)
    if outdir is None:
        plt.show()
    else:
        os.makedirs(outdir,exist_ok=True)
        fname = os.path.join(outdir,'P0pP2_P4_over_P0_shapes_th-obs'
                             + snapshot[idx_snapshot].rstrip('_') + '.png')
        fig.savefig(fname)
        print('saved ' + fname)
    plt.close(fig)

## run
for idx_snapshot in range(len(snapshot)):
    plotter_shapes(idx_snapshot=idx_snapshot,
                   outdir=outdir)
