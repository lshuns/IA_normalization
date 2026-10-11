# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2026-09-29 23:50:00
# @Last Modified by:   lshuns
# @Last Modified time: 2026-10-10 22:35:02

### Test the symmetry relations among the P_hE multipoles, for the th (3D) and obs (2D) normalizations.
###    (P_hE^0 + P_hE^2) / P_hE^0   and   P_hE^4 / P_hE^0,  both expected to vanish for A = th
###    one figure, rows = ratio, columns = mass bin, all redshifts per panel, single shape type
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

## the two cases compared: (output directory, legend label, colour map)
##    th  = 3D normalization, projected tensor divided by the 3D trace, linear in the 3D shape
##    obs = 2D normalization, divided by the projected trace, as for observed ellipticities
cases = [("NormEach3D", r'${\rm th}$',  plt.cm.Blues),
         ("NormEach",   r'${\rm obs}$', plt.cm.Oranges)]

## redshifts: shade within each colour map (darker = higher z) and marker symbol
z_shades = [0.95,0.7,0.45]
z_markers = ['o','s','^']

## coarse k binning: 6 linearly spaced bins in [0, 2] h/Mpc, few enough to invert
##    the sample covariance from 20 realizations (Hartlap factor below)
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


def plotter_ratio(idx_shape=0,outdir=None):
    """One figure for a single shape type: one row per ratio, one column per
    mass bin, each panel showing th and obs at all redshifts."""
    # gather first: res[case][quantity] has shape (n_snapshot, n_mbin, nreal, nk)
    res = np.full((len(cases),2,len(snapshot),len(mbin),nreal,nk),np.nan)
    for c, (nml,label,cmap) in enumerate(cases):
        for j in range(len(snapshot)):
            for m in range(len(mbin)):
                kc, res[c,0,j,m], res[c,1,j,m] = get_ratios(nml,m,idx_shape,j)

    # (y label, short name), one per row
    quantities = [(r'$(P_{hE}^{\ell=0}+P_{hE}^{\ell=2})\,/\,P_{hE}^{\ell=0}$', 'P0pP2_over_P0'),
                  (r'$P_{hE}^{\ell=4}\,/\,P_{hE}^{\ell=0}$',                  'P4_over_P0')]

    nseries = len(cases)*len(snapshot)
    fig, axes = plt.subplots(len(quantities),len(mbin),figsize=(TEXT_WIDTH,0.6*TEXT_WIDTH),sharex=True,sharey='row')
    for q, (ylabel,_) in enumerate(quantities):
        for m in range(len(mbin)):
            ax = axes[q,m]
            for j in range(len(snapshot)):
                for c, (nml,label,cmap) in enumerate(cases):
                    vals = res[c,q,j,m]
                    # mean over realizations, error on the mean from the scatter across the 20 sims
                    ## small offset so the six series do not overlap within a k bin
                    i_series = c*len(snapshot) + j
                    ax.errorbar(kc + 0.035*(i_series - (nseries-1)/2.), np.mean(vals,axis=0),
                                yerr = np.std(vals,axis=0,ddof=1)/np.sqrt(nreal),
                                fmt=z_markers[j], markersize=ms, markeredgecolor=ec,
                                label=label + ', $' + redshifts[j] + '$',
                                color=cmap(z_shades[j]), elinewidth=ew, capsize=cs)
            ax.axhline(0,linestyle='dashed',color='black',lw=1)
            if q == 0:
                lo, hi = mass_bins[m].split('-')
                ax.set_title(r'$\log_{10}M_{\rm h}\in[%s, %s]$'%(lo,hi))
            if q == len(quantities)-1:
                ax.set_xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
        axes[q,0].set_ylabel(ylabel)
    ax.set_xlim(K_EDGES[0],K_EDGES[-1])
    ax.set_xticks([0.5,1.0,1.5]) 
    axes[1,0].text(0.04,0.96,shape[idx_shape],transform=axes[1,0].transAxes,ha='left',va='top')
    handles, labels = axes[1,0].get_legend_handles_labels()
    order = [j*len(cases) + c for c in range(len(cases)) for j in range(len(snapshot))]
    leg = axes[1,0].legend([handles[i] for i in order], [labels[i] for i in order],
                           ncol=len(cases), columnspacing=0.4, handletextpad=0.1,
                           handlelength=0.8, borderpad=0.3, borderaxespad=0.3, # compact, to fit the panel
                           loc='upper left', bbox_to_anchor=(0.0,0.88)) # below the label
    leg.set_in_layout(False) # otherwise tight_layout widens the gaps to fit it
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.04,hspace=0.04) # shared axes, so the panels can nearly touch
    if outdir is None:
        plt.show()
    else:
        os.makedirs(outdir,exist_ok=True)
        fname = os.path.join(outdir,'P0pP2_P4_over_P0_' + shape[idx_shape] + '.png')
        fig.savefig(fname)
        print('saved ' + fname)
    plt.close(fig)


## run
for idx_shape in range(len(shape)):
    plotter_ratio(idx_shape=idx_shape,
                outdir=outdir)