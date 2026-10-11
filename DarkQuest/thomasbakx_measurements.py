#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:40:55 2026

@author: thomasbakx
"""

import os
import numpy as np
import matplotlib.pyplot as plt
#import scipy
plt.rcParams["figure.dpi"] = 150      # display DPI
plt.rcParams["savefig.dpi"] = 300     # saved figures

ms=2
ec='black'
ew=1
cs=2

# load data, use up to kmax = 2 roughly, 
mbin = ['mbin0','mbin1','mbin2','mbin3']
mass_bins = ['12-12.5','12.5-13','13-13.5','13.5-14']
shape_nml = ["NormEach","NormMean","NormEach3D","NormMean3D","NormEach3D_3DRandomized","NormEach_3DRandomized"] #only 0 gives spurious effect
shape = ["RIAC","RIVC","RnonI","SIAC","SIVC","SnonI"]
snapshot = ['_S000_','_S004_','_S020_'] 
redshifts = ['z=1.5','z=0.5','z=0.0']
tr = ['hh','hE','hB','EE','EB','BB']
mul = ['0','2','4']
real = [1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23,24] #12-15 not available
nreal = len(real)
nbin = 120 #only the first 150 bins = kmax 2.2 - CIC1024 not reliable otherwise

def get_fname(idx_shape_nml,idx_mbin,idx_shape,idx_snapshot,idx_tr,idx_mul,ireal):
    fname = "./ext/Shape_Normalization/output_" + shape_nml[idx_shape_nml] + "/pk_all/P" + tr[idx_tr] + mul[idx_mul] + '_R' + str(ireal).zfill(3) \
    + snapshot[idx_snapshot] + "CIC1024_kmin0.0_kmax3_nbin200_" + mbin[idx_mbin] + "_shape" + shape[idx_shape]
    if idx_shape_nml > 3:
        fname += "_3DmisalignSigmaMF0p37_seed12345"
    return fname + ".dat"

def get_rebinned_signal(idx_shape_nml,idx_mbin,idx_shape,idx_snapshot,idx_tr,idx_mul,rebin):
    compress = int(nbin/rebin)
    #plot a signal 
    res = np.zeros((rebin,nreal))
    rebin_nmodes = np.zeros(rebin)
    rebin_centers = np.zeros(rebin)
    for re in enumerate(real):
        idx = re[0]
        ar = np.loadtxt(get_fname(idx_shape_nml,idx_mbin,idx_shape,idx_snapshot,idx_tr,idx_mul,re[1]))
        bin_nmodes_all = ar[:,3]
        bin_centers_all = ar[:,0]
        res_all = ar[:,1]
        #if idx_tr == 5 and idx_mul ==0:
            #res_all -= np.mean(res_all[:4]) #rough shot-noise subtraction
        for j in range(rebin): #rebin data
            bin_nmodes = bin_nmodes_all[compress * j:compress * (j+1)]
            rebin_nmodes = np.sum(bin_nmodes)
            res[j,idx] = np.dot(bin_nmodes,res_all[compress * j:compress * (j+1)]) / rebin_nmodes
            rebin_centers[j] = np.dot(bin_nmodes,bin_centers_all[compress * j:compress * (j+1)]) / rebin_nmodes
            
    return rebin_centers,res
        

def _errorbar(ax,k,res,idx_color,label):
    #mean over realizations, error on the mean
    ax.errorbar(k, np.mean(res,axis=1), yerr = np.sqrt(np.diag(np.cov(res)))/np.sqrt(nreal),
                fmt='o', markersize=ms, markeredgecolor=ec, label=label,
                color='C'+str(idx_color), elinewidth=ew, capsize=cs)

def _plot_panels(cases,labels,idx_tr,title_suffix,rebin,ratio_guide=False):
    #cases: list of (idx_shape_nml,idx_mbin,idx_shape,idx_snapshot) tuples, one per curve
    t = tr[idx_tr]
    fig, axes = plt.subplots(1,3,figsize=(13,4))

    skipped = []
    for j in range(len(cases)):
        i_nml, i_mbin, i_shape, i_snap = cases[j]
        #not every shape_nml has every shape on disk (the _3DRandomized runs are RIAC only)
        if not os.path.exists(get_fname(i_nml,i_mbin,i_shape,i_snap,idx_tr,0,real[0])):
            skipped.append(labels[j])
            continue
        k, p0 = get_rebinned_signal(i_nml,i_mbin,i_shape,i_snap,idx_tr,0,rebin)
        _, p2 = get_rebinned_signal(i_nml,i_mbin,i_shape,i_snap,idx_tr,1,rebin)
        _, p4 = get_rebinned_signal(i_nml,i_mbin,i_shape,i_snap,idx_tr,2,rebin)
        kk = k + 0.02*j

        _errorbar(axes[0], kk, p4/p0, j, labels[j])
        _errorbar(axes[1], kk, p4, j, labels[j])
        #summed per realization, so the error bar keeps the l=0/l=2 correlation
        _errorbar(axes[2], kk, p0 + p2, j, labels[j])

    for ax in axes:
        ax.hlines(0,0,2,linestyle='dashed',color='black')
        ax.set_xlabel(r'$k$ [$h/$Mpc]')
        ax.legend(fontsize=6)
    if ratio_guide:
        axes[0].hlines(0.01,0,2,linestyle='dashed',color='black',alpha=0.5)

    axes[0].set_title(t + '4/' + t + '0')
    axes[1].set_title(t + '4')
    axes[2].set_title(t + '0 + ' + t + '2')

    if len(skipped) > 0: #say so rather than silently dropping curves
        print('no data, skipped: ' + ', '.join(skipped) + '  (' + title_suffix + ')')
        title_suffix += '   [no data: ' + ', '.join(skipped) + ']'

    fig.suptitle(title_suffix)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def plotter_varied_shape_nml(idx_mbin,idx_shape,idx_snapshot,idx_tr,rebin):
    cases = [(j,idx_mbin,idx_shape,idx_snapshot) for j in range(len(shape_nml))]
    _plot_panels(cases, shape_nml, idx_tr,
                 mass_bins[idx_mbin] + '__' + shape[idx_shape] + '__' + redshifts[idx_snapshot],
                 rebin)

def plotter_varied_mass(idx_shape,idx_snapshot,idx_tr,rebin):
    cases = [(0,j,idx_shape,idx_snapshot) for j in range(len(mbin))]
    _plot_panels(cases, mass_bins, idx_tr,
                 shape[idx_shape] + '__' + redshifts[idx_snapshot],
                 rebin, ratio_guide=True)

def plotter_varied_redshift(idx_mbin, idx_shape,idx_tr, rebin):
    cases = [(0,idx_mbin,idx_shape,j) for j in range(len(redshifts))]
    _plot_panels(cases, redshifts, idx_tr,
                 mass_bins[idx_mbin] + '__' + shape[idx_shape],
                 rebin, ratio_guide=True)

def plotter_varied_shape(idx_mbin,idx_snapshot,idx_tr,rebin):
    #one figure per shape type, rather than all shapes on shared axes
    for j in range(len(shape)):
        plotter_varied_shape_nml(idx_mbin,j,idx_snapshot,idx_tr,rebin)

plotter_varied_shape(0,0,1,6)
#plotter_varied_shape_nml(0,0,0,1,6)
#plotter_varied_mass(0,0,1,6)
#plotter_varied_redshift(0, 0, 1, 6)
