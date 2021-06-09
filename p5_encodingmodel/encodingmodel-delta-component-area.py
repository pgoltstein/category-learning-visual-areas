#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 8, 2020

Loaded data is in dict like this:
data_dict = {"R2": bR2, "mouse_id": mouse_id, "settings": settings }
bR2[area][group] = {'mean': np.array(bd.mean), 'ci95low': np.array(bd.ci95[0]), 'ci95up': np.array(bd.ci95[1]), 'ci99low': np.array(bd.ci99[0]), 'ci99up': np.array(bd.ci99[1])}

@author: pgoltstein
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get all imports
import numpy as np
import pandas as pd
import collections
import sys
sys.path.append('../xx_analysissupport')
BEHAVIOR_CATEGORY_CHRONIC_INFOINT = "../../data/p3a_chronicimagingbehavior/performance_category_chronic_infointegr.npy"
import CAplot, CAgeneral, CAencodingmodel, CArec, CAstats
import CAanalysissupport as CAsupp
import warnings

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings, which are due to columns/rows full with NaN's in numpy matrices
warnings.filterwarnings('ignore')

CAplot.font_size["title"] = 6
CAplot.font_size["label"] = 6
CAplot.font_size["tick"] = 6
CAplot.font_size["text"] = 6
CAplot.font_size["legend"] = 6

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Local functions

class mean_stderr(object):
        def __init__(self, data, axis=0):
            """ Sets the data sample and calculates mean and stderr """
            n = np.sum( ~np.isnan( data ), axis=axis )
            self._mean = np.nanmean( data, axis=axis )
            self._stderr = np.nanstd( data, axis=axis ) / np.sqrt( n )

        @property
        def mean(self):
            return self._mean

        @property
        def stderr(self):
            return self._stderr

def test_before_after( data_bf, area, min_n_samples=5, paired=False, bonferroni=1, name1="bs", name2="lrn", suppr_out=False, datatype="Data"):
    if type(data_bf) is list:
        data1 = data_bf[0]
        data2 = data_bf[1]
    else:
        data1 = np.array(data_bf[:,0])
        data2 = np.array(data_bf[:,1])
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    meansem1 = mean_stderr(data1, axis=0)
    meansem2 = mean_stderr(data2, axis=0)
    if not suppr_out:
        print("    {}: {}={:5.3f} (+-{:5.3f}), {}={:5.3f} (+-{:5.3f}) (n={},{})".format( area, name1, meansem1.mean, meansem1.stderr, name2, meansem2.mean, meansem2.stderr, data1.size, data2.size))
        try:
            if paired:
                if datatype == "Fraction":
                    p = CAstats.report_chisquare_test( data1, data2, n_indents=6, bonferroni=bonferroni )
                else:
                    p = CAstats.report_wmpsr_test( data1, data2, n_indents=6, bonferroni=bonferroni )
            else:
                if datatype == "Fraction":
                    p = CAstats.report_chisquare_test( data1, data2, n_indents=6, bonferroni=bonferroni )
                else:
                    p = CAstats.report_mannwhitneyu_test( data1, data2, n_indents=6, bonferroni=bonferroni )
        except ValueError as e:
            print(e)
            p = 1.0
    return meansem1, meansem2, 1.0 if p<0.05 else 0.0



def get_R2_matrix( model_area_R2, area, predictor_group_names, timepoint ):
    R2 = []
    for gr in predictor_group_names:
        R2.append(model_area_R2[gr][area][:,timepoint])
    R2 = np.stack(R2,axis=1)
    R2[R2<-1.0] = np.NaN
    R2[R2>1.0] = np.NaN
    return R2

def get_R2_matrix_comp( model_area_R2, predictor_group_name, areas, timepoint ):
    R2 = []
    for area in areas:
        R2a = model_area_R2[predictor_group_name][area][:,timepoint]
        R2a[R2a<-1.0] = np.NaN
        R2a[R2a>1.0] = np.NaN
        R2.append(R2a)
    return R2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic settings
figpath = "../../figureout/"
areas = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
mice = ['21a','21b','F02','F03','F04','K01','K02','K03','K06','K07']
tpnames = ["before learning", "after learning"]
n_areas = len(areas)
n_mice = len(mice)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load model data
predictor_group_names = ["Task","Stimulus","Stimulus-only","Category","Choice","Reward","Run"]
predictor_group_names = predictor_group_names[::-1]
maximum_R2, unique_R2, maximum_sign, unique_sign = CAsupp.model_load_data_unique_max_R2(predictor_group_names)
n_groups = len(predictor_group_names)

# Analysis settings
cmap = CAplot.matplotlib.cm.get_cmap('tab20')
cmap_light = []
cmap_dark = []
for x in range(0,20,2):
    cmap_light.append(cmap(x+1))
    cmap_dark.append(cmap(x))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show unique R2 per area and component
uFRmat_L = np.zeros((n_groups,n_areas,2))
uFRmat_sign_L = np.zeros((n_groups,n_areas))
uFRmat_G = np.zeros((n_groups,n_areas,2))
uFRmat_sign_G = np.zeros((n_groups,n_areas))
uFRmat_S = np.zeros((n_groups,n_areas,2))
uFRmat_sign_S = np.zeros((n_groups,n_areas))
uFRmat = np.zeros((n_groups,n_areas,2))
uFRmat_sign = np.zeros((n_groups,n_areas))

uR2mat_L = np.zeros((n_groups,n_areas,2))
uR2mat_sign_L = np.zeros((n_groups,n_areas))
uR2mat_G = np.zeros((n_groups,n_areas,2))
uR2mat_sign_G = np.zeros((n_groups,n_areas))
uR2mat_S = np.zeros((n_groups,n_areas,2))
uR2mat_sign_S = np.zeros((n_groups,n_areas))
uR2mat = np.zeros((n_groups,n_areas,2))
uR2mat_sign = np.zeros((n_groups,n_areas))

for a_nr,area in enumerate(areas):
    print("Processing area {}".format(area))

    # Fraction of unique R2
    print("  Unique contribution (Fraction), all neurons")
    uR2_before = get_R2_matrix( unique_sign, area, predictor_group_names, timepoint=0 )
    uR2_after = get_R2_matrix( unique_sign, area, predictor_group_names, timepoint=2 )
    for gr_nr,gr in enumerate(predictor_group_names):
        meansem1, meansem2, sign = test_before_after( [uR2_before[:,gr_nr],uR2_after[:,gr_nr]], gr, min_n_samples=5, paired=False, bonferroni=7*9, name1="bs", name2="lrn", suppr_out=False, datatype="Fraction")
        uFRmat[gr_nr,a_nr,0] = meansem1.mean
        uFRmat[gr_nr,a_nr,1] = meansem2.mean
        uFRmat_sign[gr_nr,a_nr] = sign

    # Unique R2
    print("  Unique contribution (R2)")
    uR2_before = get_R2_matrix( unique_R2, area, predictor_group_names, timepoint=0 )
    uR2_after = get_R2_matrix( unique_R2, area, predictor_group_names, timepoint=2 )
    for gr_nr,gr in enumerate(predictor_group_names):
        meansem1, meansem2, sign = test_before_after( [uR2_before[:,gr_nr],uR2_after[:,gr_nr]], gr, min_n_samples=5, paired=False, bonferroni=7*9, name1="bs", name2="lrn", suppr_out=False, datatype="Data")
        uR2mat[gr_nr,a_nr,0] = meansem1.mean
        uR2mat[gr_nr,a_nr,1] = meansem2.mean
        uR2mat_sign[gr_nr,a_nr] = sign

    # Fraction and value of unique R2, lost, gained, stable
    for gr_nr,gr in enumerate(predictor_group_names):
        uFR = unique_sign[gr][area][:,[0,2]]
        uR2 = unique_R2[gr][area][:,[0,2]]

        signtuned_L = np.logical_and( uFR[:,0]==True,  uFR[:,1]==False )
        signtuned_G = np.logical_and( uFR[:,0]==False, uFR[:,1]==True )
        signtuned_S = np.nansum(uFR*1.0,axis=1) == 2

        uFRmat_L[gr_nr,a_nr,:] = np.nanmean(signtuned_L,axis=0)
        uFRmat_G[gr_nr,a_nr,:] = np.nanmean(signtuned_G,axis=0)
        uFRmat_S[gr_nr,a_nr,:] = np.nanmean(signtuned_S,axis=0)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show unique contribution data in color plots, all modulated neurons
fig = CAplot.init_figure(fig_size=(24,10))

# Unique R2 before and after learning
minmax = 0.02
for tp in range(2):
    ax = CAplot.plt.subplot2grid( (2,3), (0,tp) )
    X = uR2mat[:,:,tp]
    im = CAplot.plt.imshow( X, aspect="equal", cmap="Reds", vmin=0, vmax=minmax )
    for gr_nr,gr in enumerate(predictor_group_names):
        for a_nr,area in enumerate(areas):
            if X[gr_nr,a_nr] == 0:
                CAplot.plt.text( a_nr, gr_nr, r'$\cdot$', color="#777777", horizontalalignment='center', verticalalignment='center')
    CAplot.plt.colorbar(im, ticks=[0,minmax], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=8)
    CAplot.finish_panel( CAplot.plt.gca(), title="R2 of unique contribution\n (across all neurons, {})".format(tpnames[tp]), ylabel=None, xlabel="Area", legend="off", y_minmax=[-0.5,n_groups-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,n_areas-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,n_groups,1), y_ticklabels=predictor_group_names, x_ticks=np.arange(0,n_areas,1), x_ticklabels=areas, despine=True )

# Unique R2, difference before and after learning
minmax = 0.02
X = uR2mat[:,:,1]-uR2mat[:,:,0]
ax = CAplot.plt.subplot2grid( (2,3), (0,2) )
im = CAplot.plt.imshow( X, aspect="equal", cmap="seismic", vmin=-1*minmax, vmax=minmax )
for gr_nr,gr in enumerate(predictor_group_names):
    for a_nr,area in enumerate(areas):
        if X[gr_nr,a_nr] == 0:
            CAplot.plt.text( a_nr, gr_nr, r'$\cdot$', color="#777777", horizontalalignment='center', verticalalignment='center')
CAplot.plt.colorbar(im, ticks=[-1*minmax,0,minmax], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=8)
CAplot.finish_panel( ax, title='R2 of unique contribution\n (across all neurons, after minus before learning)', ylabel=None, xlabel="Area", legend="off", y_minmax=[-0.5,n_groups-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,n_areas-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,n_groups,1), y_ticklabels=predictor_group_names, x_ticks=np.arange(0,n_areas,1), x_ticklabels=areas, despine=True )

for gr_nr,gr in enumerate(predictor_group_names):
    for a_nr,area in enumerate(areas):
        if uR2mat_sign[gr_nr,a_nr] == True:
            CAplot.plt.text( a_nr, gr_nr-0.3, "*", color="#ffffff", horizontalalignment='center', verticalalignment='center')

# Fraction of neurons with unique R2, before and after learning
minmax = 0.2
for tp in range(2):
    ax = CAplot.plt.subplot2grid( (2,3), (1,tp) )
    X = uFRmat[:,:,tp]
    im = CAplot.plt.imshow( X, aspect="equal", cmap="Reds", vmin=0, vmax=minmax )
    for gr_nr,gr in enumerate(predictor_group_names):
        for a_nr,area in enumerate(areas):
            if X[gr_nr,a_nr] == 0:
                CAplot.plt.text( a_nr, gr_nr, r'$\cdot$', color="#777777", horizontalalignment='center', verticalalignment='center')
    CAplot.plt.colorbar(im, ticks=[0,minmax], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=8)
    CAplot.finish_panel( CAplot.plt.gca(), title="Fraction of neurons with a significant contribution\n ({})".format(tpnames[tp]), ylabel=None, xlabel="Area", legend="off", y_minmax=[-0.5,n_groups-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,n_areas-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,n_groups,1), y_ticklabels=predictor_group_names, x_ticks=np.arange(0,n_areas,1), x_ticklabels=areas, despine=True )

# Fraction of neurons with unique R2, difference before and after learning
minmax = 0.1
X = uFRmat[:,:,1]-uFRmat[:,:,0]
ax = CAplot.plt.subplot2grid( (2,3), (1,2) )
im = CAplot.plt.imshow( X, aspect="equal", cmap="seismic", vmin=-1*minmax, vmax=minmax )
for gr_nr,gr in enumerate(predictor_group_names):
    for a_nr,area in enumerate(areas):
        if X[gr_nr,a_nr] == 0:
            CAplot.plt.text( a_nr, gr_nr, r'$\cdot$', color="#777777", horizontalalignment='center', verticalalignment='center')
CAplot.plt.colorbar(im, ticks=[-1*minmax,0,minmax], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=8)
CAplot.finish_panel( ax, title='Fraction of neurons with a significant contribution\n (after minus before learning)', ylabel=None, xlabel="Area", legend="off", y_minmax=[-0.5,n_groups-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,n_areas-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,n_groups,1), y_ticklabels=predictor_group_names, x_ticks=np.arange(0,n_areas,1), x_ticklabels=areas, despine=True )

for gr_nr,gr in enumerate(predictor_group_names):
    for a_nr,area in enumerate(areas):
        if uFRmat_sign[gr_nr,a_nr] == True:
            CAplot.plt.text( a_nr, gr_nr-0.3, "*", color="#ffffff", horizontalalignment='center', verticalalignment='center')

CAplot.finish_figure( filename="5g-ED7g-Encodingmodel-UniqueR2-2Dplots", path=figpath, wspace=0.8, hspace=0.8 )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show fraction uniquely modulated in color plots, lost, gained, stable neurons
fig = CAplot.init_figure(fig_size=(24,5))

# Fraction modulated lost neurons
minmax = 0.2
ax = CAplot.plt.subplot2grid( (1,3), (0,0) )
X = uFRmat_L[:,:,0]
im = CAplot.plt.imshow( X, aspect="equal", cmap="Reds", vmin=0, vmax=minmax )
for gr_nr,gr in enumerate(predictor_group_names):
    for a_nr,area in enumerate(areas):
        if X[gr_nr,a_nr] == 0:
            CAplot.plt.text( a_nr, gr_nr, r'$\cdot$', color="#777777", horizontalalignment='center', verticalalignment='center')
CAplot.plt.colorbar(im, ticks=[0,minmax], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=8)
CAplot.finish_panel( CAplot.plt.gca(), title="Fraction of lost neurons\nwith a significant contribution", ylabel=None, xlabel="Area", legend="off", y_minmax=[-0.5,n_groups-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,n_areas-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,n_groups,1), y_ticklabels=predictor_group_names, x_ticks=np.arange(0,n_areas,1), x_ticklabels=areas, despine=True )

# Fraction modulated gained neurons
ax = CAplot.plt.subplot2grid( (1,3), (0,1) )
X = uFRmat_G[:,:,1]
im = CAplot.plt.imshow( X, aspect="equal", cmap="Reds", vmin=0, vmax=minmax )
for gr_nr,gr in enumerate(predictor_group_names):
    for a_nr,area in enumerate(areas):
        if X[gr_nr,a_nr] == 0:
            CAplot.plt.text( a_nr, gr_nr, r'$\cdot$', color="#777777", horizontalalignment='center', verticalalignment='center')
CAplot.plt.colorbar(im, ticks=[0,minmax], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=8)
CAplot.finish_panel( CAplot.plt.gca(), title="Fraction of gained neurons\nwith a significant contribution", ylabel=None, xlabel="Area", legend="off", y_minmax=[-0.5,n_groups-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,n_areas-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,n_groups,1), y_ticklabels=predictor_group_names, x_ticks=np.arange(0,n_areas,1), x_ticklabels=areas, despine=True )


# Fraction modulated gained neurons
ax = CAplot.plt.subplot2grid( (1,3), (0,2) )
X = uFRmat_S[:,:,1]
im = CAplot.plt.imshow( X, aspect="equal", cmap="Reds", vmin=0, vmax=minmax )
for gr_nr,gr in enumerate(predictor_group_names):
    for a_nr,area in enumerate(areas):
        if X[gr_nr,a_nr] == 0:
            CAplot.plt.text( a_nr, gr_nr, r'$\cdot$', color="#777777", horizontalalignment='center', verticalalignment='center')
CAplot.plt.colorbar(im, ticks=[0,minmax], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=8)
CAplot.finish_panel( CAplot.plt.gca(), title="Fraction of stable neurons\nwith a significant contribution", ylabel=None, xlabel="Area", legend="off", y_minmax=[-0.5,n_groups-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,n_areas-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,n_groups,1), y_ticklabels=predictor_group_names, x_ticks=np.arange(0,n_areas,1), x_ticklabels=areas, despine=True )

CAplot.finish_figure( filename="5g-Encodingmodel-UniqueFraction-2Dplots", path=figpath, wspace=0.8, hspace=0.8 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
