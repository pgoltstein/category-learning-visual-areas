#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28, 2018

Loaded data is in dict like this:
data_dict = {"R2": bR2, "weight": bweight, "gweight": bgweight, "mouse_id": mouse_id, "settings": settings }
bR2[area][group] = {'mean': np.array(bd.mean), 'ci95low': np.array(bd.ci95[0]), 'ci95up': np.array(bd.ci95[1]), 'ci99low': np.array(bd.ci99[0]), 'ci99up': np.array(bd.ci99[1])}

@author: pgoltstein
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get all imports
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import collections
import sys
sys.path.append('../xx_analysissupport')
BEHAVIOR_CATEGORY_CHRONIC_INFOINT = "../../data/p3a_chronicimagingbehavior/performance_category_chronic_infointegr.npy"

import CAplot, CAgeneral, CAencodingmodel, CArec, CAstats
import CAanalysissupport as CAsupp
import matdata
import warnings

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings, which are due to columns/rows full with NaN's in numpy matrices
warnings.filterwarnings('ignore')

figpath = "../../figureout/"

suppress_connecting_individual_datapoints_for_speed = True

CAplot.font_size["title"] = 6
CAplot.font_size["label"] = 6
CAplot.font_size["tick"] = 6
CAplot.font_size["text"] = 6
CAplot.font_size["legend"] = 6

# Create custom colormap
newcolors = ([0,0,1.0,1.0], [0,0,0.7,1.0], [0,0,0,1.0], [0.7,0,0,1.0], [1.0,0,0,1.0])
customcmap = LinearSegmentedColormap.from_list('customcmap',newcolors,N=100)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Local functions

def get_selective_predictors(mouse,side,component):
    predictors = []
    if component.lower() == "category":
        if side.lower() == "left":
            predictors = [["Left category"],]
        elif side.lower() == "right":
            predictors = [["Right category"],]
            cat_predictor = "Right category"
    elif component.lower() == "stimulus-only":
        data_dict = np.load( BEHAVIOR_CATEGORY_CHRONIC_INFOINT, allow_pickle=True ).item()
        if side.lower() == "left":
            cat_nr = 1.0
        elif side.lower() == "right":
            cat_nr = 0.0
        category_mat = (data_dict["category_id"][mouse]==cat_nr)*1.0
        spf_ori_ids = np.argwhere(category_mat==1.0)
        for spf,ori in spf_ori_ids:
            predictors += [["Orientation {}".format(ori), "Spatial freq {}".format(spf)],]
    elif component.lower() == "choice all":
        if side.lower() == "left":
            predictors = [["Left choice"],["Left first lick sequence"],["Left lick"]]
        elif side.lower() == "right":
            predictors = [["Right choice"],["Right first lick sequence"],["Right lick"]]
    elif component.lower() == "choice":
        if side.lower() == "left":
            predictors = [["Left choice"],]
        elif side.lower() == "right":
            predictors = [["Right choice"],]
    elif component.lower() == "first lick sequence":
        if side.lower() == "left":
            predictors = [["Left first lick sequence"],]
        elif side.lower() == "right":
            predictors = [["Right first lick sequence"],]
    elif component.lower() == "lick":
        if side.lower() == "left":
            predictors = [["Left lick"],]
        elif side.lower() == "right":
            predictors = [["Right lick"],]
    elif component.lower() == "reward":
        if side.lower() == "left":
            predictors = [["Reward"],]
        elif side.lower() == "right":
            predictors = [["No reward"],]
    return predictors


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


def test_before_after( data_bf, area, min_n_samples=5, paired=False, bonferroni=1, name1="bs", name2="lrn", suppr_out=False):
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
        if paired:
            CAstats.report_wmpsr_test( data1, data2, n_indents=6, bonferroni=bonferroni )
        else:
            CAstats.report_mannwhitneyu_test( data1, data2, n_indents=6, bonferroni=bonferroni )
    return meansem1,meansem2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic settings
areas = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
mice = ['21a','21b','F02','F03','F04','K01','K02','K03','K06','K07']
n_areas = len(areas)
n_mice = len(mice)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load model data
model_data = CAsupp.model_load_data("cat-trained", R2="m")
model_data_sh = CAsupp.model_load_data("shuffled-trials", R2="m")
CAplot.print_dict(model_data["settings"])

select_component = "Stimulus"
components = ["Category","Stimulus-only"]
min_n_samples = 5

R2 = model_data["R2"]
R2_sh = model_data_sh["R2"]
weight = model_data["weight"]
mouse_id = model_data["mouse_id"]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show weights
w_pref_per_neuron_ST = [collections.OrderedDict(),collections.OrderedDict()]
w_nonp_per_neuron_ST =  [collections.OrderedDict(),collections.OrderedDict()]
w_pref_per_neuron_LG =  [collections.OrderedDict(),collections.OrderedDict()]
w_nonp_per_neuron_LG =  [collections.OrderedDict(),collections.OrderedDict()]
bhv_perf = collections.OrderedDict()

for a_nr,area in enumerate(areas):
    print("Processing area {}".format(area))

    # Get behavioral performance for this area
    bhv_perf[area] = CAgeneral.beh_per_mouse( model_data["bhv-perf"][area], mouse_no_dict=CArec.mouse_no, timepoint=2 )

    # loop to-compare components
    for c_nr,component in enumerate(components):

        # Get significantly predictive neurons
        signtuned_lost = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="lost", n_timepoints=2, component=select_component)
        signtuned_gained = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="gained", n_timepoints=2, component=select_component)
        signtuned_LG = np.stack( [signtuned_lost[:,0], signtuned_gained[:,1]], axis=1 )
        signtuned_ST = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="stable", n_timepoints=2, component=select_component)

        # Prepare matrix for mean and per-frame weights
        left_preds = get_selective_predictors("F02","Left",component)
        n_neurons,n_timepoints = weight[area][left_preds[0][0]]["mean"].shape
        compW_left_LG = np.zeros((n_neurons,n_timepoints))
        compW_left_ST = np.zeros((n_neurons,n_timepoints))

        right_preds = get_selective_predictors("F02","Right",component)
        n_neurons,n_timepoints = weight[area][right_preds[0][0]]["mean"].shape
        compW_right_LG = np.zeros((n_neurons,n_timepoints))
        compW_right_ST = np.zeros((n_neurons,n_timepoints))

        # Get mean weights
        prev_mouse_name = ""
        for n in range(n_neurons):
            mouse_name = CArec.mouse_name[mouse_id[area][n]]

            # Reload predictor names if necessary
            if mouse_name != prev_mouse_name:
                prev_mouse_name = str(mouse_name)
                left_preds = get_selective_predictors(mouse_name,"Left",component)
                right_preds = get_selective_predictors(mouse_name,"Right",component)

            # Get left predictor weights
            for left_pred in left_preds:
                for pred in left_pred:
                    compW_left_LG[n,:] += weight[area][pred]["mean"][n,:]
                    compW_left_ST[n,:] += weight[area][pred]["mean"][n,:]
            compW_left_LG[n,:] = compW_left_LG[n,:] / len(left_preds)
            compW_left_ST[n,:] = compW_left_ST[n,:] / len(left_preds)

            # Get right predictor weights
            for right_pred in right_preds:
                for pred in right_pred:
                    compW_right_LG[n,:] += weight[area][pred]["mean"][n,:]
                    compW_right_ST[n,:] += weight[area][pred]["mean"][n,:]
            compW_right_LG[n,:] = compW_right_LG[n,:] / len(left_preds)
            compW_right_ST[n,:] = compW_right_ST[n,:] / len(left_preds)

        # Eliminate 2nd baseline timepoint
        compW_left_LG = compW_left_LG[:,[0,2]]
        compW_left_ST = compW_left_ST[:,[0,2]]
        compW_right_LG = compW_right_LG[:,[0,2]]
        compW_right_ST = compW_right_ST[:,[0,2]]
        n_timepoints = 2

        # Re-sort mean weights into preferred and non-preferred side
        compW_lr_LG = np.stack((compW_left_LG,compW_right_LG),axis=2)
        compW_lr_ST = np.stack((compW_left_ST,compW_right_ST),axis=2)
        compW_pref_LG = np.max( compW_lr_LG, axis=2 )
        compW_pref_ST = np.max( compW_lr_ST, axis=2 )
        compW_nonp_LG = np.min( compW_lr_LG, axis=2 )
        compW_nonp_ST = np.min( compW_lr_ST, axis=2 )

        # Select significant neurons
        for tp in range(2):
            compW_pref_LG[ signtuned_LG[:,tp]==False, tp ] = np.NaN
            compW_pref_ST[ signtuned_ST[:,tp]==False, tp ] = np.NaN
            compW_nonp_LG[ signtuned_LG[:,tp]==False, tp ] = np.NaN
            compW_nonp_ST[ signtuned_ST[:,tp]==False, tp ] = np.NaN

        # Store weights and per-frame weights in data containers
        w_pref_per_neuron_LG[c_nr][area] = compW_pref_LG
        w_pref_per_neuron_ST[c_nr][area] = compW_pref_ST
        w_nonp_per_neuron_LG[c_nr][area] = compW_nonp_LG
        w_nonp_per_neuron_ST[c_nr][area] = compW_nonp_ST


######################################################################

print("\nStable cells:\n")
allY0 = []
allY1 = []
allYd = []
catsel_diff = np.zeros((n_areas,2), dtype=float, order='C')
stimsel_diff = np.zeros((n_areas,2), dtype=float, order='C')

for a_nr,area in enumerate(areas):
    print("{}".format(area))
    # Y0=category component, Y1=stimulus component
    Y0 = (w_pref_per_neuron_ST[0][area] - w_nonp_per_neuron_ST[0][area]) / (w_pref_per_neuron_ST[0][area] + w_nonp_per_neuron_ST[0][area])
    Y1 = (w_pref_per_neuron_ST[1][area] - w_nonp_per_neuron_ST[1][area]) / (w_pref_per_neuron_ST[1][area] + w_nonp_per_neuron_ST[1][area])
    Yd = Y0-Y1
    b0, a0 = test_before_after( Y0, area+", stable cells, semantic CTI (bs vs lrn)", min_n_samples=min_n_samples, paired=True, bonferroni=8, suppr_out=False )
    b1, a1 = test_before_after( Y1, area+", stable cells, feature CTI (bs vs lrn)", min_n_samples=min_n_samples, paired=True, bonferroni=8, suppr_out=False )
    catsel_diff[a_nr,0] = a0.mean-b0.mean
    stimsel_diff[a_nr,0] = a1.mean-b1.mean
    allY0.append(np.array(Y0))
    allY1.append(np.array(Y1))

allY0 = np.concatenate(allY0, axis=0, out=None)
allY1 = np.concatenate(allY1, axis=0, out=None)


# ######################################################################

print("\nPlotting semantic-cti, merged areas, all individual stable neurons")
fig = CAplot.init_figure(fig_size=(3,4.5))
ax = CAplot.plt.subplot2grid( (1,1), (0,0) )

Y_for_df = np.concatenate([ allY0[:,0], allY0[:,1] ], axis=0)
X_for_df = np.concatenate([ np.zeros_like(allY0[:,0]), np.zeros_like(allY0[:,0])+1 ], axis=0)
XY_for_df = np.stack([X_for_df.ravel(),Y_for_df.ravel()], axis=1)
dfW_per_neuron = pd.DataFrame(XY_for_df, columns=["X","Y"])
b1, b2 = test_before_after( allY0, "All stable cells, semantic CTI (bs vs lrn)", min_n_samples=min_n_samples, paired=True, bonferroni=1, name1="bs", name2="lrn" )

CAplot.sns.stripplot(data=dfW_per_neuron, x="X", y="Y", ax=ax, size=1, linewidth=1, edgecolor="None", dodge=True, palette=[ (.5,.5,.5,1), (.5,.5,.5,1)])
if not suppress_connecting_individual_datapoints_for_speed:
    print("!! Connecting individual data points, this takes a long time .. set suppress_connecting_individual_datapoints_for_speed to False in order to skip this process")
    c1,c2 = ax.collections[0],ax.collections[1]
    for (x1,y1),(x2,y2) in zip(c1.get_offsets(),c2.get_offsets()):
        CAplot.plt.plot( [x1,x2], [y1,y2], color=(.9,.9,.9,1), marker=None, linewidth=0.5 )

CAplot.plt.plot( [0,0], [b1.mean-b1.stderr, b1.mean+b1.stderr], linewidth=1, marker=None, color="#000000")
CAplot.plt.plot( 0, b1.mean, color="#000000", markerfacecolor="#000000", marker="^", markeredgewidth=1, markeredgecolor="#000000", markersize=5)
CAplot.plt.plot( [1,1], [b2.mean-b2.stderr, b2.mean+b2.stderr], linewidth=1, marker=None, color="#000000")
CAplot.plt.plot( 1, b2.mean, color="#000000", markerfacecolor="#000000", marker="s", markeredgewidth=1, markeredgecolor="#000000", markersize=4)
CAplot.plt.plot( [0,1], [b1.mean, b2.mean], linewidth=1, marker=None, color="#000000")

CAplot.finish_panel( ax=ax, title=None, xlabel="Area", ylabel="Semantic CTI", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55, x_ticks=np.arange(2), x_ticklabels=["B","L"], x_tick_rotation=0, tick_size=None, label_size=None, title_size=None, legend_size=None, despine=True, legendpos=0)

CAplot.finish_figure( filename="6b-SemanticCTI-AllStableMerged", path=figpath, wspace=0.8, hspace=0.8 )


# ######################################################################

print("\nPlotting feature-cti, merged areas, all individual stable neurons")
fig = CAplot.init_figure(fig_size=(3,4.5))
ax = CAplot.plt.subplot2grid( (1,1), (0,0) )

Y_for_df = np.concatenate([ allY1[:,0], allY1[:,1] ], axis=0)
X_for_df = np.concatenate([ np.zeros_like(allY1[:,0]), np.zeros_like(allY1[:,0])+1 ], axis=0)
XY_for_df = np.stack([X_for_df.ravel(),Y_for_df.ravel()], axis=1)
dfW_per_neuron = pd.DataFrame(XY_for_df, columns=["X","Y"])
b1, b2 = test_before_after( allY1, "All stable cells, feature CTI (bs vs lrn)", min_n_samples=min_n_samples, paired=True, bonferroni=1, name1="bs", name2="lrn" )

CAplot.sns.stripplot(data=dfW_per_neuron, x="X", y="Y", ax=ax, size=1, linewidth=1, edgecolor="None", dodge=True, palette=[ (.5,.5,.5,1), (.5,.5,.5,1)])
if not suppress_connecting_individual_datapoints_for_speed:
    print("!! Connecting individual data points, this takes a long time .. set suppress_connecting_individual_datapoints_for_speed to False in order to skip this process")
    c1,c2 = ax.collections[0],ax.collections[1]
    for (x1,y1),(x2,y2) in zip(c1.get_offsets(),c2.get_offsets()):
        CAplot.plt.plot( [x1,x2], [y1,y2], color=(.9,.9,.9,1), marker=None, linewidth=0.5 )

CAplot.plt.plot( [0,0], [b1.mean-b1.stderr, b1.mean+b1.stderr], linewidth=1, marker=None, color="#000000")
CAplot.plt.plot( 0, b1.mean, color="#000000", markerfacecolor="#000000", marker="^", markeredgewidth=1, markeredgecolor="#000000", markersize=5)
CAplot.plt.plot( [1,1], [b2.mean-b2.stderr, b2.mean+b2.stderr], linewidth=1, marker=None, color="#000000")
CAplot.plt.plot( 1, b2.mean, color="#000000", markerfacecolor="#000000", marker="s", markeredgewidth=1, markeredgecolor="#000000", markersize=4)
CAplot.plt.plot( [0,1], [b1.mean, b2.mean], linewidth=1, marker=None, color="#000000")

CAplot.finish_panel( ax=ax, title=None, xlabel="Area", ylabel="Feature CTI", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55, x_ticks=np.arange(2), x_ticklabels=["B","L"], x_tick_rotation=0, tick_size=None, label_size=None, title_size=None, legend_size=None, despine=True, legendpos=0)

CAplot.finish_figure( filename="6d-FeatureCTI-AllStableMerged", path=figpath, wspace=0.8, hspace=0.8 )


# ######################################################################

print("\nPlotting semantic CTI as scatter plots, merged areas, all individual stable neurons")
fig = CAplot.init_figure(fig_size=(14,8))
areas.remove("P")
ax = CAplot.plt.subplot2grid( (1,2), (0,0) )

Y0 = np.array(allY0)
x_0, y_0 = test_before_after( Y0, area, min_n_samples=min_n_samples, paired=True, bonferroni=1, suppr_out=True )

CAplot.plt.plot( [-1,1], [-1,1], linestyle=":", linewidth=1, marker=None, color=(.5,.5,.5,1))

CAplot.plt.plot( Y0[:,0], Y0[:,1], linestyle="None", marker="o", markersize=2, markerfacecolor=(.5,.5,.5,1), markeredgecolor="None" )

if x_0 is not None and y_0 is not None:

    CAplot.plt.plot( [x_0.mean-x_0.stderr, x_0.mean+x_0.stderr], [y_0.mean, y_0.mean], linewidth=1, marker=None, color=(.0,.0,.0,1))

    CAplot.plt.plot( [x_0.mean, x_0.mean], [y_0.mean-y_0.stderr, y_0.mean+y_0.stderr], linewidth=1, marker=None, color=(.0,.0,.0,1))

    CAplot.plt.plot( x_0.mean, y_0.mean, linestyle="None", marker="s", markersize=3, markerfacecolor=(.0,.0,.0,1), markeredgecolor="None" )

CAplot.finish_panel( ax=ax, title="", xlabel="Semantic CTI (baseline)", ylabel="Semantic CTI (learned)", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=[0.2,1], x_margin=0.02, x_axis_margin=0.01, despine=True)
ax.set_aspect("equal")


print("\nPlotting feature CTI as scatter plots, merged areas, all individual stable neurons")
ax = CAplot.plt.subplot2grid( (1,2), (0,1) )

# Calculate delta-CTI
Y1 = np.array(allY1)

x_1, y_1 = test_before_after( Y1, area, min_n_samples=min_n_samples, paired=True, bonferroni=1, suppr_out=True )

CAplot.plt.plot( [-1,1], [-1,1], linestyle=":", linewidth=1, marker=None, color=(.5,.5,.5,1))

CAplot.plt.plot( Y1[:,0], Y1[:,1], linestyle="None", marker="o", markersize=2, markerfacecolor=(.5,.5,.5,1), markeredgecolor="None" )

if x_1 is not None and y_1 is not None:

    CAplot.plt.plot( [x_1.mean-x_1.stderr, x_1.mean+x_1.stderr], [y_1.mean, y_1.mean], linewidth=1, marker=None, color=(.0,.0,.0,1))

    CAplot.plt.plot( [x_1.mean, x_1.mean], [y_1.mean-y_1.stderr, y_1.mean+y_1.stderr], linewidth=1, marker=None, color=(.0,.0,.0,1))

    CAplot.plt.plot( x_1.mean, y_1.mean, linestyle="None", marker="s", markersize=3, markerfacecolor=(.0,.0,.0,1), markeredgecolor="None" )

CAplot.finish_panel( ax=ax, title="", xlabel=" Feature CTI (baseline)", ylabel="Feature CTI (learned)", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=[0.2,1], x_margin=0.02, x_axis_margin=0.01, despine=True)
ax.set_aspect("equal")

CAplot.finish_figure( filename="6ED9ab-SemanticFeatureCTI-AllStableMerged-Scatterplots", path=figpath, wspace=0.6, hspace=0.6 )


######################################################################

print("\nPlotting colormaps with before vs after difference, for area-map")

fig = CAplot.init_figure_axes(fig_size=(15,12))

areas = ['V1','LM','AL','RL','AM','PM','LI','P','POR']

ax = CAplot.plt.subplot2grid( (1,2), (0,0) )
X = catsel_diff
im = CAplot.plt.imshow( X, aspect="equal", cmap=customcmap, vmin=-0.2, vmax=0.2 )
CAplot.plt.colorbar(im, ticks=[-0.2,0,0.2], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=6)
CAplot.finish_panel( CAplot.plt.gca(), title="Semantic CTI", ylabel="Area", xlabel=None, legend="off", y_minmax=[-0.5,X.shape[0]-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,X.shape[1]-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,X.shape[0],1), y_ticklabels=areas, x_ticks=np.arange(0,X.shape[1],1), x_ticklabels=["Stable","-"], despine=True )

ax = CAplot.plt.subplot2grid( (1,2), (0,1) )
X = stimsel_diff
im = CAplot.plt.imshow( X, aspect="equal", cmap=customcmap, vmin=-0.2, vmax=0.2 )
CAplot.plt.colorbar(im, ticks=[-0.2,0,0.2], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=6)
CAplot.finish_panel( CAplot.plt.gca(), title="Feature CTI", ylabel="Area", xlabel=None, legend="off", y_minmax=[-0.5,X.shape[0]-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,X.shape[1]-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,X.shape[0],1), y_ticklabels=areas, x_ticks=np.arange(0,X.shape[1],1), x_ticklabels=["Stable","-"], despine=True )

CAplot.finish_figure( filename="6ce-SemanticFeatureCTI-DeltaBeforeAfterLearning-Colormaps", path=figpath, wspace=0.8, hspace=0.8 )


######################################################################

print("\nPlotting delta CTI per area, all individual stable neurons")

areas = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
Y_stable = []
X_stable = []
H_stable = []
b1_stable = [[] for _ in range(9)]
b2_stable = [[] for _ in range(9)]
for a_nr,area in enumerate(areas):

    # Y0=category component, Y1=stimulus component
    Y0 = (w_pref_per_neuron_ST[0][area] - w_nonp_per_neuron_ST[0][area]) / (w_pref_per_neuron_ST[0][area] + w_nonp_per_neuron_ST[0][area])
    Y1 = (w_pref_per_neuron_ST[1][area] - w_nonp_per_neuron_ST[1][area]) / (w_pref_per_neuron_ST[1][area] + w_nonp_per_neuron_ST[1][area])
    Yd = Y0-Y1

    Y_stable += [Yd[:,0],]
    X_stable += [np.zeros_like(Yd[:,0]) + a_nr,]
    H_stable += [np.zeros_like(Yd[:,0]),]
    Y_stable += [Yd[:,1],]
    X_stable += [np.zeros_like(Yd[:,1]) + a_nr,]
    H_stable += [np.zeros_like(Yd[:,0]) + 1.0,]

    bd, ad = test_before_after( Yd, area+", stable cells, delta CTI (bs vs lrn)", min_n_samples=min_n_samples, paired=True, bonferroni=8 )
    b1_stable[a_nr] = bd
    b2_stable[a_nr] = ad

fig = CAplot.init_figure(fig_size=(8,4.5))
ax = CAplot.plt.subplot2grid( (1,1), (0,0) )

X = np.concatenate(X_stable, axis=0)
Y = np.concatenate(Y_stable, axis=0)
H = np.concatenate(H_stable, axis=0)
XY = np.stack([X.ravel(),Y.ravel(),H.ravel()], axis=1)
dfW_per_neuron = pd.DataFrame(XY, columns=["X","Y","H"])

CAplot.sns.stripplot(data=dfW_per_neuron, x="X", y="Y", hue="H", ax=ax, size=1, linewidth=1, edgecolor="None", dodge=True, palette=[ (.5,.5,.5,1), (.5,.5,.5,1)])

for a_nr,area in enumerate(areas):
    if not suppress_connecting_individual_datapoints_for_speed:
        print("!! Connecting individual data points, this takes a long time .. set suppress_connecting_individual_datapoints_for_speed to False in order to skip this process")
        c1,c2 = ax.collections[(a_nr*2)],ax.collections[(a_nr*2)+1]
        for (x1,y1),(x2,y2) in zip(c1.get_offsets(),c2.get_offsets()):
            CAplot.plt.plot( [x1,x2], [y1,y2], color=(.9,.9,.9,1), marker=None, linewidth=0.5 )

    if b1_stable[a_nr] is not None:
        CAplot.plt.plot( [a_nr-0.225,a_nr-0.225], [b1_stable[a_nr].mean-b1_stable[a_nr].stderr, b1_stable[a_nr].mean+b1_stable[a_nr].stderr], linewidth=1, marker=None, color="#000000")
        CAplot.plt.plot( a_nr-0.225, b1_stable[a_nr].mean, color="#000000", markerfacecolor="#000000", marker="^", markeredgewidth=1, markeredgecolor="#000000", markersize=5)
    if b2_stable[a_nr] is not None:
        CAplot.plt.plot( [a_nr+0.225,a_nr+0.225], [b2_stable[a_nr].mean-b2_stable[a_nr].stderr, b2_stable[a_nr].mean+b2_stable[a_nr].stderr], linewidth=1, marker=None, color="#000000")
        CAplot.plt.plot( a_nr+0.225, b2_stable[a_nr].mean, color="#000000", markerfacecolor="#000000", marker="s", markeredgewidth=1, markeredgecolor="#000000", markersize=4)
    if b1_stable[a_nr] is not None and b2_stable[a_nr] is not None:
        CAplot.plt.plot( [a_nr-0.225,a_nr+0.225], [b1_stable[a_nr].mean, b2_stable[a_nr].mean], linewidth=1, marker=None, color="#000000")

ax.get_legend().set_visible(False)

CAplot.finish_panel( ax=ax, title=None, xlabel="Area", ylabel="Delta CTI", legend="off", y_minmax=[-1,1], y_step=[0.5,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,8], x_step=None, x_margin=0.75, x_axis_margin=0.55, x_ticks=np.arange(9), x_ticklabels=areas, x_tick_rotation=0, tick_size=None, label_size=None, title_size=None, legend_size=None, despine=True, legendpos=0)

CAplot.finish_figure( filename="6f-DeltaCTI-StableNeurons-PerArea", path=figpath, wspace=0.8, hspace=0.8 )


######################################################################

print("\nPlotting delta CTI per area as scatterplots, all individual stable neurons")

fig = CAplot.init_figure(fig_size=(14,8))
areas.remove("P")
for a_nr,area in enumerate(areas):
    ax = CAplot.plt.subplot2grid( (2,4), (int(a_nr/4),int(np.mod(a_nr,4))) )

    # Calculate delta-CTI
    Y0 = (w_pref_per_neuron_ST[0][area] - w_nonp_per_neuron_ST[0][area]) / (w_pref_per_neuron_ST[0][area] + w_nonp_per_neuron_ST[0][area])
    Y1 = (w_pref_per_neuron_ST[1][area] - w_nonp_per_neuron_ST[1][area]) / (w_pref_per_neuron_ST[1][area] + w_nonp_per_neuron_ST[1][area])
    Yd = Y0-Y1

    x_diff, y_diff = test_before_after( Yd, area, min_n_samples=min_n_samples, paired=True, bonferroni=8, suppr_out=True )

    CAplot.plt.plot( [-1,1], [-1,1], linestyle=":", linewidth=1, marker=None, color=(.5,.5,.5,1))

    CAplot.plt.plot( Yd[:,0], Yd[:,1], linestyle="None", marker="o", markersize=2, markerfacecolor=(.5,.5,.5,1), markeredgecolor="None" )

    if x_diff is not None and y_diff is not None:

        CAplot.plt.plot( [x_diff.mean-x_diff.stderr, x_diff.mean+x_diff.stderr], [y_diff.mean, y_diff.mean], linewidth=1, marker=None, color=(.0,.0,.0,1))

        CAplot.plt.plot( [x_diff.mean, x_diff.mean], [y_diff.mean-y_diff.stderr, y_diff.mean+y_diff.stderr], linewidth=1, marker=None, color=(.0,.0,.0,1))

        CAplot.plt.plot( x_diff.mean, y_diff.mean, linestyle="None", marker="s", markersize=3, markerfacecolor=(.0,.0,.0,1), markeredgecolor="None" )

    CAplot.finish_panel( ax=ax, title=area, xlabel="DeltaCTI (baseline)", ylabel="DeltaCTI (learned)", legend="off", y_minmax=[-1,1], y_step=[0.5,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[-1,1], x_step=[0.5,1], x_margin=0.02, x_axis_margin=0.01, despine=True)
    ax.set_aspect("equal")

CAplot.finish_figure( filename="6ED9c-DeltaCTI-StableNeurons-PerArea-Scatterplots", path=figpath, wspace=0.6, hspace=0.6 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
