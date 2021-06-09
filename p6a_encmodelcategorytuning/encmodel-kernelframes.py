#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28, 2018

Loaded data in dict like this:
data_dict = {"R2": bR2, "weight": bweight, "gweight": bgweight, "mouse_id": mouse_id, "settings": settings }
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
import CAplot, CAgeneral, CAencodingmodel, CArec, CAstats
import CAanalysissupport as CAsupp
import matdata
import warnings

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings, which are due to columns/rows full with NaN's in numpy matrices
warnings.filterwarnings('ignore')

figpath = "../../figureout/"
BEHAVIOR_CATEGORY_CHRONIC_INFOINT = "../../data/p3a_chronicimagingbehavior/performance_category_chronic_infointegr.npy"

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
    elif component.lower() == "choice both":
        if side.lower() == "left":
            predictors = [["Left choice"],["Left first lick sequence"]]
        elif side.lower() == "right":
            predictors = [["Right choice"],["Right first lick sequence"]]
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
    elif component.lower() == "run onset":
        if side.lower() == "left":
            predictors = [["Trial running onset"],]
        elif side.lower() == "right":
            predictors = [["Trial running onset"],]
    elif component.lower() == "speed":
        if side.lower() == "left":
            predictors = [["Speed"],]
        elif side.lower() == "right":
            predictors = [["Speed"],]
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
areas = ['V1','AL','POR']
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
frames = model_data["pfweight"]
mouse_id = model_data["mouse_id"]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show weights
w_pref_per_neuron_ST = [collections.OrderedDict(),collections.OrderedDict()]
w_nonp_per_neuron_ST =  [collections.OrderedDict(),collections.OrderedDict()]
w_pref_per_neuron_LG =  [collections.OrderedDict(),collections.OrderedDict()]
w_nonp_per_neuron_LG =  [collections.OrderedDict(),collections.OrderedDict()]
f_pref_per_neuron_ST = [collections.OrderedDict(),collections.OrderedDict()]
f_nonp_per_neuron_ST =  [collections.OrderedDict(),collections.OrderedDict()]
f_pref_per_neuron_LG =  [collections.OrderedDict(),collections.OrderedDict()]
f_nonp_per_neuron_LG =  [collections.OrderedDict(),collections.OrderedDict()]
bhv_perf = collections.OrderedDict()

n_neurons_area = {}
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

        # signtuned_ST = signtuned_LG

        print("  - found {} significantly tuned stable cells".format(np.nansum(signtuned_ST)))
        n_neurons_area[area] = np.nansum(signtuned_ST)

        # Prepare matrix for mean and per-frame weights
        left_preds = get_selective_predictors("F02","Left",component)
        n_frames,n_neurons,n_timepoints = frames[area][left_preds[0][0]]["mean"].shape
        compW_left_LG = np.zeros((n_neurons,n_timepoints))
        compW_left_ST = np.zeros((n_neurons,n_timepoints))
        compF_left_LG = np.zeros((n_frames,n_neurons,n_timepoints))
        compF_left_ST = np.zeros((n_frames,n_neurons,n_timepoints))

        right_preds = get_selective_predictors("F02","Right",component)
        n_frames,n_neurons,n_timepoints = frames[area][right_preds[0][0]]["mean"].shape
        compW_right_LG = np.zeros((n_neurons,n_timepoints))
        compW_right_ST = np.zeros((n_neurons,n_timepoints))
        compF_right_LG = np.zeros((n_frames,n_neurons,n_timepoints))
        compF_right_ST = np.zeros((n_frames,n_neurons,n_timepoints))

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
                    compF_left_LG[:,n,:] += frames[area][pred]["mean"][:,n,:]
                    compF_left_ST[:,n,:] += frames[area][pred]["mean"][:,n,:]
            compW_left_LG[n,:] = compW_left_LG[n,:] / len(left_preds)
            compW_left_ST[n,:] = compW_left_ST[n,:] / len(left_preds)
            compF_left_LG[:,n,:] = compF_left_LG[:,n,:] / len(left_preds)
            compF_left_ST[:,n,:] = compF_left_ST[:,n,:] / len(left_preds)

            # Get right predictor weights
            for right_pred in right_preds:
                for pred in right_pred:
                    compW_right_LG[n,:] += weight[area][pred]["mean"][n,:]
                    compW_right_ST[n,:] += weight[area][pred]["mean"][n,:]
                    compF_right_LG[:,n,:] += frames[area][pred]["mean"][:,n,:]
                    compF_right_ST[:,n,:] += frames[area][pred]["mean"][:,n,:]
            compW_right_LG[n,:] = compW_right_LG[n,:] / len(left_preds)
            compW_right_ST[n,:] = compW_right_ST[n,:] / len(left_preds)
            compF_right_LG[:,n,:] = compF_right_LG[:,n,:] / len(left_preds)
            compF_right_ST[:,n,:] = compF_right_ST[:,n,:] / len(left_preds)

        # Eliminate 2nd baseline timepoint
        compW_left_LG = compW_left_LG[:,[0,2]]
        compW_left_ST = compW_left_ST[:,[0,2]]
        compW_right_LG = compW_right_LG[:,[0,2]]
        compW_right_ST = compW_right_ST[:,[0,2]]
        compF_left_LG = compF_left_LG[:,:,[0,2]]
        compF_left_ST = compF_left_ST[:,:,[0,2]]
        compF_right_LG = compF_right_LG[:,:,[0,2]]
        compF_right_ST = compF_right_ST[:,:,[0,2]]
        n_timepoints = 2

        # Re-sort mean weights into preferred and non-preferred side
        compW_pref_LG = np.zeros_like(compW_left_LG)
        compW_nonp_LG = np.zeros_like(compW_left_LG)
        compF_pref_LG = np.zeros_like(compF_left_LG)
        compF_nonp_LG = np.zeros_like(compF_left_LG)

        compW_pref_ST = np.zeros_like(compW_left_ST)
        compW_nonp_ST = np.zeros_like(compW_left_ST)
        compF_pref_ST = np.zeros_like(compF_left_ST)
        compF_nonp_ST = np.zeros_like(compF_left_ST)

        for tp in range(2):
            for n in range(n_neurons):
                if compW_left_LG[n,tp] > compW_right_LG[n,tp]:
                    compW_pref_LG[n,tp] = compW_left_LG[n,tp]
                    compF_pref_LG[:,n,tp] = compF_left_LG[:,n,tp]
                    compW_nonp_LG[n,tp] = compW_right_LG[n,tp]
                    compF_nonp_LG[:,n,tp] = compF_right_LG[:,n,tp]
                else:
                    compW_pref_LG[n,tp] = compW_right_LG[n,tp]
                    compF_pref_LG[:,n,tp] = compF_right_LG[:,n,tp]
                    compW_nonp_LG[n,tp] = compW_left_LG[n,tp]
                    compF_nonp_LG[:,n,tp] = compF_left_LG[:,n,tp]
                if compW_left_ST[n,tp] > compW_right_ST[n,tp]:
                    compW_pref_ST[n,tp] = compW_left_ST[n,tp]
                    compF_pref_ST[:,n,tp] = compF_left_ST[:,n,tp]
                    compW_nonp_ST[n,tp] = compW_right_ST[n,tp]
                    compF_nonp_ST[:,n,tp] = compF_right_ST[:,n,tp]
                else:
                    compW_pref_ST[n,tp] = compW_right_ST[n,tp]
                    compF_pref_ST[:,n,tp] = compF_right_ST[:,n,tp]
                    compW_nonp_ST[n,tp] = compW_left_ST[n,tp]
                    compF_nonp_ST[:,n,tp] = compF_left_ST[:,n,tp]

        # Select significant neurons
        for tp in range(2):
            compW_pref_LG[ signtuned_LG[:,tp]==False, tp ] = np.NaN
            compW_pref_ST[ signtuned_ST[:,tp]==False, tp ] = np.NaN
            compW_nonp_LG[ signtuned_LG[:,tp]==False, tp ] = np.NaN
            compW_nonp_ST[ signtuned_ST[:,tp]==False, tp ] = np.NaN
            compF_pref_LG[ :, signtuned_LG[:,tp]==False, tp ] = np.NaN
            compF_pref_ST[ :, signtuned_ST[:,tp]==False, tp ] = np.NaN
            compF_nonp_LG[ :, signtuned_LG[:,tp]==False, tp ] = np.NaN
            compF_nonp_ST[ :, signtuned_ST[:,tp]==False, tp ] = np.NaN

        # Store weights and per-frame weights in data containers
        w_pref_per_neuron_LG[c_nr][area] = compW_pref_LG
        w_pref_per_neuron_ST[c_nr][area] = compW_pref_ST
        w_nonp_per_neuron_LG[c_nr][area] = compW_nonp_LG
        w_nonp_per_neuron_ST[c_nr][area] = compW_nonp_ST

        f_pref_per_neuron_LG[c_nr][area] = compF_pref_LG
        f_pref_per_neuron_ST[c_nr][area] = compF_pref_ST
        f_nonp_per_neuron_LG[c_nr][area] = compF_nonp_LG
        f_nonp_per_neuron_ST[c_nr][area] = compF_nonp_ST


for a_nr,area in enumerate(areas):

    if n_neurons_area[area] == 0:
        continue

    fig,ax = CAplot.init_figure_axes(fig_size=(25,7))
    if len(components) == 1:
        y_off = np.nanmax( np.nanmean(f_pref_per_neuron_ST[0][area],axis=1) )
    else:
        y_off = np.max([ np.nanmax( np.nanmean(f_pref_per_neuron_ST[0][area],axis=1) ), np.nanmax( np.nanmean(f_pref_per_neuron_ST[1][area],axis=1) ) ])

    xvalues = np.arange(n_frames)
    for comp in range(len(components)):
        ax = CAplot.plt.subplot2grid( (1,5), (0,comp) )

        ydata = mean_stderr( f_pref_per_neuron_ST[comp][area][:,:,0], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#888888", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#888888" )

        ydata = mean_stderr( f_nonp_per_neuron_ST[comp][area][:,:,0], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#888888", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#888888" )

        ydata = mean_stderr( f_pref_per_neuron_ST[comp][area][:,:,1], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#000000", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#000000" )

        ydata = mean_stderr( f_nonp_per_neuron_ST[comp][area][:,:,1], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#000000", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#000000" )

        CAplot.finish_panel( ax, title=components[comp], ylabel="Spike rate", xlabel="Frames", legend="off", y_minmax=[0,y_off*1.1], y_step=None, y_margin=0, y_axis_margin=0, x_minmax=[0,n_frames-1], x_step=None, x_margin=0.75, x_axis_margin=0.55, despine=True)

    if len(components) == 2:

        # Y0=category component, Y1=stimulus component
        Y0 = (w_pref_per_neuron_ST[0][area] - w_nonp_per_neuron_ST[0][area]) / (w_pref_per_neuron_ST[0][area] + w_nonp_per_neuron_ST[0][area])
        Y1 = (w_pref_per_neuron_ST[1][area] - w_nonp_per_neuron_ST[1][area]) / (w_pref_per_neuron_ST[1][area] + w_nonp_per_neuron_ST[1][area])
        Yd = Y0-Y1

        # Just to verify we are looking at the right kernels and results
        meansem1,meansem2 = test_before_after( Yd, area, min_n_samples=5, paired=False, bonferroni=1, name1="bs", name2="lrn", suppr_out=True)

        xvalues = np.arange(n_frames)

        F0 = (f_pref_per_neuron_ST[0][area] - f_nonp_per_neuron_ST[0][area]) / (f_pref_per_neuron_ST[0][area] + f_nonp_per_neuron_ST[0][area])
        F1 = (f_pref_per_neuron_ST[1][area] - f_nonp_per_neuron_ST[1][area]) / (f_pref_per_neuron_ST[1][area] + f_nonp_per_neuron_ST[1][area])
        Fd = F0-F1

        ax = CAplot.plt.subplot2grid( (1,5), (0,2) )
        ydata = mean_stderr( F0[:,:,0], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#888888", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#888888" )

        ydata = mean_stderr( F0[:,:,1], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#000000", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#000000" )

        CAplot.finish_panel( ax, title="Category", ylabel="CTI", xlabel="Frames", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,n_frames-1], x_step=None, x_margin=0.75, x_axis_margin=0.55, despine=True)

        ax = CAplot.plt.subplot2grid( (1,5), (0,3) )
        ydata = mean_stderr( F1[:,:,0], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#888888", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#888888" )

        ydata = mean_stderr( F1[:,:,1], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#000000", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#000000" )

        CAplot.finish_panel( ax, title="Feature", ylabel="CTI", xlabel="Frames", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,n_frames-1], x_step=None, x_margin=0.75, x_axis_margin=0.55, despine=True)

        ax = CAplot.plt.subplot2grid( (1,5), (0,4) )
        ydata = mean_stderr( Fd[:,:,0], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#888888", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#888888" )

        ydata = mean_stderr( Fd[:,:,1], axis=1 )
        CAplot.plt.fill_between( xvalues, ydata.mean-ydata.stderr, ydata.mean+ydata.stderr, facecolor="#000000", alpha=0.3, linewidth=0 )
        CAplot.plt.plot( xvalues, ydata.mean, color="#000000" )

        CAplot.finish_panel( ax, title="Delta", ylabel="delta CTI", xlabel="Frames", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,n_frames-1], x_step=None, x_margin=0.75, x_axis_margin=0.55, despine=True)


    CAplot.finish_figure( filename="6ED8bc-KernelFrames-Area-{}".format(area), wspace=0.8, hspace=0.8 )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
