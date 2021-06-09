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
import scipy.stats as scistats
import collections
import sys
sys.path.append('../xx_analysissupport')
import CAplot, CAgeneral, CAencodingmodel, CArec, CAstats
import CAanalysissupport as CAsupp
import warnings

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings, which are due to columns/rows full with NaN's in numpy matrices
warnings.filterwarnings('ignore')

figpath = "../../figureout/"
BEHAVIOR_CATEGORY_CHRONIC_INFOINT = "../../data/p3a_chronicimagingbehavior/performance_category_chronic_infointegr.npy"

CAplot.font_size["title"] = 6
CAplot.font_size["label"] = 6
CAplot.font_size["tick"] = 6
CAplot.font_size["text"] = 6
CAplot.font_size["legend"] = 6

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
    elif component.lower() == "stimulus-all":
        data_dict = np.load( BEHAVIOR_CATEGORY_CHRONIC_INFOINT, allow_pickle=True ).item()
        if side.lower() == "left":
            cat_predictor = "Left category"
            cat_nr = 1.0
        elif side.lower() == "right":
            cat_predictor = "Right category"
            cat_nr = 0.0
        category_mat = (data_dict["category_id"][mouse]==cat_nr)*1.0
        spf_ori_ids = np.argwhere(category_mat==1.0)
        for spf,ori in spf_ori_ids:
            predictors += [[cat_predictor, "Orientation {}".format(ori), "Spatial freq {}".format(spf)],]
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
    data1 = np.array(data_bf[:,0])
    data1 = data1[~np.isnan(data1)]
    data2 = np.array(data_bf[:,1])
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

model_data_prot = CAsupp.model_load_data("prot", R2="m")
model_data_prot_sh = CAsupp.model_load_data("prot-shuffled-trials", R2="m")

cat_neurons = np.load("../../data/p5_encodingmodel/grouplist-cat.npy", allow_pickle=True).item()
prot_neurons = np.load("../../data/p5_encodingmodel/grouplist-prot.npy", allow_pickle=True).item()
cat_mouse_nrs = cat_neurons["mouse_nrs"]
prot_mouse_nrs = prot_neurons["mouse_nrs"]
cat_groups = cat_neurons["groups"]
prot_groups = prot_neurons["groups"]

# Find matching index
cat_ix = collections.OrderedDict()
prot_ix = collections.OrderedDict()
for a_nr,area in enumerate(areas):
    cat_ix[area] = []
    prot_ix[area] = []
    cat_ids = (cat_mouse_nrs[area]*10000) + cat_groups[area]
    prot_ids = (prot_mouse_nrs[area]*10000) + prot_groups[area]
    print("Matching cells in area {}".format(area))
    for cat_n,cat_id in enumerate(cat_ids):
        prot_n = np.argwhere(prot_ids==cat_id)
        if len(prot_n) > 0:
            cat_ix[area].append(int(cat_n))
            prot_ix[area].append(int(prot_n))
    print("  - {} cat cells".format(len(cat_groups[area])))
    print("  - {} prot cells".format(len(prot_groups[area])))
    print("  - Found {} matches".format(len(cat_ix[area])))
    cat_ix[area] = np.array(cat_ix[area])
    prot_ix[area] = np.array(prot_ix[area])

# CAplot.print_dict(model_data["settings"])

select_component = "Stimulus"
component_0 = "Category"
component_1 = "Stimulus-only"
components = [component_0,component_1]
component_prot = "Choice"
min_n_samples = 5

R2 = model_data["R2"]
R2_sh = model_data_sh["R2"]
R2_prot = model_data_prot["R2"]
R2_prot_sh = model_data_prot_sh["R2"]
weight = model_data["weight"]
weight_prot = model_data_prot["weight"]
mouse_id = model_data["mouse_id"]
mouse_id_prot = model_data_prot["mouse_id"]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show weights
csi_cat_ST = collections.OrderedDict()
csi_stim_ST = collections.OrderedDict()
w_choice_prot_ST = collections.OrderedDict()
csi_cat_LG = collections.OrderedDict()
csi_stim_LG = collections.OrderedDict()
w_choice_prot_LG = collections.OrderedDict()
bhv_perf = collections.OrderedDict()

for a_nr,area in enumerate(areas):
    print("Processing area {}".format(area))

    # Get behavioral performance for this area
    bhv_perf[area] = CAgeneral.beh_per_mouse( model_data["bhv-perf"][area], mouse_no_dict=CArec.mouse_no, timepoint=2 )

    # Get significantly predictive neurons in category session
    signtuned_lost = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="lost", n_timepoints=2, component=select_component)
    signtuned_gained = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="gained", n_timepoints=2, component=select_component)
    signtuned_LG = np.stack( [signtuned_lost[:,0], signtuned_gained[:,1]], axis=1 )
    signtuned_ST = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="stable", n_timepoints=2, component=select_component)

    # loop to-compare components
    for c_nr,component in enumerate(components):

        # Prepare matrix for mean and per-frame weights
        left_preds = get_selective_predictors("F02","Left",component_0)
        n_neurons,n_timepoints = weight[area][left_preds[0][0]]["mean"].shape
        compW_left_LG = np.zeros((n_neurons,n_timepoints))
        compW_left_ST = np.zeros((n_neurons,n_timepoints))

        right_preds = get_selective_predictors("F02","Right",component_0)
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
        if c_nr == 0:
            csi_cat_ST[area] = (compW_pref_ST-compW_nonp_ST)/(compW_pref_ST+compW_nonp_ST)
            csi_cat_LG[area] = (compW_pref_LG-compW_nonp_LG)/(compW_pref_LG+compW_nonp_LG)
        elif c_nr == 1:
            csi_stim_ST[area] = (compW_pref_ST-compW_nonp_ST)/(compW_pref_ST+compW_nonp_ST)
            csi_stim_LG[area] = (compW_pref_LG-compW_nonp_LG)/(compW_pref_LG+compW_nonp_LG)

    #----------------------------------------------
    # Now get the choice weights on prototype task

    # Prepare matrix for mean and per-frame weights
    left_preds = get_selective_predictors("F02","Left",component_prot)
    n_neurons,n_timepoints = weight_prot[area][left_preds[0][0]]["mean"].shape
    compW_left_LG = np.zeros((n_neurons,n_timepoints))
    compW_left_ST = np.zeros((n_neurons,n_timepoints))

    right_preds = get_selective_predictors("F02","Right",component_prot)
    n_neurons,n_timepoints = weight_prot[area][right_preds[0][0]]["mean"].shape
    compW_right_LG = np.zeros((n_neurons,n_timepoints))
    compW_right_ST = np.zeros((n_neurons,n_timepoints))

    # Get mean weights
    prev_mouse_name = ""
    for n in range(n_neurons):
        mouse_name = CArec.mouse_name[mouse_id_prot[area][n]]

        # Reload predictor names if necessary
        if mouse_name != prev_mouse_name:
            prev_mouse_name = str(mouse_name)
            left_preds = get_selective_predictors(mouse_name,"Left",component_prot)
            right_preds = get_selective_predictors(mouse_name,"Right",component_prot)

        # Get left predictor weights
        for left_pred in left_preds:
            for pred in left_pred:
                compW_left_LG[n,:] += weight_prot[area][pred]["mean"][n,:]
                compW_left_ST[n,:] += weight_prot[area][pred]["mean"][n,:]
        compW_left_LG[n,:] = compW_left_LG[n,:] / len(left_preds)
        compW_left_ST[n,:] = compW_left_ST[n,:] / len(left_preds)

        # Get right predictor weights
        for right_pred in right_preds:
            for pred in right_pred:
                compW_right_LG[n,:] += weight_prot[area][pred]["mean"][n,:]
                compW_right_ST[n,:] += weight_prot[area][pred]["mean"][n,:]
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

    # Store weights and per-frame weights in data containers
    w_choice_prot_ST[area] = (compW_pref_ST-compW_nonp_ST)/(compW_pref_ST+compW_nonp_ST)
    w_choice_prot_LG[area] = (compW_pref_LG-compW_nonp_LG)/(compW_pref_LG+compW_nonp_LG)


# Keep only neurons of the same group
for a_nr,area in enumerate(areas):
    csi_cat_ST[area] = csi_cat_ST[area][cat_ix[area],:]
    csi_cat_LG[area] = csi_cat_LG[area][cat_ix[area],:]
    csi_stim_ST[area] = csi_stim_ST[area][cat_ix[area],:]
    csi_stim_LG[area] = csi_stim_LG[area][cat_ix[area],:]
    w_choice_prot_ST[area] = w_choice_prot_ST[area][prot_ix[area],:]
    w_choice_prot_LG[area] = w_choice_prot_LG[area][prot_ix[area],:]


#------------------------------------------------------------------------------

print("\nChoice selectivity, stable cells, increased versus decreased category selectivity")
all_deltacti_ST = []
all_choice_ST = []
choice_incr_decr_ST = []
choice_incr_decr_diff_ST = []
for a_nr,area in enumerate(areas):

    # Calculate delta CTI, add to 'all' list
    deltacti_bs = csi_cat_ST[area][:,0]-csi_stim_ST[area][:,0]
    deltacti_lrn = csi_cat_ST[area][:,1]-csi_stim_ST[area][:,1]
    choice_sel = w_choice_prot_ST[area][:,1]
    all_deltacti_ST.append(deltacti_lrn)
    all_choice_ST.append(np.array(choice_sel))

    # Array with all reduced and increased csi neurons
    ix = np.logical_or( deltacti_lrn >= deltacti_bs, np.isnan(deltacti_bs) )
    choice_decr = np.array(choice_sel)
    choice_decr[ix] = np.NaN # lrn > bs = NaN
    ix = np.logical_or( deltacti_lrn < deltacti_bs, np.isnan(deltacti_lrn) )
    choice_incr = np.array(choice_sel)
    choice_incr[ix] = np.NaN  # bs > lrn = NaN

    # Test full data matrix (decreased versus increased)
    choice_incr_decr = np.stack([choice_decr,choice_incr], axis=1)
    choice_incr_decr_ST.append(choice_incr_decr)
    choice_incr_decr_diff_ST.append(choice_incr-np.nanmean(choice_decr))


#--------------------------------------------------------------------
# Correlation between delta CTI and choice selectivity for all cells

# Clean up NaN's
all_deltacti_ST = np.concatenate(all_deltacti_ST, axis=0)
all_choice_ST = np.concatenate(all_choice_ST, axis=0)
isnotnanrow = ~np.logical_or(np.isnan(all_deltacti_ST),np.isnan(all_choice_ST))
all_deltacti_ST = all_deltacti_ST[isnotnanrow]
all_choice_ST = all_choice_ST[isnotnanrow]
r,p = scistats.pearsonr(all_deltacti_ST.ravel(),all_choice_ST.ravel())
print("Correlation delta CTI (stable cells) and choice selectivity\n  r={:6.4f}, p={:6.4f}, n={:0.0f}".format(r,p,len(all_deltacti_ST.ravel())))

# Make figure
fig,ax = CAplot.init_figure_axes(fig_size=(5,4))
x_data = all_deltacti_ST.ravel()
y_data = all_choice_ST.ravel()

# Get linear fit of data
xvalues = np.array([-1,1])
slope, intercept, r_value, p_value, std_err = scistats.linregress(x_data,y_data)
line = slope*xvalues+intercept

# Plot data, linear fit and finish up
CAplot.plt.plot( x_data, y_data, color="None", markerfacecolor="#999999", markersize=3, markeredgewidth=0, marker="o" )
CAplot.plt.plot(xvalues,line, linewidth=1, marker=None, color="#000000")
CAplot.finish_panel( ax, title=None, ylabel="Choice selectivity (2 stim)", xlabel="delta CTI", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[-1,1], x_step=[0.4,1], x_margin=0.02, x_axis_margin=0.01, despine=True)

# Finish figure layout and save
CAplot.finish_figure( filename="6h-DeltaCTI-Vs-Choice-Correlation" )


#---------------------------------------------------------------------------

print("\nChoice selectivity of increased and decreased delta CTI cells")

print("Testing for a difference in any of the groups:")
CAstats.report_kruskalwallis(choice_incr_decr_diff_ST, n_indents=0, alpha=0.05)
print("Posthoc:")

incr_col = (0.7,0,0,1)
decr_col = (0,0,0.7,1)
d_off = 0
i_off = 0
mk = 5
# Loop areas
areas.remove("P")
for a_nr,area in enumerate(areas):
    # if area != "POR":
    #     continue

    # Calculate delta CTI, add to 'all' list
    deltacti_bs = csi_cat_ST[area][:,0]-csi_stim_ST[area][:,0]
    deltacti_lrn = csi_cat_ST[area][:,1]-csi_stim_ST[area][:,1]
    choice_sel = w_choice_prot_ST[area][:,1]

    # Array with all reduced and increased csi neurons
    ix = np.logical_or( deltacti_lrn >= deltacti_bs, np.isnan(deltacti_bs) )
    choice_decr = np.array(choice_sel)
    choice_decr[ix] = np.NaN # lrn > bs = NaN
    ix = np.logical_or( deltacti_lrn < deltacti_bs, np.isnan(deltacti_lrn) )
    choice_incr = np.array(choice_sel)
    choice_incr[ix] = np.NaN  # bs > lrn = NaN

    # Test full data matrix (decreased versus increased)
    choice_incr_decr = np.stack([choice_decr,choice_incr], axis=1)
    b0, a0 = test_before_after( choice_incr_decr, area+", stable cells (decr-dcti vs incr-dcti)", min_n_samples=min_n_samples, paired=False, bonferroni=1, name1="decr", name2="incr" )

    # Create pandas dataframe with two conditions (decr & incr)
    D = np.stack([ np.concatenate([choice_decr,choice_incr],axis=0), np.concatenate( [np.zeros_like(choice_decr), np.zeros_like(choice_incr)+1], axis=0 ) ], axis=1)
    df = pd.DataFrame(D, columns=["data","group"])

    # Plot basic figure containing distribution or single neuron data
    fig,ax = CAplot.init_figure_axes(fig_size=(4,4))
    # CAplot.sns.violinplot(data=df, x="group", y="data", ax=ax, inner=None, dodge=True, palette=[ (.9,.9,.9,1), (.9,.9,.9,1)])
    CAplot.sns.swarmplot(data=df, x="group", y="data", ax=ax, size=3, linewidth=1, edgecolor="None", dodge=True, palette=[ (.5,.5,.5,1), (.5,.5,.5,1)])

    # Draw mean markers
    if b0 is not None:
        CAplot.plt.plot( [0+d_off,0+d_off], [b0.mean-b0.stderr, b0.mean+b0.stderr], linewidth=1, marker=None, color=decr_col, zorder=1000)
        CAplot.plt.plot( 0+d_off, b0.mean, color=decr_col, markerfacecolor=decr_col, marker="o", markeredgewidth=0, markeredgecolor=decr_col, markersize=mk, zorder=1001)
    if b0 is not None:
        CAplot.plt.plot( [1+i_off,1+i_off], [a0.mean-a0.stderr, a0.mean+a0.stderr], linewidth=1, marker=None, color=incr_col, zorder=1002)
        CAplot.plt.plot( 1+i_off, a0.mean, color=incr_col, markerfacecolor=incr_col, marker="o", markeredgewidth=0, markeredgecolor=incr_col, markersize=mk, zorder=1003)

    # Finish up
    # ax.get_legend().set_visible(False)
    CAplot.finish_panel( ax=ax, title=None, ylabel="Choice selectivity (2 stim)", xlabel="Template", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55, x_ticks=np.arange(2), x_ticklabels=["Decr","Incr"], x_tick_rotation=0, tick_size=None, label_size=None, title_size=None, legend_size=None, despine=True, legendpos=0)
    CAplot.finish_figure( filename="6i-DeltaCTI-Vs-Choice-DecreasedIncreased-Area-{}".format(area), path=figpath, wspace=0.8, hspace=0.8 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
