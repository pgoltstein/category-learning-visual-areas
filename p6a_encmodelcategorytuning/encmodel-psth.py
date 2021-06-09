#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13, 2020

Script to get tuning curves matched with encoding model results

@author: pgoltstein
"""

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Get all imports
import numpy as np
import pandas as pd
import os, sys
sys.path.append('../xx_analysissupport')
import CAplot, CAgeneral, CAencodingmodel, CArec, CAstats
import CAanalysissupport as CAsupp
import matdata
import warnings

figpath = "../../figureout/"
base_path = "../../data/chronicrecordings"
model_basepath = "../../data/p5_encodingmodel/"
data_path = '../../data/p6a_encmodelcategorytuning/'

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings, which are due to columns/rows full with NaN's in numpy matrices
warnings.filterwarnings('ignore')

CAplot.font_size["title"] = 6
CAplot.font_size["label"] = 6
CAplot.font_size["tick"] = 6
CAplot.font_size["text"] = 6
CAplot.font_size["legend"] = 6


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Basic settings
load_instead_of_calc = True
areas = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
mice = ['21a','21b','F02','F03','F04','K01','K02','K03','K06','K07']
n_areas = len(areas)
n_mice = len(mice)

# Model settings
select_model_component = "Stimulus"

# PSTH settings
settings = {}
settings["stimulus_type"] = "Category"
settings["n_missing_allowed"] = 1
settings["data_type"] = "spike"
settings["stim_lock"] = "vis_on"
settings["frame_range_bs"] = [-16,-1]
settings["frame_range_stim"] = [1,16]
settings["frame_range_psth"] = [-15,75]


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Load model data
if not load_instead_of_calc:
    model_data = CAsupp.model_load_data("cat-trained", R2="m")
    model_data_sh = CAsupp.model_load_data("shuffled-trials", R2="m")
    model_neurons = np.load(model_basepath+"grouplist-cat.npy", allow_pickle=True).item()
    model_mouse_nrs = model_neurons["mouse_nrs"]
    model_groups = model_neurons["groups"]
    R2 = model_data["R2"]
    R2_sh = model_data_sh["R2"]
    mouse_id_model = model_data["mouse_id"]

# Data output
all_psth = {}
all_tm = {}
all_selectivity = {}
all_cti = {}
all_cat_grid = {}

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Loop areas, find recording data and loop mice
for a_nr,area in enumerate(areas):

    if load_instead_of_calc:
        break

    # Select recordings
    settings["include_recordings"] = [  ("Baseline Out-of-Task",0), ("Baseline Out-of-Task",1), ("Baseline Task",0), ("Baseline Task",1), ("Learned " + settings["stimulus_type"] + " Task",0), ("Learned Out-of-Task",0)   ]

    # Get mouse-recording-sets that have the required recordings
    location_dir = os.path.join( base_path, area )
    print(location_dir)
    mouse_rec_sets,n_mice,n_recs = CArec.get_mouse_recording_sets( os.path.join(location_dir,"*"), settings["include_recordings"], settings["n_missing_allowed"] )

    # Set up data containers
    mouse_psth = []
    mouse_tm = []
    mouse_selectivity = []
    mouse_cti = []
    mouse_cat_grid = []

    # Loop mice
    for m_nr,m_dir in enumerate(mouse_rec_sets):

        # Load chronic imaging recording from directory
        print("\nLoading: {}".format(m_dir))
        crec = CArec.chronicrecording( m_dir )

        # Select recordings and stimuli to include
        recs = []
        for name,nr in settings["include_recordings"]:
            if len(crec.recs[name]) > nr:
                recs.append(crec.recs[name][nr])
            else:
                recs.append(None)
        rec_mouse = recs[0].mouse
        print("Mouse: {}".format(rec_mouse))
        category_stimuli = crec.category_stimuli

        #<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
        # Get the group number of the stimulus modulated neurons in the model

        mouse_no_model = CArec.mouse_no[rec_mouse]
        print("mouse_no_model = {}".format(mouse_no_model))

        signtuned_lost = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="lost", n_timepoints=2, component=select_model_component)
        signtuned_gained = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="gained", n_timepoints=2, component=select_model_component)
        signtuned_LG = np.stack( [signtuned_lost[:,0], signtuned_gained[:,1]], axis=1 )
        sign_model_stable = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="stable", n_timepoints=2, component=select_model_component)

        # sign_model = (signtuned_lost + signtuned_gained + sign_model_stable)>0
        sign_model = sign_model_stable

        sign_ix = np.logical_and( sign_model[:,1], mouse_id_model[area]==mouse_no_model )
        rec_model_groups = model_groups[area][sign_ix]
        print("Selected groups: {}".format(rec_model_groups))
        n_groups = len(rec_model_groups)
        n_frames = len(np.arange( settings["frame_range_psth"][0], settings["frame_range_psth"][1] ))

        # Set up data containers
        rec_psth = np.full((n_groups,6,6,20,n_frames), np.NaN)
        rec_tm = np.full((n_groups,6,6,20), np.NaN)
        rec_selectivity = np.full((n_groups,6), np.NaN)
        rec_cti = np.full((n_groups,6), np.NaN)
        rec_cat_id_grid = np.full((n_groups,6,6,20), np.NaN)

        #<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
        # Loop recordings
        for r_nr,rec in enumerate(recs):
            if rec is None:
                print("{}) Not present: {} #{}".format( r_nr,
                    settings["include_recordings"][r_nr][0],
                    settings["include_recordings"][r_nr][1] ))
                continue

            print("{}) {} (mouse={}, date={})".format( \
                r_nr, rec.timepoint, rec.mouse, rec.date ))

            # Set neuron list to include only groups that are also in model
            rec_groups = rec.groups
            rec_groups_also_in_model = []
            for g in rec_groups:
                if g in rec_model_groups:
                    rec_groups_also_in_model.append(g)
            rec.neuron_groups = rec_groups_also_in_model
            print("  rec.neuron_groups: {}".format(rec.neuron_groups))
            print("  Refound {} out of {} neurons".format( len(rec_groups_also_in_model), len(rec_model_groups) ))
            rec_neuron_groups = rec.neuron_groups

            # Get data matrix
            data_mat = rec.spikes if 'spike' in settings["data_type"] else rec.dror

            # Get stimulus IDs and grid IDs
            stimulus_ids = rec.stimuli
            category_ids = rec.get_trial_category_id(category_stimuli)
            stim_id_grid = rec.get_1d_stimulus_ix_in_2d_grid()
            n_spatialf,n_direction = stim_id_grid.shape
            cat_id_grid = np.full((6,20), np.NaN)
            for sf in range(n_spatialf):
                for dir_ in range(n_direction):
                    stim_id = stim_id_grid[sf,dir_]
                    stim_ixs = np.where(stimulus_ids==stim_id)[0]
                    if len(category_ids[stim_ixs]) > 0:
                        cat_id_grid[sf,dir_] = category_ids[stim_ixs][0]

            # Get PSTH
            psth = CAgeneral.psth( data_mat, rec.vis_on_frames, settings["frame_range_psth"] )
            tm = CAgeneral.tm( data_mat, rec.vis_on_frames, settings["frame_range_stim"] )
            bs = CAgeneral.tm( data_mat, rec.vis_on_frames, settings["frame_range_bs"] )

            # Find tuning curve at category
            n_neurons = len(rec.neurons)
            for n in range(n_neurons):
                left_cat = []
                right_cat = []
                n_ix = int(np.where(rec_model_groups==rec_neuron_groups[n])[0])
                rec_cat_id_grid[n_ix,r_nr,:,:] = cat_id_grid
                for sf in range(n_spatialf):
                    for dir_ in range(n_direction):
                        stim_id = stim_id_grid[sf,dir_]
                        stim_ixs = np.where(stimulus_ids==stim_id)[0]
                        rec_psth[n_ix,r_nr,sf,dir_,:] = np.mean(psth[stim_ixs,n,:],axis=0)
                        rec_tm[n_ix,r_nr,sf,dir_] = np.mean(tm[stim_ixs,n])
                        if len(category_ids[stim_ixs]) > 0:
                            if category_ids[stim_ixs][0] == 1:
                                left_cat.append( rec_tm[n_ix,r_nr,sf,dir_] )
                            if category_ids[stim_ixs][0] == 2:
                                right_cat.append( rec_tm[n_ix,r_nr,sf,dir_] )

                rec_selectivity[n_ix,r_nr] = (np.nanmean(left_cat) - np.nanmean(right_cat)) / (np.nanmean(left_cat) + np.nanmean(right_cat))
                rec_cti[n_ix,r_nr] = CAgeneral.calc_cat_select( np.array(left_cat), np.array(right_cat) )

        mouse_psth.append(rec_psth)
        mouse_tm.append(rec_tm)
        mouse_selectivity.append(rec_selectivity)
        mouse_cti.append(rec_cti)
        mouse_cat_grid.append(rec_cat_id_grid)

    all_psth[area] = np.concatenate(mouse_psth, axis=0)
    all_tm[area] = np.concatenate(mouse_tm, axis=0)
    all_selectivity[area] = np.concatenate(mouse_selectivity, axis=0)
    all_cti[area] = np.concatenate(mouse_cti, axis=0)
    all_cat_grid[area] = np.concatenate(mouse_cat_grid, axis=0)

    print("\n")


if not load_instead_of_calc:

    # Save the just calculated data
    data = {"all_psth": all_psth, "all_tm": all_tm, "all_selectivity": all_selectivity, "all_cti": all_cti, "all_cat_grid": all_cat_grid}
    np.save( os.path.join(data_path,"tuningcurvedata_spike_{}.npy".format(select_model_component)), data )

else:
    # Load data
    tm_data = np.load( os.path.join( data_path, "tuningcurvedata_spike_{}.npy".format(select_model_component) ), allow_pickle=True ).item()
    all_psth = tm_data["all_psth"]
    all_tm = tm_data["all_tm"]
    all_selectivity = tm_data["all_selectivity"]
    all_cti = tm_data["all_cti"]
    all_cat_grid = tm_data["all_cat_grid"]


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Nice tuning curve plots
def realign_according_to_cat( tm, cat_grid ):
    # remove nan row
    if np.sum(np.isnan(cat_grid[-1,:])) == cat_grid.shape[1]:
        cat_grid = cat_grid[:-1,:]
        tm = tm[:-1,:]
    if np.sum(np.isnan(cat_grid[0,:])) == cat_grid.shape[1]:
        cat_grid = cat_grid[1:,:]
        tm = tm[1:,:]

    # Move to center
    ixs_list = np.array( list(range(20)) + list(range(20)) + list(range(20)) + list(range(20)) )
    cat_start = int(np.argwhere( np.nanmax(cat_grid,axis=0)>0 )[0])
    shift_amount = 20 - (7 - cat_start)
    shift_ixs = ixs_list[shift_amount:shift_amount+20]
    cat_grid = cat_grid[:,shift_ixs]
    tm = tm[:,shift_ixs]

    # Flip left category to left side
    if np.nanmax(cat_grid[:,7]) == 2:
        cat_grid = cat_grid[:,::-1]
        tm = tm[:,::-1]

    # Flip left category to be left top
    if cat_grid[0,7] == 1:
        cat_grid = cat_grid[::-1,:]
        tm = tm[::-1,:]

    # Return aligned tuning curves
    return tm,cat_grid

def realign_psth_according_to_cat( psth, cat_grid ):
    # remove nan row
    if np.sum(np.isnan(cat_grid[-1,:])) == cat_grid.shape[1]:
        cat_grid = cat_grid[:-1,:]
        psth = psth[:-1,:,:]
    if np.sum(np.isnan(cat_grid[0,:])) == cat_grid.shape[1]:
        cat_grid = cat_grid[1:,:]
        psth = psth[1:,:,:]

    # Move to center
    ixs_list = np.array( list(range(20)) + list(range(20)) + list(range(20)) + list(range(20)) )
    cat_start = int(np.argwhere( np.nanmax(cat_grid,axis=0)>0 )[0])
    shift_amount = 20 - (7 - cat_start)
    shift_ixs = ixs_list[shift_amount:shift_amount+20]
    cat_grid = cat_grid[:,shift_ixs]
    psth = psth[:,shift_ixs,:]

    # Flip left category to left side
    if np.nanmax(cat_grid[:,7]) == 2:
        cat_grid = cat_grid[:,::-1]
        psth = psth[:,::-1,:]

    # Flip left category to be left top
    if cat_grid[0,7] == 1:
        cat_grid = cat_grid[::-1,:]
        psth = psth[::-1,:,:]

    # Return aligned tuning curves
    return psth,cat_grid

def plot_psth_grid( ax, xvalues, psth, y_scale, category_id_grid ):
    """ Plots an entire grid with psths at the according places """
    x_scale = 1.2*(xvalues[-1]-xvalues[0])
    n_neurons,n_spatialf,n_direction,n_frames = psth.shape
    x_plotted = []
    for sf in range(n_spatialf):
        for dir_ in range(n_direction):
            color = "#999999"
            if category_id_grid[sf,dir_]>0 and category_id_grid[sf,dir_]<1.5:
                color = "#60C3DB"
            elif category_id_grid[sf,dir_]>=1.5:
                color = "#F60951"
            mean_curve,sem_curve,_ = CAgeneral.mean_sem(psth[:,sf,dir_,:],axis=0)
            CAplot.psth_in_grid( gx=dir_, gy=sf, x=xvalues, y=mean_curve, e=sem_curve, x_scale=x_scale, y_scale=y_scale, color=color )
            x_plotted.append(dir_)
    x_left_ticklabels = (min(x_plotted)*x_scale)+xvalues[0]-(0.2*x_scale)
    for y in range(n_spatialf):
        CAplot.plt.text(x_left_ticklabels, y*y_scale, "{}".format(CArec.spatialfs[y]),
            rotation=0, ha='right', va='center', size=6, color='#000000' )
    y_bottom_ticklabels = -0.2*y_scale
    for x in range(n_direction):
        if x >= min(x_plotted) and x <= max(x_plotted):
            CAplot.plt.text((x*x_scale)+np.median(xvalues), y_bottom_ticklabels,
                "{}".format(CArec.directions[x]), rotation=0,
                ha='center', va='top', size=6, color='#000000' )
    ax.set_ylim(-0.6*y_scale,(n_spatialf*y_scale)+(0.2*y_scale))
    ax.set_xlim(-6,212)


areas = ["V1","POR"]
for a_nr,area in enumerate(areas):

    psthmat = np.delete(all_psth[area], 3, 1)
    tunmat = np.delete(all_tm[area], 3, 1)
    catgrid = np.delete(all_cat_grid[area], 3, 1)

    print("Area {}, psthmat shape = {}".format(area,psthmat.shape))
    print("Area {}, tunmat shape = {}".format(area,tunmat.shape))
    print("Area {}, catgrid shape = {}".format(area,catgrid.shape))
    n_neurons,n_timepoints,n_spf,n_dir,n_frames = psthmat.shape

    psth_L = np.full((n_neurons,n_timepoints,5,20,n_frames), np.NaN)
    tm_L = np.full((n_neurons,n_timepoints,5,20), np.NaN)
    cat_L = np.full((n_neurons,n_timepoints,5,20), np.NaN)
    psth_R = np.full((n_neurons,n_timepoints,5,20,n_frames), np.NaN)
    tm_R = np.full((n_neurons,n_timepoints,5,20), np.NaN)
    cat_R = np.full((n_neurons,n_timepoints,5,20), np.NaN)
    LRpref = np.full((n_neurons,), np.NaN)
    for nr in range(n_neurons):
        for tp in [3,0,1,2,4]:
            if np.sum(np.isnan(tunmat[nr,tp,:,:])) == tunmat.shape[2] * tunmat.shape[3]:
                continue

            P, _ = realign_psth_according_to_cat( psthmat[nr,tp,:,:,:], np.array(catgrid[nr,tp,:,:]) )
            X, C = realign_according_to_cat( tunmat[nr,tp,:,:], catgrid[nr,tp,:,:] )

            # Set left/right preferring on first loop iteration
            if np.isnan(LRpref[nr]):
                if np.nanmean(X[C==1]) > np.nanmean(X[C==2]):
                    # Left preferring neuron
                    psth_L[nr,tp,:,:,:] = P
                    tm_L[nr,tp,:,:] = X
                    cat_L[nr,tp,:,:] = C
                else:
                    # Right preferring neuron
                    psth_R[nr,tp,:,:,:] = P
                    tm_R[nr,tp,:,:] = X
                    cat_R[nr,tp,:,:] = C
            else:
                if LRpref[nr] == 1:
                    # Left preferring neuron
                    psth_L[nr,tp,:,:,:] = P
                    tm_L[nr,tp,:,:] = X
                    cat_L[nr,tp,:,:] = C
                else:
                    # Right preferring neuron
                    psth_R[nr,tp,:,:,:] = P
                    tm_R[nr,tp,:,:] = X
                    cat_R[nr,tp,:,:] = C

    tm_L = np.nanmean(tm_L, axis=0)
    cat_L = np.nanmean(cat_L, axis=0)
    tm_R = np.nanmean(tm_R, axis=0)
    cat_R = np.nanmean(cat_R, axis=0)

    vmin = 0
    vmax = np.max([np.nanmax(tm_L),np.nanmax(tm_R)])

    xvalues = np.arange(*settings["frame_range_psth"])/10
    fig = CAplot.init_figure(fig_size=(24,8))
    tp_cnt = 0
    for tp in [0,1,4]:
        ax = CAplot.plt.subplot2grid( (2,3), (0,tp_cnt) )
        plot_psth_grid( ax=ax, xvalues=xvalues, psth=psth_L[:,tp,:,:,:], y_scale=0.1, category_id_grid=cat_R[tp,:,:] ) # cat_R maps nicely to the trained categories
        CAplot.plt.axis('off')

        ax = CAplot.plt.subplot2grid( (2,3), (1,tp_cnt) )
        plot_psth_grid( ax=ax, xvalues=xvalues, psth=psth_R[:,tp,:,:,:], y_scale=0.1, category_id_grid=cat_R[tp,:,:] )
        CAplot.plt.axis('off')

        tp_cnt += 1

    CAplot.finish_figure( filename="6ED10ab-Psth-Area-{}".format(area), path=figpath, wspace=0.3, hspace=0.3 )



#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show
CAplot.plt.show()

# That's all folks!
