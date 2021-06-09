#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 9, 2018

This module contains supporting functions and classes to build an encoding model for calcium imaging data (e.g. represented in the rec structure)

TO DO:
- Eye movements; process in matlab; create regressor; add in model

@author: pgoltstein
"""


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports

import collections
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.stats
import sys, time, os, glob
sys.path.append('../xx_analysissupport')
import CArec, CAplot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global variables
CONSTANT = ["Constant",]
STIM_GO = ["Stim on, go",]
CATEGORIES = ["Left category","Right category"]
STIMULI = ["Stimulus {}".format(s) for s in range(10)]
ORIENTATIONS = ["Orientation {}".format(o) for o in CArec.directions]
SPATIALFS = ["Spatial freq {}".format(f) for f in CArec.spatialfs]
FIRST_LICK = ["Left first lick", "Right first lick", "Left choice, first lick", "Right choice, first lick"]
FIRST_SEQ = ["Left first lick sequence","Right first lick sequence"]
CHOICE = ["Left choice","Right choice"]
LICK = ["Left lick","Right lick"]
RUN = ["Trial running onset","Trial running onset, left choice","Trial running onset, right choice","Speed"]
REWARD = ["Reward","No reward"]

PREDICTOR_GROUPS = collections.OrderedDict( (
    ("Constant", CONSTANT),
    ("Stimulus", CATEGORIES+STIMULI+ORIENTATIONS+SPATIALFS),
    ("Category", CATEGORIES),
    ("Stimulus-only", STIMULI+ORIENTATIONS+SPATIALFS),
    ("Task", STIM_GO),
    ("Choice", FIRST_LICK+FIRST_SEQ+CHOICE+LICK),
    ("Reward", REWARD),
    ("Run", RUN) ) )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions to set up complete models

def get_offsets(rec):
    """ Calculates time offsets between to be modeled events """
    offsets = {}
    offsets["Run onset"] = rec.offset("Stimulus onset", "Trial run onset")
    offsets["First lick"] = rec.offset("Stimulus onset", "First lick")
    offsets["First chosen lick"] = rec.offset("Stimulus onset", "First chosen lick")
    offsets["First chosen lick sequence"] = rec.offset("Stimulus onset", "First chosen lick sequence")
    offsets["Response window"] = rec.offset("Stimulus onset", "Response window")
    offsets["Choice lick"] = rec.offset("Stimulus onset", "Choice lick")
    offsets["Reward"] = rec.offset("Stimulus onset", "Drop")
    return offsets

def init_uber_simple_model(rec, learned_stimuli, stimulus="Category", step=0.33, data_smooth=0.2, principle_range=(-0.9,2.0), unit="Seconds"):
    """ Sets up the model that includes only category stimuli
        returns encodingmodel
    """
    offsets = get_offsets(rec)

    # Internal settings
    smooth = step/2
    stim_range = [principle_range[0], principle_range[1]]
    contin_range = [-1.0,1.0]

    # Initialize kernels
    stimulus_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=stim_range[0], stop=stim_range[1], smooth=smooth, trial_centered=True)
    continuous_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=contin_range[0], stop=contin_range[1], smooth=smooth, trial_centered=False)

    # Initialize reponse model
    include_trials = rec.vis_on_frames[ rec.get_trial_category_id( learned_stimuli ) > 0 ]
    include_trials_stimulus_ids = rec.stimuli[ rec.get_trial_category_id( learned_stimuli ) > 0 ]
    model = encodingmodel( sf=rec.sf, n_frames=rec.n_frames, trial_onset_ixs=include_trials, trial_stimulus_ids=include_trials_stimulus_ids, smooth=data_smooth)
    model.constant = True

    # Add stimulus centered predictors
    if stimulus.lower() == "category":
        model.add_predictor("Left category", rec.boxcar_predictor(name="Left category stimulus onset", learned_stimuli=learned_stimuli), stimulus_kernel, 0.0 )
        model.add_predictor("Right category", rec.boxcar_predictor(name="Right category stimulus onset", learned_stimuli=learned_stimuli), stimulus_kernel, 0.0 )

    elif stimulus.lower() == "orientation-spatialf":
        orientations = np.unique( rec.direction )
        n_orientations = len(orientations)
        orientation_names = ["Orientation {:0.0f}".format(ori) for ori in orientations]
        orientation_kernels = [stimulus_kernel,]*n_orientations
        orientation_offsets = [0.0,]*n_orientations
        model.add_predictor(orientation_names, rec.boxcar_predictor(name="Orientation stimuli onsets", learned_stimuli=learned_stimuli), orientation_kernels, orientation_offsets )

        spatialfs = np.unique( rec.spatialf )
        n_spatialfs = len(spatialfs)
        spatialf_names = ["Spatial freq {:0.2f}".format(spf) for spf in spatialfs]
        spatialf_kernels = [stimulus_kernel,]*n_spatialfs
        spatialf_offsets = [0.0,]*n_spatialfs
        model.add_predictor(spatialf_names, rec.boxcar_predictor(name="Spatial frequency stimuli onsets", learned_stimuli=learned_stimuli), spatialf_kernels, spatialf_offsets )

    elif stimulus.lower() == "stimulus":
        n_stimuli = len( learned_stimuli["left"]["dir"]) + len(learned_stimuli["right"]["dir"] )
        stimulus_names = ["Stimulus {}".format(s) for s in range(n_stimuli)]
        stimulus_kernels = [stimulus_kernel,]*10
        stimulus_offsets = [0.0,]*10
        model.add_predictor(stimulus_names, rec.boxcar_predictor(name="individual stimuli onsets", learned_stimuli=learned_stimuli), stimulus_kernels, stimulus_offsets )

    # All stimulus independent predictors
    model.add_predictor("Speed", rec.boxcar_predictor(name="Speed"), continuous_kernel, 0 )
    return model


def init_full_model_flexible_kernels(rec, learned_stimuli, stimulus="Category", step=0.4, data_smooth=0.2, principle_range=(-0.9,2.0), unit="Seconds"):
    """ Sets up the model that includes all reasonable factors
        returns encodingmodel
    """
    offsets = get_offsets(rec)
    if np.isnan(offsets["Run onset"]):
        include_run_onset = False
    else:
        include_run_onset = True

    # Internal settings
    smooth = step/2
    stim_range = [principle_range[0], principle_range[1]]
    if include_run_onset:
        run_onset_range = (principle_range[0]-offsets["Run onset"], principle_range[1])
    first_lick_range = (principle_range[0]-offsets["First lick"], principle_range[1])
    first_chosen_lick_range = (principle_range[0]-offsets["First chosen lick"], principle_range[1])
    first_lick_sequence_range = (principle_range[0]-offsets["First chosen lick sequence"], principle_range[1])
    choice_range = (principle_range[0]-offsets["Choice lick"], principle_range[1])
    reward_range = [-step-0.01,principle_range[1]]
    contin_range = [-1.0,1.0]

    # Initialize kernels
    stimulus_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=stim_range[0], stop=stim_range[1], smooth=smooth, trial_centered=True)
    if include_run_onset:
        run_onset_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=run_onset_range[0], stop=run_onset_range[1], smooth=smooth, trial_centered=True)
    first_lick_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=first_lick_range[0], stop=first_lick_range[1], smooth=smooth, trial_centered=True)
    first_chosen_lick_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=first_chosen_lick_range[0], stop=first_chosen_lick_range[1], smooth=smooth, trial_centered=True)
    first_lick_sequence_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=first_lick_sequence_range[0], stop=first_lick_sequence_range[1], smooth=smooth, trial_centered=True)
    choice_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=choice_range[0], stop=choice_range[1], smooth=smooth, trial_centered=True)
    reward_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=reward_range[0], stop=reward_range[1], smooth=smooth, trial_centered=True)
    continuous_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=contin_range[0], stop=contin_range[1], smooth=smooth, trial_centered=False)

    # Initialize reponse model
    include_trials = rec.vis_on_frames[ rec.get_trial_category_id( learned_stimuli ) > 0 ]
    include_trials_stimulus_ids = rec.stimuli[ rec.get_trial_category_id( learned_stimuli ) > 0 ]
    model = encodingmodel( sf=rec.sf, n_frames=rec.n_frames, trial_onset_ixs=include_trials, trial_stimulus_ids=include_trials_stimulus_ids, smooth=data_smooth)
    model.constant = True

    # Add stimulus centered predictors
    if stimulus.lower() in ["category","combined"]:
        model.add_predictor("Left category", rec.boxcar_predictor(name="Left category stimulus onset", learned_stimuli=learned_stimuli), stimulus_kernel, 0.0 )
        model.add_predictor("Right category", rec.boxcar_predictor(name="Right category stimulus onset", learned_stimuli=learned_stimuli), stimulus_kernel, 0.0 )

    if stimulus.lower() in ["orientation-spatialf","combined"]:
        orientations = np.unique( rec.direction )
        n_orientations = len(orientations)
        orientation_names = ["Orientation {:0.0f}".format(ori) for ori in orientations]
        orientation_kernels = [stimulus_kernel,]*n_orientations
        orientation_offsets = [0.0,]*n_orientations
        model.add_predictor(orientation_names, rec.boxcar_predictor(name="Orientation stimuli onsets", learned_stimuli=learned_stimuli), orientation_kernels, orientation_offsets )

        spatialfs = np.unique( rec.spatialf )
        n_spatialfs = len(spatialfs)
        spatialf_names = ["Spatial freq {:0.2f}".format(spf) for spf in spatialfs]
        spatialf_kernels = [stimulus_kernel,]*n_spatialfs
        spatialf_offsets = [0.0,]*n_spatialfs
        model.add_predictor(spatialf_names, rec.boxcar_predictor(name="Spatial frequency stimuli onsets", learned_stimuli=learned_stimuli), spatialf_kernels, spatialf_offsets )

    if stimulus.lower() == "stimulus":
        n_stimuli = len( learned_stimuli["left"]["dir"]) + len(learned_stimuli["right"]["dir"] )
        stimulus_names = ["Stimulus {}".format(s) for s in range(n_stimuli)]
        stimulus_kernels = [stimulus_kernel,]*10
        stimulus_offsets = [0.0,]*10
        model.add_predictor(stimulus_names, rec.boxcar_predictor(name="individual stimuli onsets", learned_stimuli=learned_stimuli), stimulus_kernels, stimulus_offsets )

    model.add_predictor("Stim on, go", rec.boxcar_predictor(name="stimulus onset, go", learned_stimuli=learned_stimuli), stimulus_kernel, 0.0 )

    # # First lick
    # model.add_predictor("Left first lick", rec.boxcar_predictor(name="left first lick"), first_lick_kernel, offsets["First lick"] )
    # model.add_predictor("Right first lick", rec.boxcar_predictor(name="Right first lick"), first_lick_kernel, offsets["First lick"] )

    # # First lick on chosen side
    # model.add_predictor("Left choice, first lick", rec.boxcar_predictor(name="left first lick, chosen side"), first_chosen_lick_kernel, offsets["First chosen lick"] )
    # model.add_predictor("Right choice, first lick", rec.boxcar_predictor(name="Right first lick, chosen side"), first_chosen_lick_kernel, offsets["First chosen lick"] )

    # Running onset
    boxcarpred = rec.boxcar_predictor(name="Trial running onset")
    if len(boxcarpred[1]) > 0:
        model.add_predictor("Trial running onset", boxcarpred, run_onset_kernel, offsets["Run onset"] )
    # boxcarpred = rec.boxcar_predictor(name="Trial running onset, left choice")
    # if len(boxcarpred[1]) > 0:
    #     model.add_predictor("Trial running onset, left choice", boxcarpred, run_onset_kernel, offsets["Run onset"] )
    # boxcarpred = rec.boxcar_predictor(name="Trial running onset, right choice")
    # if len(boxcarpred[1]) > 0:
    #     model.add_predictor("Trial running onset, right choice", boxcarpred, run_onset_kernel, offsets["Run onset"] )

    # First lick of sequence on chose side
    model.add_predictor("Left first lick sequence", rec.boxcar_predictor(name="Left first lick sequence"), first_lick_sequence_kernel, offsets["First chosen lick sequence"] )
    model.add_predictor("Right first lick sequence", rec.boxcar_predictor(name="Right first lick sequence"), first_lick_sequence_kernel, offsets["First chosen lick sequence"] )

    # Choice lick in response window
    model.add_predictor("Left choice", rec.boxcar_predictor(name="Left choice lick"), choice_kernel, offsets["Choice lick"] )
    model.add_predictor("Right choice", rec.boxcar_predictor(name="Right choice lick"), choice_kernel, offsets["Choice lick"] )

    # Reward
    model.add_predictor("Reward", rec.boxcar_predictor(name="Reward"), reward_kernel, offsets["Reward"] )
    model.add_predictor("No reward", rec.boxcar_predictor(name="No reward, choice lick"), reward_kernel, offsets["Choice lick"] )

    # All stimulus independent predictors
    # model.add_predictor("Lick", rec.boxcar_predictor(name="Lick"), continuous_kernel, 0 )
    model.add_predictor("Left lick", rec.boxcar_predictor(name="Left lick"), continuous_kernel, 0 )
    model.add_predictor("Right lick", rec.boxcar_predictor(name="Right lick"), continuous_kernel, 0 )
    model.add_predictor("Speed", rec.boxcar_predictor(name="Speed"), continuous_kernel, 0 )

    return model


def init_full_model_fixed_kernels(rec, learned_stimuli, stimulus="Category", step=7, data_smooth=0.5, principle_range=(-15,45), unit="Frames"):
    """ Sets up the model
        returns encodingmodel
    """
    offsets = get_offsets(rec)
    if np.isnan(offsets["Run onset"]):
        include_run_onset = False
    else:
        include_run_onset = True

    # Internal settings
    smooth = step/2

    # Initialize kernels
    stimulus_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=principle_range[0], stop=principle_range[1], smooth=smooth, trial_centered=True)
    if include_run_onset:
        run_onset_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=principle_range[0], stop=principle_range[1], smooth=smooth, trial_centered=True)
    first_lick_sequence_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=principle_range[0], stop=principle_range[1], smooth=smooth, trial_centered=True)
    choice_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=principle_range[0], stop=principle_range[1], smooth=smooth, trial_centered=True)
    reward_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=0.0, stop=principle_range[1], smooth=smooth, trial_centered=True)
    if unit == "Frames":
        continuous_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=-15, stop=15, smooth=smooth, trial_centered=False)
    else:
        continuous_kernel = kernel(sf=rec.sf, unit=unit, step=step, start=-1.0, stop=1.0, smooth=smooth, trial_centered=False)

    # Initialize reponse model
    include_trials = rec.vis_on_frames[ rec.get_trial_category_id( learned_stimuli ) > 0 ]
    include_trials_stimulus_ids = rec.stimuli[ rec.get_trial_category_id( learned_stimuli ) > 0 ]
    model = encodingmodel( sf=rec.sf, n_frames=rec.n_frames, trial_onset_ixs=include_trials, trial_stimulus_ids=include_trials_stimulus_ids, smooth=data_smooth)
    model.constant = True

    # Add stimulus centered predictors
    if stimulus.lower() in ["category","combined"]:
        model.add_predictor("Left category", rec.boxcar_predictor(name="Left category stimulus onset", learned_stimuli=learned_stimuli), stimulus_kernel, 0.0 )
        model.add_predictor("Right category", rec.boxcar_predictor(name="Right category stimulus onset", learned_stimuli=learned_stimuli), stimulus_kernel, 0.0 )

    if stimulus.lower() in ["orientation-spatialf","combined"]:
        orientations = np.unique( rec.direction )
        n_orientations = len(orientations)
        orientation_names = ["Orientation {:0.0f}".format(ori) for ori in orientations]
        orientation_kernels = [stimulus_kernel,]*n_orientations
        orientation_offsets = [0.0,]*n_orientations
        model.add_predictor(orientation_names, rec.boxcar_predictor(name="Orientation stimuli onsets", learned_stimuli=learned_stimuli), orientation_kernels, orientation_offsets )

        spatialfs = np.unique( rec.spatialf )
        n_spatialfs = len(spatialfs)
        spatialf_names = ["Spatial freq {:0.2f}".format(spf) for spf in spatialfs]
        spatialf_kernels = [stimulus_kernel,]*n_spatialfs
        spatialf_offsets = [0.0,]*n_spatialfs
        model.add_predictor(spatialf_names, rec.boxcar_predictor(name="Spatial frequency stimuli onsets", learned_stimuli=learned_stimuli), spatialf_kernels, spatialf_offsets )

    if stimulus.lower() == "stimulus":
        n_stimuli = len( learned_stimuli["left"]["dir"]) + len(learned_stimuli["right"]["dir"] )
        stimulus_names = ["Stimulus {}".format(s) for s in range(n_stimuli)]
        stimulus_kernels = [stimulus_kernel,]*10
        stimulus_offsets = [0.0,]*10
        model.add_predictor(stimulus_names, rec.boxcar_predictor(name="individual stimuli onsets", learned_stimuli=learned_stimuli), stimulus_kernels, stimulus_offsets )

    # Running onset
    boxcarpred = rec.boxcar_predictor(name="Trial running onset")
    if len(boxcarpred[1]) > 0:
        model.add_predictor("Trial running onset", boxcarpred, run_onset_kernel, offsets["Run onset"] )

    # First lick of sequence on chose side
    model.add_predictor("Left first lick sequence", rec.boxcar_predictor(name="Left first lick sequence"), first_lick_sequence_kernel, offsets["First chosen lick sequence"] )
    model.add_predictor("Right first lick sequence", rec.boxcar_predictor(name="Right first lick sequence"), first_lick_sequence_kernel, offsets["First chosen lick sequence"] )

    # Choice lick in response window
    model.add_predictor("Left choice", rec.boxcar_predictor(name="Left choice lick"), choice_kernel, offsets["Choice lick"] )
    model.add_predictor("Right choice", rec.boxcar_predictor(name="Right choice lick"), choice_kernel, offsets["Choice lick"] )

    # Reward
    model.add_predictor("Reward", rec.boxcar_predictor(name="Reward"), reward_kernel, offsets["Reward"] )
    model.add_predictor("No reward", rec.boxcar_predictor(name="No reward, choice lick"), reward_kernel, offsets["Choice lick"] )

    # All stimulus independent predictors
    model.add_predictor("Left lick", rec.boxcar_predictor(name="Left lick"), continuous_kernel, 0 )
    model.add_predictor("Right lick", rec.boxcar_predictor(name="Right lick"), continuous_kernel, 0 )
    model.add_predictor("Speed", rec.boxcar_predictor(name="Speed"), continuous_kernel, 0 )

    return model


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions to operate on model

def duplicate_model(model, without=[]):
    """ Returns a duplicate encoding model, without the predictors specified in the list 'without'
    """
    new_model = encodingmodel( sf=model.sf, n_frames=model.n_frames, trial_onset_ixs=model.trial_onset_ixs, trial_stimulus_ids=model.trial_stimulus_ids, smooth=model.smooth)
    for predictor in model.predictors:
        if not predictor in without:
            new_model.add_predictor( names=predictor, regressors=( model.regressors[predictor], model.indices[predictor] ), kernels=model.kernels[predictor], offsets=model.offsets[predictor] )
    new_model.Y = model._Y_orig
    return new_model

def estimate_predictor_R2( model, groups=None, n_repeats=100, L1=0.1 ):
    """ Iteratively steps through predictors of a model and calculates the R2 without that predictor included
    """
    R2s = collections.OrderedDict()
    models = collections.OrderedDict()

    # Get performance of original model
    print("\n   ### Full model ###")
    models["Full"] = model
    # print("    {}".format(model.predictors))
    print("    Running {}x cross-validated regression on model (L1={})...".format(n_repeats,L1))
    start_time = time.time()
    R2s["Full"] = model.crossvalidated_regression( repeats=n_repeats, L1=L1, nonnegative=True, withinrange=True, shuffle=False)
    print("     -> running time: {:0.2f} s".format(time.time()-start_time))
    print("    Maximum R2: {}".format( np.nanmax(np.nanmean(R2s["Full"],axis=0)) ))

    # Make dictionary with to be excluded predictor lists
    if groups is None:
        groups = collections.OrderedDict()
        for excl_predictor in model.predictors:
            groups[excl_predictor] = [excl_predictor,]

    # Loop over -to be excluded- predictor groups
    for group_name,excl_predictors in groups.items():
        print("\n   ### new model, group name: {} ###".format(group_name))
        models[group_name] = duplicate_model(model, without=excl_predictors)
        # print("    {}".format(models[group_name].predictors))

        # Calculate cross validated R2
        print("    Running {}x cross-validated regression on model (L1={})...".format(n_repeats,L1))
        start_time = time.time()
        R2s[group_name] = models[group_name].crossvalidated_regression( repeats=n_repeats, L1=L1, nonnegative=True, withinrange=True, shuffle=False)
        print("     -> running time: {:0.2f} s".format(time.time()-start_time))
        print("    Maximum R2: {}".format( np.nanmax(np.nanmean(R2s[group_name],axis=0)) ))

    return R2s,models


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions to operate on analyzed data

def load_model_data(datafolder,area,mice,groups):
    """ This function loads the performance and weight data stored in files per mouse/area. It takes one area name, and a list indicating all mice to be included. It reorganizes the performances to a 3d matrix, containing repeats (dim 0), neurons (dim 1) and imaging timepoints (dim 2)
        datafolder: Path to data (string)
        area:       Name of visual area (string)
        mice:       List of mouse names ( list: [string,])
        groups:     Dictionary with predictor groups
        returns
        performances:   dictionary with performance matrix per predictor group
        weights:        dictionary with weight matrix per predictor
        group_weights:  dictionary with weight matrix per predictor group
        mouse_ids:      array containing for each neuron, the unique mouse id
        settings:       Dictionary containing all model settings per mouse
    """

    # Local functions
    def recode_kernel_name(name,ori_cnt,sf_cnt):
        if "orientation" in name.lower():
            name = "Orientation {}".format(ori_cnt)
            ori_cnt += 1
        elif "spatial freq" in name.lower():
            name = "Spatial freq {}".format(sf_cnt)
            sf_cnt += 1
        return name,ori_cnt,sf_cnt

    # Predefine output dictionaries
    performances = collections.OrderedDict()
    weights = collections.OrderedDict()
    group_weights = collections.OrderedDict()
    perframe_weights = collections.OrderedDict()
    behavioral_performances = collections.OrderedDict()
    behavioral_sidebiases = collections.OrderedDict()
    mouse_ids = []

    # Load & store data; count neurons, repeats, timepoints, performance measure
    n_neurons = [0,]
    n_repeats = []
    n_timepoints = []
    p_measure = []
    n_kfr = collections.OrderedDict()
    neg_kfr = collections.OrderedDict()
    kframes = collections.OrderedDict()
    data_dicts = collections.OrderedDict()
    settings = collections.OrderedDict()
    for mouse in mice:
        filename = glob.glob( os.path.join(datafolder, 'data-wi-*encmdl-'+area+'-'+mouse+'-*.npy') )
        if len(filename) == 1:
            print("Loading data from: {}".format(filename[0]))
            data_dicts[mouse] = np.load(filename[0]).item()
            # CAplot.print_dict(data_dicts[mouse], indent=1)
            p_measure.append( data_dicts[mouse]["settings"]["output variable"])
            n_neurons.append( data_dicts[mouse]["data"][p_measure[-1]][0]["Full"].shape[1] )
            n_repeats.append( data_dicts[mouse]["data"][p_measure[-1]][0]["Full"].shape[0] )
            n_timepoints.append(len(data_dicts[mouse]["data"][p_measure[-1]]))
            settings[mouse] = data_dicts[mouse]["settings"]

            for tp in range(len(data_dicts[mouse]["data"]["kfr"])):
                kfrs = data_dicts[mouse]["data"]["kfr"][tp]
                if kfrs is None:
                    continue
                for k in kfrs.keys():
                    if k in n_kfr.keys():
                        if len(kfrs[k]) > n_kfr[k]:
                            neg_kfr[k] = np.sum(kfrs[k]<0)
                            n_kfr[k] = len(kfrs[k])
                            kframes[k] = np.arange( -1*neg_kfr[k], n_kfr[k]-neg_kfr[k], 1 ).astype(int)
                    else:
                        neg_kfr[k] = np.sum(kfrs[k]<0)
                        n_kfr[k] = len(kfrs[k])
                        kframes[k] = np.arange( -1*neg_kfr[k], n_kfr[k]-neg_kfr[k], 1 ).astype(int)
        elif len(filename) > 1:
            print("Ehhh.. multiple files found for area {} and mouse {}:\n{}".format(area,mouse,filename))

    # Number of neurons, list of mice, and indices of neurons of this mouse
    n_total_neurons = int(np.sum(n_neurons))
    neuron_ixs = np.cumsum(n_neurons)
    mice = list(data_dicts.keys())

    # Check performance measure (R2 or r)
    pm_cnt = 0
    for pm in p_measure:
        if pm == p_measure[0]:
            pm_cnt += 1
    if len(p_measure) == pm_cnt:
        p_measure = str(p_measure[0])
    else:
        raise("Found different performance output variables")

    # Check number of repeats
    if len(np.unique(n_repeats)) == 1:
        n_repeats = int(n_repeats[0])
    else:
        raise("Found different numbers of repeats")

    # Check number of timepoints
    if len(np.unique(n_timepoints)) == 1:
        n_timepoints = int(n_timepoints[0])
    else:
        raise("Found different numbers of timepoints")

    # Prepare performance dictionary and matrices for across-mouse data
    ori,sf = 0,0
    for k in data_dicts[mice[0]]["data"][p_measure][0].keys():
        k,ori,sf = recode_kernel_name(k,ori,sf)
        performances[k] = np.full( (n_repeats,n_total_neurons,n_timepoints), np.NaN )

    # Prepare weight dictionary and matrices for across-mouse data
    ori,sf = 0,0
    for k,v in data_dicts[mice[0]]["data"]["w"][0].items():
        k2,ori,sf = recode_kernel_name(k,ori,sf)
        weights[k2] = np.full( (n_repeats,n_total_neurons,n_timepoints), np.NaN )
        if k.lower() != "constant":
            perframe_weights[k2] = np.full( (n_repeats, n_kfr[k], n_total_neurons, n_timepoints), np.NaN )

    # Prepare weight-group dictionary and matrices for across-mouse data
    for gk in groups.keys():
        group_weights[gk] = np.zeros( (n_repeats,n_total_neurons,n_timepoints) )

    # Reorganize data per mouse into matrix with all neurons, sorted per mouse
    for m_nr,(mouse,data_dict) in enumerate(data_dicts.items()):
        print("Extracting mouse {} -> {}".format(mouse,mice[m_nr]))

        # Get range of neuron index for this mouse
        mouse_ix1 = neuron_ixs[m_nr]
        mouse_ix2 = neuron_ixs[m_nr+1]

        # Add behavioral performance and side bias
        behavioral_performances[mouse] = data_dict["data"]["behavioral_performance"]
        behavioral_sidebiases[mouse] = data_dict["data"]["side_bias"]

        # Loop timepoints
        for tp in range(n_timepoints):

            # Skip 'missing' timepoints
            if data_dict["data"][p_measure][tp] is None:
                continue

            # Add performances of all predictor groups
            ori,sf = 0,0
            for k in data_dict["data"][p_measure][tp].keys():
                k2,ori,sf = recode_kernel_name(k,ori,sf)
                performances[k2][:,mouse_ix1:mouse_ix2,tp] = data_dict["data"][p_measure][tp][k]

            # Add weights of all predictors
            ori,sf = 0,0
            for k in data_dict["data"]["w"][tp].keys():
                k2,ori,sf = recode_kernel_name(k,ori,sf)
                weights[k2][:,mouse_ix1:mouse_ix2,tp] = np.nansum(data_dict["data"]["w"][tp][k],axis=1)

                if k.lower() != "constant":
                    neg_ = np.sum(data_dict["data"]["kfr"][tp][k]<0)
                    nkfr = len(data_dict["data"]["kfr"][tp][k])
                    kfrms = neg_kfr[k] + np.arange(-1*neg_, nkfr-neg_, 1).astype(int)
                    perframe_weights[k2][:, kfrms[0]:kfrms[-1]+1, mouse_ix1:mouse_ix2, tp] = data_dict["data"]["w"][tp][k]

            # Add summed weights of all predictor groups
            for gk in groups.keys():
                for wk in groups[gk]:
                    if wk in data_dict["data"]["w"][tp].keys():
                        group_weights[gk][:,mouse_ix1:mouse_ix2,tp] += np.nansum(data_dict["data"]["w"][tp][wk],axis=1)

    # Create list of length n_total_neurons containing mouse number per neuron
    mouse_ids = np.zeros(n_total_neurons)
    mouse_list = []
    for (m_nr,mouse) in enumerate(mice):
        mouse_ids[ neuron_ixs[m_nr]:neuron_ixs[m_nr+1] ] = CArec.mouse_no[mouse]
        mouse_list.append(CArec.mouse_no[mouse])

    # Display summary of loaded data
    print("# neurons={}, repeats={}, timepoints={}".format( n_total_neurons, n_repeats, n_timepoints ))
    print("Performance measure: {}".format(p_measure))
    print("Mouse-neuron boundary indices: {}".format(neuron_ixs))
    print("All mice: {}".format(mice))
    print("All mouse ID's: {}".format(np.unique(mouse_list)))

    return performances, weights, group_weights, perframe_weights, kframes, mouse_ids, settings, behavioral_performances, behavioral_sidebiases


def load_model_data_light(datafolder,area,mice,groups):
    """ This function loads the performance and weight data stored in files per mouse/area. It takes one area name, and a list indicating all mice to be included. It reorganizes the performances to a 3d matrix, containing repeats (dim 0), neurons (dim 1) and imaging timepoints (dim 2)
        datafolder: Path to data (string)
        area:       Name of visual area (string)
        mice:       List of mouse names ( list: [string,])
        groups:     Dictionary with predictor groups
        returns
        performances:   dictionary with performance matrix per predictor group
        mouse_ids:      array containing for each neuron, the unique mouse id
        settings:       Dictionary containing all model settings per mouse
    """

    # Local functions
    def recode_kernel_name(name,ori_cnt,sf_cnt):
        if "orientation" in name.lower():
            name = "Orientation {}".format(ori_cnt)
            ori_cnt += 1
        elif "spatial freq" in name.lower():
            name = "Spatial freq {}".format(sf_cnt)
            sf_cnt += 1
        return name,ori_cnt,sf_cnt

    # Predefine output dictionaries
    performances = collections.OrderedDict()
    behavioral_performances = collections.OrderedDict()
    behavioral_sidebiases = collections.OrderedDict()
    mouse_ids = []

    # Load & store data; count neurons, repeats, timepoints, performance measure
    n_neurons = [0,]
    n_repeats = []
    n_timepoints = []
    p_measure = []
    data_dicts = collections.OrderedDict()
    settings = collections.OrderedDict()
    for mouse in mice:
        filename = glob.glob( os.path.join(datafolder, 'data-wi-*encmdl-'+area+'-'+mouse+'-*.npy') )
        if len(filename) == 1:
            print("Loading data from: {}".format(filename[0]))
            data_dicts[mouse] = np.load(filename[0], allow_pickle=True).item()
            # CAplot.print_dict(data_dicts[mouse], indent=1)
            p_measure.append( data_dicts[mouse]["settings"]["output variable"])
            n_neurons.append( data_dicts[mouse]["data"][p_measure[-1]][0]["Full"].shape[1] )
            n_repeats.append( data_dicts[mouse]["data"][p_measure[-1]][0]["Full"].shape[0] )
            n_timepoints.append(len(data_dicts[mouse]["data"][p_measure[-1]]))
            settings[mouse] = data_dicts[mouse]["settings"]
        elif len(filename) > 1:
            print("Ehhh.. multiple files found for area {} and mouse {}:\n{}".format(area,mouse,filename))

    # Number of neurons, list of mice, and indices of neurons of this mouse
    n_total_neurons = int(np.sum(n_neurons))
    neuron_ixs = np.cumsum(n_neurons)
    mice = list(data_dicts.keys())

    # Check performance measure (R2 or r)
    pm_cnt = 0
    for pm in p_measure:
        if pm == p_measure[0]:
            pm_cnt += 1
    if len(p_measure) == pm_cnt:
        p_measure = str(p_measure[0])
    else:
        raise("Found different performance output variables")

    # Check number of repeats
    if len(np.unique(n_repeats)) == 1:
        n_repeats = int(n_repeats[0])
    else:
        raise("Found different numbers of repeats")

    # Check number of timepoints
    if len(np.unique(n_timepoints)) == 1:
        n_timepoints = int(n_timepoints[0])
    else:
        raise("Found different numbers of timepoints")

    # Prepare performance dictionary and matrices for across-mouse data
    ori,sf = 0,0
    for k in data_dicts[mice[0]]["data"][p_measure][0].keys():
        k,ori,sf = recode_kernel_name(k,ori,sf)
        performances[k] = np.full( (n_repeats,n_total_neurons,n_timepoints), np.NaN )

    # Reorganize data per mouse into matrix with all neurons, sorted per mouse
    for m_nr,(mouse,data_dict) in enumerate(data_dicts.items()):
        print("Extracting mouse {} -> {}".format(mouse,mice[m_nr]))

        # Get range of neuron index for this mouse
        mouse_ix1 = neuron_ixs[m_nr]
        mouse_ix2 = neuron_ixs[m_nr+1]

        # Add behavioral performance and side bias
        behavioral_performances[mouse] = data_dict["data"]["behavioral_performance"]
        behavioral_sidebiases[mouse] = data_dict["data"]["side_bias"]

        # Loop timepoints
        for tp in range(n_timepoints):

            # Skip 'missing' timepoints
            if data_dict["data"][p_measure][tp] is None:
                continue

            # Add performances of all predictor groups
            ori,sf = 0,0
            for k in data_dict["data"][p_measure][tp].keys():
                k2,ori,sf = recode_kernel_name(k,ori,sf)
                performances[k2][:,mouse_ix1:mouse_ix2,tp] = data_dict["data"][p_measure][tp][k]

    # Create list of length n_total_neurons containing mouse number per neuron
    mouse_ids = np.zeros(n_total_neurons)
    mouse_list = []
    for (m_nr,mouse) in enumerate(mice):
        mouse_ids[ neuron_ixs[m_nr]:neuron_ixs[m_nr+1] ] = CArec.mouse_no[mouse]
        mouse_list.append(CArec.mouse_no[mouse])

    # Display summary of loaded data
    print("# neurons={}, repeats={}, timepoints={}".format( n_total_neurons, n_repeats, n_timepoints ))
    print("Performance measure: {}".format(p_measure))
    print("Mouse-neuron boundary indices: {}".format(neuron_ixs))
    print("All mice: {}".format(mice))
    print("All mouse ID's: {}".format(np.unique(mouse_list)))

    return performances, mouse_ids, settings, behavioral_performances, behavioral_sidebiases



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Kernel class

class kernel(object):
    """ This class represents an individual kernel in the encoding model """

    # Initialization routine
    def __init__(self, sf, unit="Seconds", step=0.5, start=-1, stop=2, smooth=0.5, trial_centered=True ):
        """ Defines a read-only kernel using default, or passed parameters """

        # Store the passed-on unit as either 'seconds' or 'frames'
        if "fr" in unit.lower():
            self._unit = "frames"
        elif unit.lower()[0] == "s":
            self._unit = "seconds"
        else:
            raise Exception("Unit '{}' is not recognized".format(unit))

        # Store other passed variables
        self._sf = sf
        self._trial_centered = trial_centered

        # Calulate the smooth in frames
        if self._unit == "seconds":
            self._smooth = smooth * self._sf
        else:
            self._smooth = smooth

        # Calculate the kernel frames
        self._calculate_frames(start, stop, step)

    def __str__(self):
        """ Shows string summary output """
        return "Kernel, {} fr, <{:4.2f}  {:4.2f}  {:4.2f}>, sigma={:4.2f} fr".format( self.n_frames, self._start, self._step, self._stop, self.smooth )

    @property
    def n_frames(self):
        """ Number of frames in kernel """
        return self._frames.shape[0]

    @property
    def n_allframes(self):
        """ Number of frames between first and last kernel frame """
        return self._allframes.shape[0]

    @property
    def smooth(self):
        """ Returns the kernel smooth in frames """
        return self._smooth

    @property
    def trial_centered(self):
        """ returns True if the x-values are in reference to trial onset """
        return self._trial_centered

    @property
    def times(self):
        """ Returns the kernel array in seconds, centered at zero """
        return self._frames / self._sf

    @property
    def alltimes(self):
        """ Returns all positions of the kernel array in seconds, centered at zero """
        return self._allframes / self._sf

    @property
    def frames(self):
        """ Returns the kernel array in frames, centered at zero """
        return self._frames

    @property
    def allframes(self):
        """ Returns all frames within the range of the kernel array, centered at zero """
        return self._allframes

    def _calculate_frames(self, start, stop, step):
        """ Calculates and sets the kernel in frames, centered at zero """
        if self._unit == "seconds":
            arr_before_zero = np.arange(0.0, (start*self._sf)-0.01, step*self._sf*-1)[::-1]
            arr_after_zero = np.arange(0.0, (stop*self._sf)+0.01, step*self._sf)[1::]
            self._step = np.round( step, decimals=2 )
        else:
            arr_before_zero = np.arange(0.0, start-0.01, step*-1)[::-1]
            arr_after_zero = np.arange(0.0, stop+0.01, step)[1::]
            self._step = np.round( step/self._sf, decimals=2 )
        arr_all_before_zero = np.arange(0.0, arr_before_zero[0]-0.5, -1)[::-1]
        arr_all_after_zero = np.arange(0.0, arr_after_zero[-1]+0.5, 1)[1::]
        self._frames = np.round( np.concatenate( [arr_before_zero,arr_after_zero])).astype(int)
        self._allframes = np.round( np.concatenate( [arr_all_before_zero,arr_all_after_zero])).astype(int)
        self._start = np.round( self._frames[0]/self._sf, decimals=2 )
        self._stop = np.round( self._frames[-1]/self._sf, decimals=2 )



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Encoding model class

class encodingmodel(object):
    """ This class manages the heavy lifting on an encoding model """

    # Initialization routine
    def __init__(self, sf, n_frames, trial_onset_ixs, trial_stimulus_ids, smooth=0):

        # Set internal variables
        self._smooth = smooth
        self._n_frames = n_frames
        self._sf = sf
        self._trial_onset_ixs = trial_onset_ixs
        self._trial_stimulus_ids = trial_stimulus_ids
        self._constant = False

        # Define unset internal variables
        self._n_neurons = None
        self._xlim = [0,0]
        self._Y = None
        self._Y_orig = None
        self._predictors = collections.OrderedDict()
        self._indices = collections.OrderedDict()
        self._kernels = collections.OrderedDict()
        self._weights = collections.OrderedDict()
        self._weightedkernels = collections.OrderedDict()
        self._xvalues = collections.OrderedDict()
        self._offsets = collections.OrderedDict()
        self._trial_centered = collections.OrderedDict()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # +++ General properties and functions +++
    @property
    def sf(self):
        return self._sf

    @property
    def n_frames(self):
        return self._n_frames

    @property
    def n_neurons(self):
        return self._n_neurons

    @property
    def xlim(self):
        return self._xlim

    @property
    def smooth(self):
        return self._smooth

    @property
    def trial_onset_ixs(self):
        return self._trial_onset_ixs

    @property
    def trial_stimulus_ids(self):
        return self._trial_stimulus_ids

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # +++ Functions for handling predictors and kernels +++
    @property
    def n_predictors(self):
        """ Number of predictors """
        return len(self._predictors)

    @property
    def constant(self):
        """ Toggle inclusion of constant factor """
        return self._constant

    @constant.setter
    def constant(self, value):
        """ Toggle inclusion of constant factor """
        self._constant = value

    @property
    def predictors(self):
        """ Returns list with predictor names """
        return list(self._predictors.keys())

    @property
    def kernels(self):
        """ Returns dictionary with kernels """
        return self._kernels

    @property
    def regressors(self):
        """ Returns dictionary with regressor arrays """
        return self._predictors

    @property
    def indices(self):
        """ Returns dictionary with indices of the regressors """
        return self._indices

    @property
    def weights(self):
        """ Returns dictionary with regressor weights """
        return self._weights

    @property
    def weightedkernels(self):
        """ Returns dictionary with regressor weightedkernels """
        return self._weightedkernels

    @property
    def xvalues(self):
        """ Returns dictionary with regressor xvalues """
        return self._xvalues

    @property
    def offsets(self):
        """ Returns dictionary with xvalue offsets """
        return self._offsets

    @property
    def trial_centered(self):
        """ Returns dictionary indicating whether the xvalues are trial centered or only offset to its own indices """
        return self._trial_centered

    def add_predictor(self, names, regressors, kernels, offsets ):
        """ Adds one or more predictor(s) to the model. When supplied lists, it will iterate over these lists using recursion
            names:        Name of regressors/predictors
            regressors:   tuple containing a vector with regression values and a list of indices with event onsets
            kernels:      Kernel from the kernel class (above)
            offsets:      Offset of event relative to stimulus onset (if appropriate, otherwise zero)
        """

        # If multiple names supplied, iterate over list and use recursion
        if isinstance(names,list):
            for name,regressor,kernel,offset in zip(names,regressors,kernels,offsets):
                self.add_predictor(name,regressor,kernel,offset)
        else:
            self._predictors[names] = regressors[0].ravel()
            self._indices[names] = regressors[1].ravel()
            self._kernels[names] = kernels
            self._weights[names] = None
            self._weightedkernels[names] = None
            self._offsets[names] = offsets
            self._xvalues[names] = kernels.alltimes + offsets
            self._trial_centered[names] = kernels.trial_centered
            self._xlim[0] = np.min([self._xlim[0],kernels.times[0]+offsets])
            self._xlim[1] = np.max([self._xlim[1],kernels.times[-1]+offsets])

    def calculate_weightkernel(self, kernel, weights):
        """ Calculates the weights of the kernel per frame """

        # Fill matrix with single boxcar for each kernel frame
        kern_mat = np.zeros((kernel.n_allframes,kernel.n_frames))
        for nr,f in enumerate(kernel.frames-np.min(kernel.frames)):
            kern_mat[f,nr] = 1.0

        # Smooth rows
        kern_mat = scipy.ndimage.gaussian_filter1d( kern_mat, sigma=kernel.smooth, axis=0)

        # Normalize to a range between 0 and 1
        for x in range(kern_mat.shape[1]):
            kern_mat[:,x] = kern_mat[:,x] / np.max(kern_mat[:,x])

        # Return sum of all rows convolved with the weight vector
        return np.dot( kern_mat, np.expand_dims(weights,axis=1) )[:,0]


    @property
    def withinrange_ixs(self):
        """ returns a boolean vector that indexes only frames within the range of the _trial_centered predictors """
        withinrange_ixs = np.zeros(self.n_frames)
        for name,ixs in self._indices.items():
            if self._trial_centered[name]:
                start,stop = np.round(self._xvalues[name][0]*self.sf), np.round(self._xvalues[name][-1]*self.sf)
                for ix in ixs:
                    withinrange_ixs[int(ix+start):int(ix+stop)] = 1.0
        return withinrange_ixs==1.0

    @property
    def trial_subset_ixs(self):
        """ Returns two boolean vectors that index mutually exclusive subsets of trials, ratio 70/30 """

        # Get range of frames to include around stimulus onset
        trial_range = [0,0]
        for name,xvalues in self._xvalues.items():
            if self._trial_centered[name]:
                trial_range[0] = np.min([trial_range[0],xvalues[0]*self.sf])
                trial_range[1] = np.max([trial_range[1],xvalues[-1]*self.sf])
        trial_range = np.array(trial_range).astype(int)

        # Get sorting vector
        trial_sort = np.zeros_like(self.trial_onset_ixs)
        trial_sort[:np.round(trial_sort.shape[0]*0.7).astype(int)] = 1.0
        np.random.shuffle(trial_sort)

        # Sort trials to boolean vectors
        subset1_ixs = np.zeros(self.n_frames)
        subset2_ixs = np.zeros(self.n_frames)
        for nr,ix in enumerate(self.trial_onset_ixs):
            if trial_sort[nr] == 1:
                subset1_ixs[int(ix+trial_range[0]):int(ix+trial_range[1])] = 1
            else:
                subset2_ixs[int(ix+trial_range[0]):int(ix+trial_range[1])] = 1
        return subset1_ixs==1,subset2_ixs==1

    def replace_category_predictors_with_( self, X, kernel_indices, rec, crec, swap=False, shuffle=False ):
        """ shuffles which stimulus belongs to which category, but leaves everything else intact """

        if shuffle:
            learned_stimuli = crec.shuffled_category_stimuli
        elif swap:
            learned_stimuli = crec.swapped_category_stimuli
        for catname in ["Left category","Right category"]:
            predictor,_ = rec.boxcar_predictor(name=catname+" stimulus onset", learned_stimuli=learned_stimuli)
            predictor = np.reshape(predictor,(predictor.shape[0],1))
            kernel = self._kernels[catname]

            # Create the predictor block
            C = np.ones((self.n_frames,1))
            for shift in kernel.frames:
                C = np.concatenate( [C, scipy.ndimage.gaussian_filter1d( np.roll(predictor,shift), sigma=kernel.smooth, axis=0)], axis=1 )
            C = np.delete(C, 0, axis=1)

            # Normalize regressors to a range between 0 and 1
            for x in range(C.shape[1]):
                if len(np.unique(C[:,x])) > 1:
                    C[:,x] = (C[:,x]-np.min(C[:,x])) / (np.max(C[:,x])-np.min(C[:,x]))

            # Replace predictor in X
            X[:,kernel_indices[catname]] = C
        return X


    def psth(self, Y, name):
        """ Creates a psth based on the predictor onsets and indices """
        ixs = self._indices[name].astype(int)
        frames = self._kernels[name].allframes.astype(int)
        if Y.ndim == 1:
            psth = np.zeros((len(ixs),len(frames)))
            for t,ix in enumerate(ixs):
                psth[t,:] = Y[ix+frames]
        else:
            psth = np.zeros((len(ixs),len(frames),Y.shape[1]))
            for t,ix in enumerate(ixs):
                psth[t,:,:] = Y[ix+frames,:]
        return psth

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # +++ Functions for creating the design matrix +++
    @property
    def X(self):
        """ Puts predictors together in design matrix """

        # Start with zeros column, remove this column at the end
        X = np.ones((self.n_frames,1))

        # Iterate over predictors and kernels
        for nr,((name,predictor),(name,kernel)) in enumerate(zip(self._predictors.items(),self._kernels.items())):

            predictor = np.reshape(predictor,(predictor.shape[0],1))

            if kernel is None:
                X = np.concatenate( [X, predictor], axis=1 )
            else:
                for shift in kernel.frames:
                    X = np.concatenate( [X, scipy.ndimage.gaussian_filter1d( np.roll(predictor,shift), sigma=kernel.smooth, axis=0)], axis=1 )

        # Remove leading zeros column
        if not self.constant:
            X = np.delete(X, 0, axis=1)

        # Normalize regressors to a range between 0 and 1
        for x in range(X.shape[1]):
            if len(np.unique(X[:,x])) > 1:
                X[:,x] = (X[:,x]-np.min(X[:,x])) / (np.max(X[:,x])-np.min(X[:,x]))

        # Return predictor matrix
        return X

    @property
    def Y(self):
        """ Property that represents the response matrix """
        return self._Y

    @Y.setter
    def Y(self, data_matrix, clip=1.0):
        """ Sets and optionally smooths the response matrix ()
            data_matrix:   2d matrix [frames x neuron-no]
        """
        self._Y_orig = data_matrix
        data_matrix = np.array(data_matrix)
        self._n_neurons = data_matrix.shape[1]

        # Clip data
        if clip is not None:
            print("   -> clipped {} datapoints to {}".format(np.sum(data_matrix>clip),clip))
            data_matrix[data_matrix>clip] = clip

        if self._smooth > 0:
            # Smooth the data matrix if this was determined at class init
            data_smooth_fr = self._smooth*self.sf
            self._Y = scipy.ndimage.gaussian_filter1d(data_matrix, sigma=data_smooth_fr, axis=0)
        else:
            self._Y = data_matrix


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # +++ Functions for performing the regression +++
    def regression(self, L1=1.0, nonnegative=False, withinrange=False, framerange=None, shuffle=False):
        """ Run regression per single trace using scipy.optimize.nnls or scipy.optimize.lsq_linear. Because the function does not include a regularization parameter, we add it via the predictor matrix
            L1: Lambda value for L1 regularization
            Returns a list of 2D matrices [kernel-no][kernel-weights x neuron#]
        """

        # Get data to be regressed
        X = self.X
        Y = self.Y

        # Select frames part of framerange
        if framerange is not None:
            X = X[framerange,:]
            Y = Y[framerange,:]

        # Select frames within the range of the regressors
        if withinrange:
            withinrange_ixs = self.withinrange_ixs
            if framerange is not None:
                withinrange_ixs = withinrange_ixs[framerange]
            X = X[withinrange_ixs,:]
            Y = Y[withinrange_ixs,:]

        # Shuffle the training targets if requested
        if shuffle:
            Y = np.array(Y)
            np.random.shuffle(Y)

        # Add regularization
        Xorig = np.array(X)
        Yorig = np.array(Y)
        if L1 > 0:
            L1 = L1 * np.sqrt(Y.shape[0])
            X = np.concatenate([X, L1*np.eye(X.shape[1])], axis=0)
            Y = np.concatenate([Y, np.zeros((X.shape[1],Y.shape[1]))], axis=0)

        # Get number of kernel frames
        kernel_indices = collections.OrderedDict()
        if not self.constant:
            ix = 0
        else:
            ix = 1
            self._weights["Constant"] = np.zeros((1,self.n_neurons))
        for name,kernel in self._kernels.items():
            self._weights[name] = np.zeros((kernel.n_frames,self.n_neurons))
            self._weightedkernels[name] = np.zeros((kernel.n_allframes,self.n_neurons))
            kernel_indices[name] = np.arange(ix,ix+kernel.n_frames)
            ix += kernel.n_frames

        # Loop neurons in Y
        R2 = np.zeros(self.n_neurons)
        for n,y in enumerate(Y.T):

            # Solve equation using non-negative least squares to get weights
            if nonnegative:
                (w,_) = scipy.optimize.nnls(X, y)
            else:
                w = scipy.optimize.lsq_linear(X, y)["x"]

            # Store weights to individual kernels
            if self.constant:
                self._weights["Constant"][0,n] = w[0]
            for name in self._kernels.keys():
                self._weights[name][:,n] = w[ kernel_indices[name] ]

                # Calculate and store weightkernel
                self._weightedkernels[name][:,n] = self.calculate_weightkernel( self._kernels[name], w[ kernel_indices[name] ])

            # Calculate model performance
            p = np.dot( Xorig, np.expand_dims(w,axis=1) )[:,0]
            y_orig = Yorig[:,n]
            SStotal = np.sum( (y_orig-np.mean(y_orig))**2 )
            # SStotal = np.sum( y_orig**2 )
            if SStotal > 0:
                SSresidual = np.sum( (y_orig-p)**2 )
                R2[n] = 1.0 - (SSresidual/SStotal)
            else:
                R2[n] = np.NaN

        return R2


    def crossvalidated_regression(self, repeats=10, L1=1.0, nonnegative=False, withinrange=False, shuffle=False, shuffle_trials=False, shuffle_group=None, shuffle_all_but_group=None, shuffle_categories=False, swap_categories=False, return_var_per_group=None, output_var="R2m", progress_indicator=True, rec=None, crec=None, shuffle_group_dict=None):
        """ Run regression per single trace using scipy.optimize.nnls or scipy.optimize.lsq_linear. Because the function does not include a regularization parameter, we add it via the predictor matrix. Repeatedly splits the data in half, based on trials.
            L1: Lambda value for L1 regularization
            return_var_per_group: Dictionary with groups
            Returns a list of 2D matrices [kernel-no][kernel-weights x neuron#]
        """

        # Get design matrix and data to be regressed
        X = self.X
        Y = self.Y

        # Get number of kernel frames
        kernel_indices = collections.OrderedDict()
        if not self.constant:
            ix = 0
        else:
            ix = 1
            self._weights["Constant"] = np.zeros((repeats,1,self.n_neurons))
            kernel_indices["Constant"] = 0
        for name,kernel in self._kernels.items():
            self._weights[name] = np.zeros((repeats,kernel.n_frames,self.n_neurons))
            self._weightedkernels[name] = np.zeros((repeats,kernel.n_allframes,self.n_neurons))
            kernel_indices[name] = np.arange(ix,ix+kernel.n_frames)
            ix += kernel.n_frames

        # Get withinrange subselector
        if withinrange:
            withinrange_ixs = self.withinrange_ixs
        else:
            withinrange_ixs = np.ones(self.n_frames)==1

        # Prepare return variables
        return_var = np.zeros((repeats,self.n_neurons))
        if return_var_per_group is not None:
            group_return_var = collections.OrderedDict()
            group_return_var["Full"] = None
            for name in return_var_per_group.keys():
                group_return_var[name] = np.zeros((repeats,self.n_neurons))

        # Loop repeats
        n_loops = repeats*self.n_neurons
        loop_count = 0.0
        if progress_indicator:
            print("    Progress: {:6.2f}%".format(0), end="", flush=True)
        for r in range(repeats):

            # Recreate the category predictors if shuffle categories
            if shuffle_categories:
                X = self.replace_category_predictors_with_( X=X, kernel_indices=kernel_indices, rec=rec, crec=crec, swap=False, shuffle=True )
            elif swap_categories:
                X = self.replace_category_predictors_with_( X=X, kernel_indices=kernel_indices, rec=rec, crec=crec, swap=True, shuffle=False )

            # Select frames part of framerange
            if not shuffle_trials:
                subs1,subs2 = self.trial_subset_ixs
                Xsubs1 = X[ np.logical_and(subs1,withinrange_ixs), : ]
                Ysubs1 = Y[ np.logical_and(subs1,withinrange_ixs), : ]
                Xsubs2 = X[ np.logical_and(subs2,withinrange_ixs), : ]
                Ysubs2 = Y[ np.logical_and(subs2,withinrange_ixs), : ]
            else:
                subs1,subs2 = self.trial_subset_ixs
                subs3,subs4 = self.trial_subset_ixs
                Xsubs1 = X[ np.logical_and(subs1,withinrange_ixs), : ]
                Ysubs1 = Y[ np.logical_and(subs3,withinrange_ixs), : ]
                Xsubs2 = X[ np.logical_and(subs2,withinrange_ixs), : ]
                Ysubs2 = Y[ np.logical_and(subs4,withinrange_ixs), : ]
                min_fr1 = np.min((Xsubs1.shape[0],Ysubs1.shape[0]))
                min_fr2 = np.min((Xsubs2.shape[0],Ysubs2.shape[0]))
                Xsubs1 = Xsubs1[:min_fr1,:]
                Ysubs1 = Ysubs1[:min_fr1,:]
                Xsubs2 = Xsubs2[:min_fr2,:]
                Ysubs2 = Ysubs2[:min_fr2,:]

            # Shuffle frames for a single group in time
            if shuffle_group is not None:
                subs1_nframes = Xsubs1.shape[0]
                subs2_nframes = Xsubs2.shape[0]
                if shuffle_group.lower() != "shuffle-no-groups":
                    # Loop all kernel-names in the group
                    for w_name in shuffle_group_dict[shuffle_group]:
                        # if kernel name is in model, then shuffle
                        if w_name in kernel_indices.keys():
                            # Make random index for subsamples 1 and 2
                            rand_ixs1 = np.arange(subs1_nframes)
                            np.random.shuffle(rand_ixs1)
                            rand_ixs2 = np.arange(subs2_nframes)
                            np.random.shuffle(rand_ixs2)
                            # Use the index to shuffle the entries for the kernel
                            for k_ix in kernel_indices[w_name]:
                                Xsubs1[:,k_ix] = Xsubs1[rand_ixs1,k_ix]
                                Xsubs2[:,k_ix] = Xsubs2[rand_ixs2,k_ix]

            # Shuffle frames of all groups but one
            if shuffle_all_but_group is not None:
                subs1_nframes = Xsubs1.shape[0]
                subs2_nframes = Xsubs2.shape[0]
                if shuffle_all_but_group.lower() == "shuffle-all-groups":
                    exclude_these_kernels = []
                else:
                    exclude_these_kernels = shuffle_group_dict[shuffle_all_but_group]
                # Loop all kernel-names in the model
                for w_name in kernel_indices.keys():
                    # if kernel name is in model, then shuffle
                    if w_name not in exclude_these_kernels:
                        # Make random index for subsamples 1 and 2
                        rand_ixs1 = np.arange(subs1_nframes)
                        np.random.shuffle(rand_ixs1)
                        rand_ixs2 = np.arange(subs2_nframes)
                        np.random.shuffle(rand_ixs2)
                        # Use the index to shuffle the entries for the kernel
                        if type(kernel_indices[w_name]) is int:
                            k_ix = kernel_indices[w_name]
                            Xsubs1[:,k_ix] = Xsubs1[rand_ixs1,k_ix]
                            Xsubs2[:,k_ix] = Xsubs2[rand_ixs2,k_ix]
                        else:
                            for k_ix in kernel_indices[w_name]:
                                Xsubs1[:,k_ix] = Xsubs1[rand_ixs1,k_ix]
                                Xsubs2[:,k_ix] = Xsubs2[rand_ixs2,k_ix]

            # Shuffle the training targets if requested
            if shuffle:
                np.random.shuffle(Ysubs1)

            # Add regularization
            if L1 > 0:
                L1_r = L1 * np.sqrt(Ysubs1.shape[0])
                Xsubs1 = np.concatenate([Xsubs1, L1_r*np.eye(Xsubs1.shape[1])], axis=0)
                Ysubs1 = np.concatenate([Ysubs1, np.zeros((Xsubs1.shape[1],Ysubs1.shape[1]))], axis=0)

            # Loop neurons in Y
            for n,y in enumerate(Ysubs1.T):

                # Solve equation using non-negative least squares to get weights
                if nonnegative:
                    (w,_) = scipy.optimize.nnls(Xsubs1, y)
                else:
                    w = scipy.optimize.lsq_linear(Xsubs1, y)["x"]

                # Store weights to individual kernels
                if self.constant:
                    self._weights["Constant"][r,0,n] = w[0]
                for name in self._kernels.keys():
                    self._weights[name][r,:,n] = w[ kernel_indices[name] ]
                    self._weightedkernels[name][r,:,n] = self.calculate_weightkernel( self._kernels[name], w[ kernel_indices[name] ])

                # Calculate cross validated model performance
                p2 = np.dot( Xsubs2, np.expand_dims(w,axis=1) )[:,0]
                y2 = Ysubs2[:,n]

                if "R2" in output_var:
                    if output_var == "R2m":
                        SStotal = np.sum( (y2-np.mean(y2))**2 )
                    elif output_var == "R2z":
                        SStotal = np.sum( y2**2 )
                    if SStotal > 0:
                        SSresidual = np.sum( (y2-p2)**2 )
                        return_var[r,n] = 1.0 - (SSresidual/SStotal)
                    else:
                        return_var[r,n] = np.NaN
                elif output_var == "r":
                    return_var[r,n] = scipy.stats.stats.spearmanr(y2,p2).correlation

                # Calculate cross validated performance per predictor
                if return_var_per_group is not None:

                    # Loop over groups
                    for name in return_var_per_group.keys():

                        # Init a new weight vector
                        w_ = np.zeros_like(w)

                        # Set the weights of the included predictors
                        for w_name in return_var_per_group[name]:
                            if w_name in kernel_indices.keys():
                                w_[kernel_indices[w_name]] = w[kernel_indices[w_name]]

                        # Calculate cross validated model performance
                        p2 = np.dot( Xsubs2, np.expand_dims(w_,axis=1) )[:,0]
                        y2 = Ysubs2[:,n]

                        if "R2" in output_var:
                            if output_var == "R2m":
                                SStotal = np.sum( (y2-np.mean(y2))**2 )
                            elif output_var == "R2z":
                                SStotal = np.sum( y2**2 )
                            if SStotal > 0:
                                SSresidual = np.sum( (y2-p2)**2 )
                                group_return_var[name][r,n] = 1.0 - (SSresidual/SStotal)
                            else:
                                group_return_var[name][r,n] = np.NaN
                        elif output_var == "r":
                            group_return_var[name][r,n] = scipy.stats.stats.spearmanr(y2,p2).correlation

                # Show progress
                loop_count += 1.0
                if progress_indicator:
                    print((7*'\b')+'{:6.2f}%'.format(100.0*loop_count/n_loops), end='', flush=True)

        if progress_indicator:
            print( (7*'\b')+'{:6.2f}% .. done!'.format(100.0) )

        if return_var_per_group is not None:
            group_return_var["Full"] = return_var
            return_var = group_return_var

        return return_var


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # +++ Functions for evaluating the regression +++

    def predict(self, X):
        """ Calculates the model prediction for a given set of weights.
            X:       2d design matrix [trials x features]
            returns a 2d matrix [predictions x neurons]
        """
        prediction = np.zeros((self.n_frames,self.n_neurons))
        for n in range(self.n_neurons):
            if not self.constant:
                weights = []
            else:
                weights = [self._weights["Constant"][:,n],]
            for name in self._kernels.keys():
                weights.append( self._weights[name][:,n] )
            kernel_weights = np.expand_dims( np.concatenate(weights,axis=0), axis=1 )
            prediction[:,n] = np.dot(X,kernel_weights)[:,0]
        return prediction

    def R2(self, Y, P, withinrange=False):
        """ Calculates the R2 (R-squared), which is the standard for evaluating linear model fits. It compares the actual fit with a hypothetical fit using most simple possible model (just offset single value on the Y axis). The R2 value ranges between 0 and 1, but in cases of severely poor fitting R2 can also assume negative values. R2 can be interpreted as the amount of variance that the model accounts for relative to the total amount of variance that it could account for.
            Y:  2d matrix with single cell trials [trials x neurons]
            P:  2d predicted response matrix [predictions x neurons]
            withinrange:    Indicates whether only to calculate on y-values that are within range of the trial-wise predictors
            returns a 1d matrix containing the R2 [neurons,]
        """
        if withinrange:
            withinrange_ixs = self.withinrange_ixs
            Y = Y[withinrange_ixs,:]
            P = P[withinrange_ixs,:]
        SStotal = np.sum( (Y - np.tile(np.mean(Y,axis=0), (Y.shape[0],1) ) ) **2, axis=0)
        # SStotal = np.sum( Y**2, axis=0)
        SSresidual = np.sum((Y-P)**2,axis=0)
        R2 = 1.0 - (SSresidual/SStotal)
        return R2


    def mse(self, Y, P, withinrange=False):
        """ Calculates the mean squared error between the prediction and the trial-wise response matrix
            Y:  2d matrix with single cell trials [trials x neurons]
            P:  2d predicted response matrix [predictions x neurons]
            withinrange:    Indicates whether only to calculate on y-values that are within range of the trial-wise predictors
            returns a 1d matrix containing the mse [neurons,]
        """
        if withinrange:
            withinrange_ixs = self.withinrange_ixs
            Y = Y[withinrange_ixs,:]
            P = P[withinrange_ixs,:]
        MSE = np.sum((Y-P)**2,axis=0)/(Y.shape[0]-self.n_predictors)
        return MSE
