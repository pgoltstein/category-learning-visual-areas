#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is meant to calculate the fraction of significantly responsive neurons per recording. It repeatedly subsamples N (e.g. 8) trials using bootstrapped resampling without replacement. The resulting analysis outcome is only useful at the recording-session level, because of the resampling, it does not allow identification of 'significantly responsive neurons'.

Created on Sun, Oct 25, 2018

@author: pgoltstein
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get imports
import numpy as np
import glob, os
import warnings
import argparse
import scipy.stats as scistats
import sys
sys.path.append('../xx_analysissupport')

# Add local imports
import CArec, CAgeneral


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments
parser = argparse.ArgumentParser( description = "Calculates fraction of responsive cells in all sessions for all mice that are included for an imaging region, independent of number of trials. \n (written by Pieter Goltstein - September 2018)")
parser.add_argument('imagingregion', type=str, help= 'Name of the imaging region to process')
parser.add_argument('-sd', '--shuffledata',  action="store_true", default=False, help='Flag enables data shuffling to estimate chance level')
parser.add_argument('-rd', '--resampledata',  action="store_true", default=False, help='Flag enables all frames to be resampled from baseline frames to estimate chance level')
args = parser.parse_args()

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
warnings.filterwarnings('ignore')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings
settings = {}
settings["imaging_region"] = str(args.imagingregion)
settings["location_dir"] = os.path.join( "../../data/chronicrecordings", settings["imaging_region"] )
settings["analyzed_data_dir"] = "../../data/p4_fractionresponsiveneurons"

settings["n_missing_allowed"] = 1
settings["data_type"] = "spike"
settings["stimulus_type"] = "Category"
settings["category_type"] = "Trained"

settings["bs_lock"] = "vis_on"
settings["bs_range"] = [-16,-1] # e.g. [-16,-1] = 1.0s before stimulus onset

settings["test_lock"] = "vis_on"
settings["test_range"] = [1,16] # Use e.g. [1,16] to select frame 1-15

settings["n_subsampl"] = 100 # Sub sample repeats
settings["subsampl_trials"] = 8 # 8 repetitions per stimulus

settings["alpha"] = 0.05
settings["min_peak_resp_ampl"] = 0.01


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Select recordings
settings["include_recordings"] = [  ("Baseline Out-of-Task",0), ("Baseline Out-of-Task",1), ("Baseline Task",0), ("Baseline Task",1), ("Learned " + settings["stimulus_type"] + " Task",0), ("Learned Out-of-Task",0)   ]

# Get mouse-recording-sets that have the required recordings
mouse_rec_sets,n_mice,n_recs = CArec.get_mouse_recording_sets( os.path.join(settings["location_dir"],"*"), settings["include_recordings"], settings["n_missing_allowed"] )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize data containers
if settings["stimulus_type"].lower() in "prototype":
    n_learned_stimuli = 2
elif settings["stimulus_type"].lower() in "category":
    n_learned_stimuli = 10
fr_resampled = np.full((CArec.n_mice,n_recs,n_learned_stimuli), np.NaN)
fr_lrpn_resampled = np.full((CArec.n_mice,n_recs,4), np.NaN)
catdiff_resampled_overall = np.full((CArec.n_mice,n_recs), np.NaN)
catdiff_resampled = np.full((CArec.n_mice,n_recs,n_learned_stimuli), np.NaN)
fr_resampled_per_stim = np.full((CArec.n_mice,n_recs,n_learned_stimuli), np.NaN)
behavioral_performance = np.full((CArec.n_mice,n_recs), np.NaN)
side_bias = np.full((CArec.n_mice,n_recs), np.NaN)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop mice
for m_nr,m_dir in enumerate(mouse_rec_sets):

    # Load chronic imaging recording from directory
    print("\nLoading: {}".format(m_dir))
    crec = CArec.chronicrecording( m_dir )

    # Select recordings to include
    recs = []
    for name,nr in settings["include_recordings"]:
        if len(crec.recs[name]) > nr:
            recs.append(crec.recs[name][nr])
        else:
            recs.append(None)

    # Get group numbers that are in all recordings
    group_nrs = CArec.complete_groups( recs )
    n_groups = len(group_nrs)

    # Get learned stimuli
    if settings["stimulus_type"].lower() in "prototype":
        learned_stimuli = crec.prototype_stimuli
    elif settings["stimulus_type"].lower() in "category":
        if settings["category_type"].lower() in "trained":
            learned_stimuli = crec.category_stimuli
        elif settings["category_type"].lower() in "inferred":
            learned_stimuli = crec.inferred_category_stimuli
        elif settings["category_type"].lower() in "shuffled":
            learned_stimuli = crec.shuffled_category_stimuli

    # Shuffle data if requested
    for rec in recs:
        if rec is not None:
            if args.shuffledata:
                rec.shuffle = "shuffle"
            if args.resampledata:
                rec.shuffle = "resample_from_baseline"

    # Loop recordings
    print("Gathering tuning data:")
    for r_nr,rec in enumerate(recs):
        if rec is None:
            print("{}) Not present: {} #{}".format( r_nr, settings["include_recordings"][r_nr][0], settings["include_recordings"][r_nr][1] ))
            continue

        print("{}) {} (mouse={}, no={}, date={})".format( \
            r_nr, rec.timepoint, rec.mouse, rec.mouse_no, rec.date ))

        # Add behavioral performance
        behavioral_performance[rec.mouse_no,r_nr] = np.nanmean(rec.outcome)
        print("   Behavioral performance: {}".format(behavioral_performance[rec.mouse_no,r_nr]))
        category_ids = rec.get_trial_category_id(learned_stimuli)
        side_bias[rec.mouse_no,r_nr] = np.nanmean(rec.outcome[category_ids==1]) - np.nanmean(rec.outcome[category_ids==2])
        print("   Side bias: {}".format(side_bias[rec.mouse_no,r_nr]))

        # Set neuron list to include only complete groups
        rec.neuron_groups = group_nrs

        # Get data matrix and stimulus ids
        data_mat = rec.spikes if 'spike' in settings["data_type"] else rec.dror
        category_ids = rec.get_trial_category_id(learned_stimuli)
        stimulus_ids = rec.stimuli
        directions = rec.direction
        spatialfs = rec.spatialf
        boundary_distances = rec.get_trial_boundary_distance(learned_stimuli)

        sorted_unique_stim_ids = np.zeros(n_learned_stimuli)
        sorted_unique_directions = np.zeros(n_learned_stimuli)
        sorted_unique_spatialfs = np.zeros(n_learned_stimuli)
        s_ix = 0
        # print("\nLeft:")
        for b in np.unique(boundary_distances[category_ids==1])[::-1]:
            stim_ids = stimulus_ids[ np.logical_and(boundary_distances==b,category_ids==1)]
            dir_ids = directions[ np.logical_and(boundary_distances==b,category_ids==1)]
            spf_ids = spatialfs[ np.logical_and(boundary_distances==b,category_ids==1)]
            # print( " - b={}, stims= {}".format(b, stim_ids ) )
            sorted_unique_stim_ids[s_ix] = stim_ids[0]
            sorted_unique_directions[s_ix] = dir_ids[0]
            sorted_unique_spatialfs[s_ix] = spf_ids[0]
            s_ix += 1
        category_break = int(s_ix)
        n_left_stimuli = int(category_break)
        n_right_stimuli = int(n_learned_stimuli-category_break)

        # print("\nRight:")
        for b in np.unique(boundary_distances[category_ids==2]):
            stim_ids = stimulus_ids[ np.logical_and(boundary_distances==b,category_ids==2)]
            dir_ids = directions[ np.logical_and(boundary_distances==b,category_ids==2)]
            spf_ids = spatialfs[ np.logical_and(boundary_distances==b,category_ids==2)]
            # print( " - b={}, stims= {}".format(b, stim_ids ) )
            sorted_unique_stim_ids[s_ix] = stim_ids[0]
            sorted_unique_directions[s_ix] = dir_ids[0]
            sorted_unique_spatialfs[s_ix] = spf_ids[0]
            s_ix += 1

        print("   Sorted unique stim ids:")
        print("           Nr: " + ("{:5.0f}"*n_learned_stimuli).format( *sorted_unique_stim_ids) )
        print("          Dir: " + ("{:5.0f}"*n_learned_stimuli).format( *sorted_unique_directions) )
        print("           Sf: " + ("{:5.2f}"*n_learned_stimuli).format( *sorted_unique_spatialfs) )
        print("    Cat-break: {}".format( category_break ) )


        # Get response onsets
        if settings["test_lock"] == "vis_on":
            on_frames_test = rec.vis_on_frames
        elif settings["test_lock"] == "vis_off":
            on_frames_test = rec.vis_off_frames
        elif settings["test_lock"] == "resp_on":
            on_frames_test = rec.response_win_on_frames
        if settings["bs_lock"] == "vis_on":
            on_frames_bs = rec.vis_on_frames

        # Calculate trial-wise tuning matrix
        tm = CAgeneral.tm( data_mat, on_frames_test,  settings["test_range"], return_peak=False, include_bs=False )
        tm_peak = CAgeneral.tm( data_mat, on_frames_test,  settings["test_range"], return_peak=True, include_bs=False )
        bs = CAgeneral.tm( data_mat, on_frames_bs,  settings["bs_range"], return_peak=False, include_bs=False )

        # Loop for each neuron and stimulus
        n_trials,n_neurons = tm.shape
        n_sign_neuron = np.zeros((n_neurons,n_learned_stimuli))
        catdiff_neuron_overall = np.full(n_neurons,np.NaN)
        n_lrpn_neuron = np.full((n_neurons,4),np.NaN)
        catdiff_neuron = np.full((n_neurons,n_learned_stimuli),np.NaN)
        n_sign_per_stim = np.zeros((n_neurons,n_learned_stimuli))
        for n_ix in range(n_neurons):

            # Loop stimuli and subsamples
            sign_stim = np.zeros((settings["n_subsampl"],n_learned_stimuli))
            for s_ix,s in enumerate(sorted_unique_stim_ids):

                # Get data
                stimulus_ixs = stimulus_ids==s
                data1 = tm[stimulus_ixs,n_ix]
                data2 = bs[stimulus_ixs,n_ix]
                data3 = tm_peak[stimulus_ixs,n_ix]
                n_stim_trials = np.sum(stimulus_ixs)

                # Loop subsample iterations
                for p_ix in range(settings["n_subsampl"]):

                    # Get subsample of data for this neuron & stimulus
                    subs_ix = np.random.choice(n_stim_trials, size=settings["subsampl_trials"], replace=False)
                    (_,p) = scistats.ranksums(data1[subs_ix], data2[subs_ix])
                    sign_stim[p_ix,s_ix] = (p < settings["alpha"]) and (np.mean( data3[subs_ix] - data2[subs_ix] ) > settings["min_peak_resp_ampl"])

            # Get significant n-stim fractions
            for s_ix in range(n_learned_stimuli):
                n_sign_stim = np.sum(sign_stim,axis=1) == s_ix+1
                n_sign_neuron[n_ix,s_ix] = np.sum(n_sign_stim) / settings["n_subsampl"]

            # Get fraction of responsive subsamples per stimulus
            n_sign_per_stim[n_ix,:] = np.sum(sign_stim,axis=0) / settings["n_subsampl"]

            # if there are any significant responsive stimulus-subsamples
            any_sign_stim = np.sum(sign_stim,axis=1) > 0
            if np.sum(any_sign_stim) > 0:

                # Calculate catdiff on responsiveness per subsample
                catdiff = np.zeros(settings["n_subsampl"])
                for p_ix in range(settings["n_subsampl"]):
                    catdiff[p_ix] = np.abs( (np.sum(sign_stim[p_ix,:category_break]) / n_left_stimuli) - (np.sum(sign_stim[p_ix,category_break:]) / n_right_stimuli) )
                catdiff_neuron_overall[n_ix] = np.nanmean(catdiff[any_sign_stim])

                # Divide cat-diff per n-stim group
                for s_ix in range(n_learned_stimuli):
                    n_sign_stim = np.sum(sign_stim,axis=1) == s_ix+1
                    catdiff_neuron[n_ix,s_ix] = np.nanmean(catdiff[n_sign_stim])

                # Calculate fraction of right & left responsive stimuli
                n_lrpn_neuron[n_ix,0] = np.mean( np.sum( sign_stim[any_sign_stim,:category_break], axis=1 ) / n_left_stimuli )
                n_lrpn_neuron[n_ix,1] = np.mean( np.sum( sign_stim[any_sign_stim,category_break:], axis=1 ) / n_right_stimuli )

                # Calculate fraction of pref and non-pref responsive stimuli
                if n_lrpn_neuron[n_ix,0] > n_lrpn_neuron[n_ix,1]:
                    # Left is preferred
                    n_lrpn_neuron[n_ix,2] = n_lrpn_neuron[n_ix,0]
                    n_lrpn_neuron[n_ix,3] = n_lrpn_neuron[n_ix,1]
                else:
                    # Right is preferred
                    n_lrpn_neuron[n_ix,2] = n_lrpn_neuron[n_ix,1]
                    n_lrpn_neuron[n_ix,3] = n_lrpn_neuron[n_ix,0]

        # Store in data matrix
        fr_resampled[rec.mouse_no,r_nr,:] = np.sum(n_sign_neuron,axis=0) / n_groups
        fr_lrpn_resampled[rec.mouse_no,r_nr,:] = np.nanmean(n_lrpn_neuron,axis=0)
        catdiff_resampled[rec.mouse_no,r_nr,:] = np.nanmean(catdiff_neuron,axis=0)
        catdiff_resampled_overall[rec.mouse_no,r_nr] = np.nanmean(catdiff_neuron_overall)
        fr_resampled_per_stim[rec.mouse_no,r_nr,:] = np.sum(n_sign_per_stim,axis=0) / n_groups

        # Display fraction responsive cells
        print("     Frc/stim: " + ("{:5.2f}"*n_learned_stimuli).format( *(fr_resampled_per_stim[rec.mouse_no,r_nr,:])) )
        print("     Frc n-st: " + ("{:5.2f}"*n_learned_stimuli).format( *(fr_resampled[rec.mouse_no,r_nr,:])) )
        print("    DIFF n-st: " + ("{:5.2f}"*n_learned_stimuli).format( *(catdiff_resampled[rec.mouse_no,r_nr,:])) )
        print("      L-R-P-N: " + ("{:5.2f}"*4).format( *(fr_lrpn_resampled[rec.mouse_no,r_nr,:])) )
        print("         DIFF: {:5.2f}".format( catdiff_resampled_overall[rec.mouse_no,r_nr]) )

data = { "fr_resampled": fr_resampled, "fr_resampled_per_stim": fr_resampled_per_stim, "catdiff_resampled": catdiff_resampled, "catdiff_resampled_overall": catdiff_resampled_overall, "fr_lrpn_resampled": fr_lrpn_resampled, "behavioral_performance": behavioral_performance, "side_bias": side_bias, "category_break": category_break, "n_learned_stimuli": n_learned_stimuli }
data_dict = { "data": data, "settings": settings }

save_filename = os.path.join( settings["analyzed_data_dir"], 'data_frc_'+settings["imaging_region"] )
np.save( save_filename, data_dict )
print("Saved data in: {}".format(save_filename))
