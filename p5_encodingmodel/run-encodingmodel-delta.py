#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script runs an encodingmodel and stores the R2.

Created on Wed, Nov 4, 2020

@author: pgoltstein
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get imports
import numpy as np
import collections
import sys, glob, os, time
import warnings
import argparse
sys.path.append('../xx_analysissupport')

# Add local imports
import CArec, CAgeneral, CAencodingmodel, CAplot


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments
parser = argparse.ArgumentParser( description = "Runs an encoding model. \n (written by Pieter Goltstein - Nov 2020)")
parser.add_argument('imagingregion', type=str, help= 'Name of the imaging region to process')
parser.add_argument('mousename', type=str, help= 'Name of the mouse to process')
parser.add_argument('model', type=str, help= 'Name of the model to fit; "cat" for category-only, "ori" for orientation-spatialf-only, "stim" for stimulus-only, "comb" for category-orienation-spatialf-combined, ')
parser.add_argument('-s', '--shuffle',  type=str, help='Shuffles "trials", all "frames" or "Y" data')
parser.add_argument('-g', '--shufflegroup',  type=str, help='Shuffles all included frames in the design matrix for the group specified (supply "shuffle-no-groups" to get full non-shuffled model)', default=None)
parser.add_argument('-a', '--shuffleallbutgroup',  type=str, help='Shuffles all included frames in the design matrix for all groups but the one specified (supply "shuffle-all-groups" to get fully shuffled model)', default=None)
parser.add_argument('-o', '--outputvariable',  type=str, help='Return "R2m" values (SSE based, compared to mean), "R2z" values (SSE based, compared to zero) or spearman "r" values (default=R2m)', default="R2m")
parser.add_argument('-n', '--nrepeats',  type=int, help='Number of cross-validations (default=100)', default=100)
parser.add_argument('-r', '--range',  type=str, help='Restricts model to range "trials", or fits "full" range (default=trials)', default="trials")
parser.add_argument('-l', '--learnedstimulustype',  type=str, help='Defined the type of learned stimulus/stimuli "Prototype", "Category" (default=Category)', default="Category")
parser.add_argument('-c', '--categorytype',  type=str, help='Defines the grouping of stimuli into categories "Trained", "Inferred", "Shuffled", "Swapped" (default=Trained)', default="Trained")
parser.add_argument('-k', '--kerneltype',  type=str, help='Defines the type of kernel used; "flex" for flexible kernels, "fix" for fixed kernels (default=flex)', default="flex")
parser.add_argument('-v', '--verbose',  action="store_true", default=False, help='Flag enables the display of progress')

args = parser.parse_args()

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings, which are due to columns/rows full with NaN's in numpy matrices
warnings.filterwarnings('ignore')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings
settings = {}
if str(args.imagingregion).lower() == "id":
    # Recode region by ID
    ID = int(args.mousename)
    settings["imaging_region"],settings["mouse"] = CArec.AREAMOUSE_BY_ID[ID]
else:
    settings["imaging_region"] = str(args.imagingregion)
    settings["mouse"] = str(args.mousename)

settings["location_dir"] = "../../data/chronicrecordings/{}/{}".format(settings["imaging_region"],settings["mouse"])
settings["analyzed_data_dir"] = "../../data/p5_encodingmodel/delta_model/"

settings["shuffle frames"] = True if args.shuffle == "frames" else False
settings["shuffle group"] = str(args.shufflegroup) if args.shufflegroup is not None else None
settings["shuffle all but group"] = str(args.shuffleallbutgroup) if args.shuffleallbutgroup is not None else None
settings["shuffle trials"] = True if args.shuffle == "trials" else False
settings["shuffle Y"] = True if args.shuffle == "Y" else False
settings["output variable"] = str(args.outputvariable)

settings["n_missing_allowed"] = 1
settings["data_type"] = "spike"
settings["learned stimulus type"] = str(args.learnedstimulustype)
settings["category type"] = str(args.categorytype)
settings["kernel type"] = str(args.kerneltype).lower()

settings["datasmooth"] = 0.5              # Smoothing of the spike trace
settings["kernelstep"] = 0.5               # Step size within kernel
settings["kernelrange"] = (-0.6,2.6)       # Full span of the kernel
settings["kernelunit"] = "seconds"         # Unit definition for kernel
settings["L1"] = 0.1                       # Regularization parameter
settings["n_repeats"] = int(args.nrepeats) # Cross-validation repeats

if args.model == "cat":
    settings["model"] = "Category"
elif args.model == "ori":
    settings["model"] = "Orientation-spatialf"
elif args.model == "stim":
    settings["model"] = "Stimulus"
elif args.model == "comb":
    settings["model"] = "Combined"

settings["nonnegative"] = True
settings["withinrange"] = True if args.range == "trials" else False


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Select recordings
settings["include_recordings"] = [ ("Baseline Task",0), ("Baseline Task",1), ("Learned " + settings["learned stimulus type"] + " Task",0) ]

# Get mouse-recording-sets that have the required recordings
mouse_rec_sets,n_mice,n_recs = CArec.get_mouse_recording_sets( settings["location_dir"], settings["include_recordings"], settings["n_missing_allowed"] )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize data containers
performance = [None for _ in range(n_recs)]
weight      = [None for _ in range(n_recs)]
kframes     = [None for _ in range(n_recs)]
behavioral_performance = np.full((n_recs,), np.NaN)
side_bias = np.full((n_recs,), np.NaN)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load chronic imaging recording from directory
if len(mouse_rec_sets) == 0:
    print("\nLooked for {}\nNo recording found, quitting\n".format(settings["location_dir"]))
    quit()

print("\nLoading: {}".format(mouse_rec_sets[0]), flush=True)
crec = CArec.chronicrecording( mouse_rec_sets[0] )

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
settings["shuffle categories"] = False
settings["swap categories"] = False
if settings["learned stimulus type"].lower() in "prototype":
    learned_stimuli = crec.prototype_stimuli
elif settings["learned stimulus type"].lower() in "category":
    if settings["category type"].lower() in "trained":
        learned_stimuli = crec.category_stimuli
    elif settings["category type"].lower() in "inferred":
        learned_stimuli = crec.inferred_category_stimuli
    elif settings["category type"].lower() in "shuffled":
        settings["shuffle categories"] = True
        learned_stimuli = crec.category_stimuli
    elif settings["category type"].lower() in "swapped":
        settings["swap categories"] = True
        learned_stimuli = crec.category_stimuli

# Shuffle data if requested
for rec in recs:
    if rec is not None:
        if settings["shuffle frames"]:
            rec.shuffle = "shuffle"

# Loop recordings
print("Looping over recordings:")
for r_nr,rec in enumerate(recs):
    if rec is None:
        print("\n{}) Not present: {} #{}".format( r_nr, settings["include_recordings"][r_nr][0], settings["include_recordings"][r_nr][1] ))
        continue

    print("\n{}) {} (mouse={}, no={}, date={})".format( \
        r_nr, rec.timepoint, rec.mouse, rec.mouse_no, rec.date ))

    # Add behavioral performance
    behavioral_performance[r_nr] = np.nanmean(rec.outcome)
    print("   Behavioral performance: {}".format(behavioral_performance[r_nr]))
    category_ids = rec.get_trial_category_id(learned_stimuli)
    side_bias[r_nr] = np.nanmean(rec.outcome[category_ids==1]) - np.nanmean(rec.outcome[category_ids==2])
    print("   Side bias: {}".format(side_bias[r_nr]))

    # Set neuron list to include only complete groups
    rec.neuron_groups = group_nrs

    # Get predictor offsets
    offsets = CAencodingmodel.get_offsets(rec)

    # Set up model
    print("\n   ### Initializing {} model ###".format(settings["model"]))
    if settings["kernel type"] == "fix":
        model = CAencodingmodel.init_full_model_fixed_kernels(rec, learned_stimuli, stimulus=settings["model"], step=settings["kernelstep"], principle_range=settings["kernelrange"], unit=settings["kernelunit"], data_smooth=settings["datasmooth"])
    elif settings["kernel type"] == "flex":
        model = CAencodingmodel.init_full_model_flexible_kernels(rec, learned_stimuli, stimulus=settings["model"], step=settings["kernelstep"], principle_range=settings["kernelrange"], unit=settings["kernelunit"], data_smooth=settings["datasmooth"])
    print("    * Smoothing data: {} s".format(settings["datasmooth"]))
    print("    * Non-negative least squares: {}".format(settings["nonnegative"]))
    print("    * Fitting data within-range: {}".format(settings["withinrange"]))
    print("    * Shuffling trials: {}".format(settings["shuffle trials"]))
    print("    * Shuffling group: {}".format(settings["shuffle group"]))
    print("    * Shuffling all but group: {}".format(settings["shuffle all but group"]))
    print("    * Shuffling Y-data: {}".format(settings["shuffle Y"]))
    print("    * Learned stimulus type: {}".format(settings["learned stimulus type"]))
    print("    * Category type: {}".format(settings["category type"]))
    print("    * Kernel design: {}".format(settings["kernel type"]))
    print("    * Output variable: {}".format(settings["output variable"]))
    # CAplot.print_dict(model.kernels)

    # Add spike rate data
    model.Y = rec.spikes if 'spike' in settings["data_type"] else rec.dror

    # Perform cross validated regression on category model
    print("    Running {}x cross-validated regression on category model (L1={})...".format(settings["n_repeats"],settings["L1"]), flush=True)
    start_time = time.time()
    performance[r_nr] = model.crossvalidated_regression( repeats=settings["n_repeats"], L1=settings["L1"], nonnegative=settings["nonnegative"], withinrange=settings["withinrange"], shuffle=settings["shuffle Y"], shuffle_trials=settings["shuffle trials"], shuffle_group=settings["shuffle group"], shuffle_all_but_group=settings["shuffle all but group"], shuffle_categories=settings["shuffle categories"], swap_categories=settings["swap categories"], return_var_per_group=CAencodingmodel.PREDICTOR_GROUPS, output_var=settings["output variable"], rec=rec, crec=crec, progress_indicator=args.verbose, shuffle_group_dict=CAencodingmodel.PREDICTOR_GROUPS)
    print("     -> running time: {:0.2f} s".format(time.time()-start_time))
    print("    Maximum R2: {:0.3f}".format( np.nanmax(np.nanmean(performance[r_nr]["Full"],axis=0)) ), flush=True)

    # Exctract weights
    weight[r_nr] = model.weights

    # Get the kernel frames
    kframes[r_nr] = collections.OrderedDict()
    for name,kernel in model.kernels.items():
        kframes[r_nr][name] = kernel.frames

data = { settings["output variable"]: performance, "w": weight, "kfr": kframes, "grp": group_nrs, "behavioral_performance": behavioral_performance, "side_bias": side_bias }
data_dict = { "data": data, "settings": settings }

# Create proper filename that IDs model settings
save_filename = os.path.join( settings["analyzed_data_dir"], 'data-wi' )
save_filename += "-" + settings["kernel type"] + "K"
save_filename += "-" + str(args.model).lower() + "encmdl"
save_filename += '-' + settings["imaging_region"]
save_filename += '-' + settings["mouse"]
save_filename += "-cat" if settings["learned stimulus type"].lower() == "category" else "-prot"
save_filename += "-"+settings["category type"].lower()
save_filename += "-trialrange" if settings["withinrange"] == True else "-fullrange"
save_filename += "-"+settings["output variable"]
save_filename += "-shfr" if args.shuffle == "frames" else ""
save_filename += "-shgr-"+args.shufflegroup if args.shufflegroup is not None else ""
save_filename += "-shabgr-"+args.shuffleallbutgroup if args.shuffleallbutgroup is not None else ""
save_filename += "-shtr" if args.shuffle == "trials" else ""
save_filename += "-shY" if args.shuffle == "Y" else ""

np.save( save_filename, data_dict )
print("Saved data in: {}".format(save_filename), flush=True)
