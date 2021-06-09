#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28, 2018

@author: pgoltstein
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get all imports
import sys, os
import time
import warnings
import numpy as np
import pandas as pd
sys.path.append('../xx_analysissupport')

import CArec, CAplot, CAencodingmodel, CAgeneral
import argparse

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments
parser = argparse.ArgumentParser( description = "Estimates the regularization factor for an encoding model \n (written by Pieter Goltstein - September 2018)")
parser.add_argument('imagingregion', type=str, help= 'Name of the imaging region to process')
parser.add_argument('mousename', type=str, help= 'Name of the mouse to process')
parser.add_argument('-d', '--display',  action="store_true", default=False, help='Flag enables output plot')
parser.add_argument('-sd', '--shuffledata',  action="store_true", default=False, help='Flag enables data shuffling to estimate chance level')
args = parser.parse_args()

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
warnings.filterwarnings('ignore')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Select recordings
include_recordings = [  ("Baseline Task",0),
                        ("Learned Category Task",0) ]

if str(args.imagingregion).lower() == "id":
    # Recode region by ID
    ID = int(args.mousename)
    area_mouse = CArec.AREAMOUSE_BY_ID[ID]
else:
    area_mouse = str(args.imagingregion),str(args.mousename)

mouse_rec_sets,n_mice,n_recs = CArec.get_mouse_recording_sets( "../../data/chronicrecordings/{}/{}".format(*area_mouse), include_recordings, 0 )
outputpath = "../../data/p5_encodingmodel/regularization/"
indicate_progress = True

L1s = []
for i in range(1,4):
    L1s.extend([5/(10.0**i),2/(10.0**i),1/(10.0**i)])
L1s = [1.0,] + L1s + [0.0]
L1s = np.array(L1s[::-1])
L1s = L1s[:3]
print(L1s)
n_repeats = 2
n_L1s = len(L1s)

# Load imaging recording from directory
print("\nLoading: {}".format(mouse_rec_sets[0]))
crec = CArec.chronicrecording(mouse_rec_sets[0])
recs = [ crec.recs[include_recordings[0][0]][0], crec.recs[include_recordings[1][0]][0] ]
learned_stimuli = crec.category_stimuli
groups_ids = CArec.complete_groups( recs )

# select timepoint 'learned'
rec = recs[1]
rec.neuron_groups = groups_ids
n_neurons = len(rec.neuron_groups)
print("Selected groups: {}".format(rec.neuron_groups))

if args.shuffledata:
    rec.shuffle = "shuffle"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize models
offsets = CAencodingmodel.get_offsets(rec)
model = CAencodingmodel.init_full_model_flexible_kernels(rec, learned_stimuli, stimulus="Combined", step=0.5, data_smooth=0.5, principle_range=(-0.6,2.6), unit="Seconds")
model.Y = rec.spikes

# Perform cross validated regression as function of L1
R2cv_mat = np.zeros((n_repeats,n_neurons,n_L1s))
R2_mat = np.zeros((n_neurons,n_L1s))
for nr,L1 in enumerate(L1s):
    print("Running regression on model (L1={})...".format(L1))
    start_time = time.time()
    R2_mat[:,nr] = model.regression(L1=L1, nonnegative=True, withinrange=True, shuffle=False)
    R2cv_mat[:,:,nr] = model.crossvalidated_regression( repeats=n_repeats, L1=L1, nonnegative=True, withinrange=True, shuffle=False, progress_indicator=indicate_progress)
    print(" -> running time: {:0.2f} s".format(time.time()-start_time))

data_dict = { "R2_mat": R2_mat, "R2cv_mat": R2cv_mat, "L1s": L1s }
save_filename = outputpath + 'data-regularization-' + area_mouse[0] + '-' +  area_mouse[1] + ("-shuffled" if args.shuffledata else "")
np.save( save_filename, data_dict )
print("Saved data in: {}".format(save_filename))

# Show output if requested
if args.display:

    # Average over repeats
    R2cv_mat = np.nanmean(R2cv_mat,axis=0)

    # Get mean & sem over neurons
    mean,sem,_ = CAgeneral.mean_sem(R2_mat,axis=0)
    cvmean,cvsem,_ = CAgeneral.mean_sem(R2cv_mat,axis=0)

    # Display R2 as function of L1
    fig,ax = CAplot.init_figure_axes(fig_size=(10,10))
    CAplot.line( np.arange(n_L1s), mean, e=sem, line_color='#008800', line_width=1, sem_color='#008800', shaded=True, label="Full model" )
    CAplot.line( np.arange(n_L1s), cvmean, e=cvsem, line_color='#004488', line_width=1, sem_color='#004488', shaded=True, label="Cross-validated" )
    CAplot.finish_panel( CAplot.plt.gca(), title="Model performance as function of regularization", ylabel="R2", xlabel="L1", x_minmax=None, x_margin=0.0, x_axis_margin=0.0, y_minmax=None, y_margin=0.0, y_axis_margin=0.0, x_ticks=np.arange(L1s.shape[0]), x_ticklabels=L1s, despine=False, x_tick_rotation=45 )
    CAplot.finish_figure( filename="" )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Show
    CAplot.plt.show()

# That's all folks!
