#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28, 2018

@author: pgoltstein
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get all imports
import numpy as np
import sys, os
sys.path.append('../xx_analysissupport')
import CAplot, CAgeneral, CArec

shuffled = '' # '' or '-shuffled'
mice = ['21a','21b','F02','F03','F04','K01','K02','K03','K06','K07']
datafolder = "../../data/p5_encodingmodel/regularization/"
figpath = "../../figureout/"
outlier_threshold = -0.0
include_threshold = 0.1
n_over_thresh = 4

# Load data
R2_mat = []
R2cv_mat = []

# Load byregion by ID
for ID in range(1,45):
    loc,mouse = CArec.AREAMOUSE_BY_ID[ID]
    filename = os.path.join(datafolder, 'data-regularization-'+loc+'-'+mouse+shuffled+'.npy')
    try:
        print("Loading data from: {}".format(filename))
        data_dict = np.load(filename, allow_pickle=True).item()
        CAplot.print_dict(data_dict, indent=1)
        R2_mat.append( data_dict["R2_mat"] )
        R2cv_mat.append( data_dict["R2cv_mat"] )
    except FileNotFoundError:
        print("Skipping; file not found: {}".format(filename))
loc = 'ALL'

R2_mat = np.concatenate(R2_mat,axis=0)
R2cv_mat = np.concatenate(R2cv_mat,axis=1)
L1s = data_dict["L1s"]
n_L1s = len(L1s)
n_trials = R2cv_mat.shape[0]

# Remove outliers
print("\nR2cv_mat: Removing {} negative ourliers (values below {})".format(np.sum(R2cv_mat<outlier_threshold),outlier_threshold))
R2cv_mat[R2cv_mat<outlier_threshold] = outlier_threshold
print("R2_mat: Removing {} negative ourliers (values below {})".format(np.sum(R2_mat<outlier_threshold),outlier_threshold))
R2_mat[R2_mat<outlier_threshold] = outlier_threshold

# Get neuron selector
included_neurons = np.sum(R2cv_mat[:,:,0]>include_threshold,axis=0) >= n_over_thresh

# Average over repeats
R2cv_mat = np.nanmean(R2cv_mat,axis=0)

# Get mean & sem over neurons
mean,sem,_ = CAgeneral.mean_sem(R2_mat[included_neurons,:],axis=0)
cvmean,cvsem,_ = CAgeneral.mean_sem(R2cv_mat[included_neurons,:],axis=0)

# Display R2 as function of L1
fig,ax = CAplot.init_figure_axes(fig_size=(10,10))
CAplot.line( np.arange(n_L1s), mean, e=sem, line_color='#008800', line_width=1, sem_color='#008800', shaded=True, label="Full model" )
CAplot.line( np.arange(n_L1s), cvmean, e=cvsem, line_color='#004488', line_width=1, sem_color='#004488', shaded=True, label="Cross-validated" )
CAplot.finish_panel( CAplot.plt.gca(), title="Model performance vs regularization\n({} neurons with R2>{} in {}/{} repeats".format(loc,include_threshold,n_over_thresh,n_trials), ylabel="R2", xlabel="L1", x_minmax=None, x_margin=0.0, x_axis_margin=0.0, y_minmax=[0,0.5], y_margin=0.01, y_axis_margin=0.0, x_ticks=np.arange(L1s.shape[0])[1::3], x_ticklabels=L1s[1::3], despine=True, x_tick_rotation=45, legend="on" )
CAplot.finish_figure( filename="5ED7a-Encodingmodel-Full-Regularization", path=figpath, wspace=0.5, hspace=0.5 )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
