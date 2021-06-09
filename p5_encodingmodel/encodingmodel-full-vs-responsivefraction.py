#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30, 2018

@author: pgoltstein
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get all imports
import os, glob
import numpy as np
import scipy.stats as scistats
import warnings
import sys
sys.path.append('../xx_analysissupport')
import CAplot, CAgeneral, CAencodingmodel, CArec
import CAanalysissupport as CAsupp

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
warnings.filterwarnings('ignore')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings
CAplot.font_size["title"] = 10
CAplot.font_size["label"] = 10
CAplot.font_size["tick"] = 10
CAplot.font_size["text"] = 10
CAplot.font_size["legend"] = 10

figpath = "../../figureout/"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find imaging location files
locations = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
n_locs = len(locations)
n_mice = 10
n_timepoints = 2
data_dir = "../../data/p4_fractionresponsiveneurons"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load and process recordings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_mat_bs = np.zeros((n_locs*n_mice,n_timepoints), dtype=float)
data_mat_cat = np.zeros((n_locs*n_mice,n_timepoints), dtype=float)

# Get enconding model data
model_data = CAsupp.model_load_data("cat-trained", R2="m")
model_data_sh = CAsupp.model_load_data("shuffled-trials", R2="m")
R2 = model_data["R2"]
R2_sh = model_data_sh["R2"]
mouse_id = model_data["mouse_id"]

# Loop locations
for l_nr,loc in enumerate(locations):

    # Load fraction responsive data from file
    filename = os.path.join(data_dir, 'data_frc_'+loc+'.npy')
    print("Loading data from: {}".format(filename))
    data_dict = np.load(filename, allow_pickle=True).item()
    resp_frac = np.sum(data_dict["data"]["fr_resampled"],axis=2)

    # Get significantly predictive neurons in encoding model
    signtuned = CAsupp.model_get_sign_neurons(R2[loc], R2_sh[loc], which="individual", n_timepoints=2)

    # Get fraction of tuned neurons
    F_per_mouse = CAsupp.model_get_fraction(signtuned, mouse_id[loc])

    # Collect data
    data_mat_bs[l_nr*n_mice:l_nr*n_mice+n_mice,0] = resp_frac[:,2]
    data_mat_cat[l_nr*n_mice:l_nr*n_mice+n_mice,0] = resp_frac[:,4]

    data_mat_bs[l_nr*n_mice:l_nr*n_mice+n_mice,1] = F_per_mouse[:,0]
    data_mat_cat[l_nr*n_mice:l_nr*n_mice+n_mice,1] = F_per_mouse[:,1]

notnanix = ~np.isnan(data_mat_bs.sum(axis=1))
data_mat_bs = data_mat_bs[notnanix,:]

notnanix = ~np.isnan(data_mat_cat.sum(axis=1))
data_mat_cat = data_mat_cat[notnanix,:]

data_mat = np.concatenate([data_mat_bs,data_mat_cat],axis=0)

print("\nCorrelation between fraction responsive and encoding model")
r,p = scistats.pearsonr(data_mat[:,0].ravel(),data_mat[:,1].ravel())
n = data_mat.shape[0]
print("All: r={:6.4f}, p={:E}, n={:2.0f} sessions".format(r,p,n))
print(" ")


# Plot correlation baseline and after learning alltogether
fig = CAplot.init_figure(fig_size=(6,6))
ax = CAplot.plt.subplot2grid( (1,1), (0,0) )

# Scatter plot
x_data = data_mat[:,0].ravel()
y_data = data_mat[:,1].ravel()
for nr,(x,y) in enumerate(zip(x_data,y_data)):
    CAplot.plt.plot( x, y, color="None", markerfacecolor=(0,0,0,1), markersize=3, markeredgewidth=0, marker="o", markeredgecolor=None )

# Finish panel layout
CAplot.finish_panel( ax, title=None, ylabel="Encoding model", xlabel="Fraction responsive", legend="off", y_minmax=[0,1.0], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1.0], x_step=[0.2,1], x_margin=0.02, x_axis_margin=0.01, despine=True)

# Finish figure layout and save
CAplot.finish_figure( filename="5ED7d-Encodingmodel-Full-Vs-ResponsiveFraction", path=figpath, wspace=0.8, hspace=0.8 )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
