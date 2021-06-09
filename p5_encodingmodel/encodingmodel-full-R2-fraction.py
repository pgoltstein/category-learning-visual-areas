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
import collections
import sys
sys.path.append('../xx_analysissupport')

import CAplot, CAgeneral, CAencodingmodel, CArec, CAstats
import CAanalysissupport as CAsupp
import warnings

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings, which are due to columns/rows full with NaN's in numpy matrices
warnings.filterwarnings('ignore')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic settings
areas = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
mice = ['21a','21b','F02','F03','F04','K01','K02','K03','K06','K07']
n_areas = len(areas)
figpath = "../../figureout/"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load model data
model_data = CAsupp.model_load_data("cat-trained", R2="m")
model_data_sh = CAsupp.model_load_data("shuffled-trials", R2="m")
CAplot.print_dict(model_data["settings"])

R2 = model_data["R2"]
R2_sh = model_data_sh["R2"]
mouse_id = model_data["mouse_id"]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show fraction of neurons with R2 > shuffled
F_per_mouse = []
R2_per_mouse_LG = []
R2_per_neuron_LG = []
R2_per_mouse_ST = []
R2_per_neuron_ST = []
bhv_perf = collections.OrderedDict()

for area in areas:
    print("Processing area {}".format(area))

    # Get behavioral performance for this area
    bhv_perf[area] = CAgeneral.beh_per_mouse( model_data["bhv-perf"][area], mouse_no_dict=CArec.mouse_no, timepoint=2 )

    # Get significantly predictive neurons
    signtuned = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="individual", n_timepoints=2)

    # Get fraction of tuned neurons
    F_per_mouse += [CAsupp.model_get_fraction(signtuned, mouse_id[area]),]

    # Get significantly predictive neurons
    signtuned_lost = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="lost", n_timepoints=2)
    signtuned_stable = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="stable", n_timepoints=2)
    signtuned_gained = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="gained", n_timepoints=2)

    # Get R2 of tuned neurons
    R2lost = CAsupp.model_get_R2( R2[area]["Full"]["mean"], signtuned_lost, mouse_id[area], permouse=False, negatives=0.0 )
    R2stable = CAsupp.model_get_R2( R2[area]["Full"]["mean"], signtuned_stable, mouse_id[area], permouse=False, negatives=0.0 )
    R2gained = CAsupp.model_get_R2( R2[area]["Full"]["mean"], signtuned_gained, mouse_id[area], permouse=False, negatives=0.0 )
    R2_per_neuron_LG += [np.stack([R2lost[:,0],R2gained[:,1]], axis=1),]
    R2_per_neuron_ST += [np.stack([R2stable[:,0],R2stable[:,1]], axis=1),]

    # Get mean R2 per mouse
    R2lost = CAsupp.model_get_R2( R2[area]["Full"]["mean"], signtuned_lost, mouse_id[area], permouse=True, negatives=0.0 )
    R2stable = CAsupp.model_get_R2( R2[area]["Full"]["mean"], signtuned_stable, mouse_id[area], permouse=True, negatives=0.0 )
    R2gained = CAsupp.model_get_R2( R2[area]["Full"]["mean"], signtuned_gained, mouse_id[area], permouse=True, negatives=0.0 )
    R2_per_mouse_LG += [np.stack([R2lost[:,0],R2gained[:,1]], axis=1),]
    R2_per_mouse_ST += [np.stack([R2stable[:,0],R2stable[:,1]], axis=1),]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display fraction and R2 of significant neurons overall

dfF_per_mouse = pd.DataFrame(np.concatenate(F_per_mouse,axis=0), columns=["B","L"])
dfR2_per_mouse_LG = pd.DataFrame(np.concatenate(R2_per_mouse_LG,axis=0), columns=["Lost","Gain"])
dfR2_per_neuron_LG = pd.DataFrame(np.concatenate(R2_per_neuron_LG,axis=0), columns=["Lost","Gain"])
dfR2_per_mouse_ST = pd.DataFrame(np.concatenate(R2_per_mouse_ST,axis=0), columns=["S-B","S-L"])
dfR2_per_neuron_ST = pd.DataFrame(np.concatenate(R2_per_neuron_ST,axis=0), columns=["S-B","S-L"])

fig = CAplot.init_figure(fig_size=(12,4))
ax = CAplot.plt.subplot2grid((1,3),(0,0))
CAplot.sns.swarmplot(data=dfF_per_mouse, ax=ax, color="k", size=3, linewidth=1, edgecolor="None")
c1,c2 = ax.collections[0],ax.collections[1]
for (x1,y1),(x2,y2) in zip(c1.get_offsets(),c2.get_offsets()):
    CAplot.plt.plot( [x1,x2], [y1,y2], color="#d0d0d0", marker=None )
CAplot.finish_panel( ax=ax, ylabel="Fraction", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55, x_ticks=[0,1], x_ticklabels=["B","L"], x_tick_rotation=0, tick_size=None, label_size=None, title_size=None, legend_size=None, despine=True, legendpos=0)

ax = CAplot.plt.subplot2grid((1,3),(0,1))
CAplot.sns.violinplot(data=dfR2_per_neuron_ST, ax=ax, inner=None, color="#b0b0b0")
CAplot.sns.swarmplot(data=dfR2_per_mouse_ST, ax=ax, color="k", size=3, linewidth=1, edgecolor="None")
c1,c2 = ax.collections[2],ax.collections[3]
for (x1,y1),(x2,y2) in zip(c1.get_offsets(),c2.get_offsets()):
    CAplot.plt.plot( [x1,x2], [y1,y2], color="#d0d0d0", marker=None )
CAplot.finish_panel( ax=ax, ylabel="R2", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55, x_ticks=[0,1], x_ticklabels=["S-B","S-L"], x_tick_rotation=0, tick_size=None, label_size=None, title_size=None, legend_size=None, despine=True, legendpos=0)

ax = CAplot.plt.subplot2grid((1,3),(0,2))
CAplot.sns.violinplot(data=dfR2_per_neuron_LG, ax=ax, inner=None, color="#b0b0b0")
CAplot.sns.swarmplot(data=dfR2_per_mouse_LG, ax=ax, color="k", size=3, linewidth=1, edgecolor="None")
c1,c2 = ax.collections[2],ax.collections[3]
for (x1,y1),(x2,y2) in zip(c1.get_offsets(),c2.get_offsets()):
    CAplot.plt.plot( [x1,x2], [y1,y2], color="#d0d0d0", marker=None )
CAplot.finish_panel( ax=ax, ylabel="R2", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55, x_ticks=[0,1], x_ticklabels=["Lost","Gain"], x_tick_rotation=0, tick_size=None, label_size=None, title_size=None, legend_size=None, despine=True, legendpos=0)

CAplot.finish_figure( filename="5ED7bc-Encodingmodel-Full-R2-Fraction", path=figpath, wspace=0.7, hspace=0.7 )

print(' ')
print(' -- Fraction of significant neurons --')
print('Before learning: {:0.3f} \u00B1 {:0.3f} (n={})'.format( *CAgeneral.mean_sem(dfF_per_mouse["B"]) ))
print('After learning: {:0.3f} \u00B1 {:0.3f} (n={})'.format( *CAgeneral.mean_sem(dfF_per_mouse["L"]) ))
CAstats.report_wmpsr_test(dfF_per_mouse["B"], dfF_per_mouse["L"])
print(' ')

print(' ')
print(' -- R2 of significant stable neurons --')
print('Before learning: {:0.3f} \u00B1 {:0.3f} (n={})'.format( *CAgeneral.mean_sem(dfR2_per_mouse_ST["S-B"]) ))
print('After learning: {:0.3f} \u00B1 {:0.3f} (n={})'.format( *CAgeneral.mean_sem(dfR2_per_mouse_ST["S-L"]) ))
CAstats.report_wmpsr_test(dfR2_per_mouse_ST["S-B"], dfR2_per_mouse_ST["S-L"])
print(' ')

print(' ')
print(' -- R2 of significant lost/gained neurons --')
print('Lost, before learning: {:0.3f} \u00B1 {:0.3f} (n={})'.format( *CAgeneral.mean_sem(dfR2_per_mouse_LG["Lost"]) ))
print('Gained, after learning: {:0.3f} \u00B1 {:0.3f} (n={})'.format( *CAgeneral.mean_sem(dfR2_per_mouse_LG["Gain"]) ))
CAstats.report_wmpsr_test(dfR2_per_mouse_LG["Lost"], dfR2_per_mouse_LG["Gain"])
print(' ')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
