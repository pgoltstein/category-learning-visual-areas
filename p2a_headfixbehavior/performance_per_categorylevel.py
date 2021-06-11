#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:35:35 2017

@author: pgoltstein
"""


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get imports
import numpy as np
import sys
sys.path.append('../xx_analysissupport')

# Add local imports
import CAplot, CAgeneral, matdata

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load all data
all_mice,mdata = matdata.quick_load_full_infointegr_data()

# Add data of second behavioral experiment
all_mice2,mdata2 = matdata.quick_load_second_infointegr_data()
all_mice.extend(all_mice2)
mdata.update(mdata2)

all_mice = ['A77','C01','C04','C07','W06','W08','W09','W10']

colormap = "RdBu_r"
n_mice = len(all_mice)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate average performance and reaction time per stimulus
category_id = {}
n_trials = {}
performance = {}
n_trials_level = {}
performance_level = {}

n_trials_rt = {}
reaction_time = {}
n_trials_level_rt = {}
reaction_time_level = {}
alllevels = []

# Loop mice
for m in all_mice:

    category_id[m] = np.zeros((6,7))
    n_trials[m] = np.zeros((6,7))
    performance[m] = np.zeros((6,7))
    n_trials_level[m] = np.zeros(12)
    performance_level[m] = np.zeros(12)

    n_trials_rt[m] = np.zeros((6,7))
    reaction_time[m] = np.zeros((6,7))
    n_trials_level_rt[m] = np.zeros(12)
    reaction_time_level[m] = np.zeros(12)

    # Get all data of CT2-CT5
    ctdata = []
    for ct in range(1,6):
        ctdata.extend(mdata[m]["sessions"][ct])

    print("Mouse {}".format(m))

    # Loop sessions
    for s_nr,x in enumerate(ctdata):

        if x["angleset"] == 0 and m[0] != 'W':
            continue

        outcome = x["outcome"]
        ses_reaction_time = np.zeros_like(outcome) * np.NaN

        trial_on = x["StimOnsetIx"]
        trial_off = x["RespWinStopIx"]
        licks = np.array(x["LickIx"])
        for tr,(on,off) in enumerate(zip(trial_on,trial_off)):
            licks = licks[licks>on]
            if len(licks) == 0:
                break
            if licks[0] < off:
                ses_reaction_time[tr] = (licks[0]-on)/500

        categoryid = x["categoryid"]
        angleix = x["angleix"]
        spatialfix = x["spatialfix"]
        level = x["level"]

        # Loop trials and add data per trial (skipping incomplete and missed)
        for ou,rt,lv,ang,spf,cat in zip( outcome, ses_reaction_time, level, angleix, spatialfix, categoryid ):
            alllevels.append(lv)
            if ou < 2:
                cat_level = int(lv+5)
                spf = int(spf)
                ang = int(ang)

                if ~np.isnan(rt):
                    n_trials_rt[m][spf,ang] += 1
                    reaction_time[m][spf,ang] += rt

                    n_trials_level_rt[m][cat_level] += 1
                    reaction_time_level[m][cat_level] += rt

                category_id[m][spf,ang] += cat
                n_trials[m][spf,ang] += 1
                n_trials_level[m][cat_level] += 1
                if ou == 1 and cat == 1: # Left
                    performance[m][spf,ang] += 1
                    performance_level[m][cat_level] += 1
                if ou == 0 and cat == 2: # Left
                    performance[m][spf,ang] += 1
                    performance_level[m][cat_level] += 1

    category_id[m] = np.divide(category_id[m], n_trials[m])
    performance[m] = np.divide(performance[m], n_trials[m])
    reaction_time[m] = np.divide(reaction_time[m], n_trials_rt[m])

    performance_level[m] = np.divide(performance_level[m], n_trials_level[m])
    reaction_time_level[m] = np.divide(reaction_time_level[m], n_trials_level_rt[m])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display mean performance and reaction time across mice
m_perf_mat = np.zeros((12,n_mice))
m_rt_mat = np.zeros((12,n_mice))
m_level_mat = np.zeros((12,n_mice))
for m_nr,m in enumerate(all_mice):
    m_perf_mat[:,m_nr] = performance_level[m]
    m_level_mat[:,m_nr] = np.arange(-5,7)
    m_rt_mat[:,m_nr] = reaction_time_level[m]
xvalues = np.arange(-5,7,1)
xticklabels = np.concatenate((np.arange(-6,0,1),np.arange(1,7,1)))

fig = CAplot.init_figure(fig_size=(12,6))

ax = CAplot.plt.subplot2grid( (1,2), (0,0) )
mn,sem,n = CAgeneral.mean_sem( m_perf_mat, axis=1 )
for m_nr in range(n_mice):
    CAplot.plt.plot( xvalues, m_perf_mat[:,m_nr], marker="o", markersize=1,
        color='#aaaaaa', linestyle='-', linewidth=1 )
CAplot.line( xvalues, mn, sem )
CAplot.finish_panel( ax, ylabel="Left choice (p)", xlabel="Boundary distance", legend="off", y_minmax=[0.0,1.000001], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[-5.0,6.0], x_margin=0.95, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)

ax = CAplot.plt.subplot2grid( (1,2), (0,1) )
mn,sem,n = CAgeneral.mean_sem( m_rt_mat, axis=1 )
for x,y,e in zip(xvalues, mn, sem):
    CAplot.bar( x, y, e )
for m_nr in range(n_mice):
    CAplot.plt.plot( xvalues, m_rt_mat[:,m_nr], marker="o", markersize=1,
        color='#aaaaaa', linestyle='-', linewidth=1 )

CAplot.finish_panel( ax, ylabel="Response time (s)", xlabel="Boundary distance", legend="off", y_minmax=[0.0,3.000001], y_step=[0.5,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[-5.0,6.0], x_step=[2.0,0], x_margin=0.95, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)

# Finish figure layout and save
CAplot.finish_figure( filename="2ED2de-HeadFixed-Performance-Responsetime-1Dplot", wspace=0.4, hspace=0.8 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()
