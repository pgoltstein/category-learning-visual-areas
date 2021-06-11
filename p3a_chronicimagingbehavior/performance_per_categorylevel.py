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
import argparse
sys.path.append('../xx_analysissupport')

# Add local imports
import CAplot, CAgeneral, CArec, matdata

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments
parser = argparse.ArgumentParser( description = "Makes plots of headfixed behavior in the chronic imaging experiment. \n (written by Pieter Goltstein - July 2016)")
parser.add_argument('data_set_name', type=str, help= 'Use "baseline" or "category"')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load all data
all_mice,mdata = matdata.quick_load_chronic_infointegr_data()
n_mice = len(all_mice)
colormap = "RdBu_r"
data_set_name = args.data_set_name

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate average performance and reaction time per stimulus
use_ct = 0 if data_set_name == "baseline" else 3

category_id = {}
category_level = {}
n_trials = {}
performance = {}
n_trials_level = {}
performance_level = {}

n_trials_rt = {}
reaction_time = {}
n_trials_level_rt = {}
reaction_time_level = {}

# Loop mice
for m in all_mice:
    print("Mouse {}".format(m))

    category_id[m] = np.zeros((5,6))
    category_level[m] = np.zeros((5,6))
    n_trials[m] = np.zeros((5,6))
    performance[m] = np.zeros((5,6))
    n_trials_level[m] = np.zeros(10)
    performance_level[m] = np.zeros(10)

    n_trials_rt[m] = np.zeros((5,6))
    reaction_time[m] = np.zeros((5,6))
    n_trials_level_rt[m] = np.zeros(10)
    reaction_time_level[m] = np.zeros(10)

    # Get all data of CT0 or CT3 (category)
    ctdata = mdata[m]["sessions"][use_ct]

    # Get category stimuli and category level
    category_stimuli = matdata.get_category_stimuli( mdata[m]["sessions"][3][-1] )
    recoded_category_stimuli = CArec.recode_category_stimuli(category_stimuli)
    linear_value_mat = matdata.convert_recoded_cat_to_1Dvalue_matrix( recoded_category_stimuli, mat_size=(5,6) )

    # Loop sessions
    imaging_started_criterion_reached = False
    for s_nr,x in enumerate(ctdata):

        if not imaging_started_criterion_reached:
            if x["aux-sf"] == 5000:
                imaging_started_criterion_reached = True
            else:
                print("Session {}, imaging not started, skipping".format(s_nr))
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
                ses_reaction_time[tr] = (licks[0]-on)/x["aux-sf"]

        categoryid = x["categoryid"]
        angleix = x["angleix"]
        spatialfix = x["spatialfix"]
        level = x["level"]

        # Loop trials and add data per trial (skipping incomplete and missed)
        for ou,rt,lv,ang,spf,cat in zip( outcome, ses_reaction_time, level, angleix, spatialfix, categoryid ):
            spf = int(spf)
            ang = int(ang)
            if ou < 2 and ~np.isnan(linear_value_mat[spf,ang]):
                cat_level = int(linear_value_mat[spf,ang])

                if ~np.isnan(rt):
                    n_trials_rt[m][spf,ang] += 1
                    reaction_time[m][spf,ang] += rt

                    n_trials_level_rt[m][cat_level] += 1
                    reaction_time_level[m][cat_level] += rt

                category_id[m][spf,ang] += (1 if cat==1 else 0)
                category_level[m][spf,ang] += cat_level
                n_trials[m][spf,ang] += 1
                n_trials_level[m][cat_level] += 1
                if ou == 1 and cat == 1: # Left
                    performance[m][spf,ang] += 1
                    performance_level[m][cat_level] += 1
                if ou == 0 and cat == 2: # Left
                    performance[m][spf,ang] += 1
                    performance_level[m][cat_level] += 1

    category_id[m] = np.divide(category_id[m], n_trials[m])
    category_level[m] = np.divide(category_level[m], n_trials[m])
    performance[m] = np.divide(performance[m], n_trials[m])
    reaction_time[m] = np.divide(reaction_time[m], n_trials_rt[m])

    performance_level[m] = np.divide(performance_level[m], n_trials_level[m])
    reaction_time_level[m] = np.divide(reaction_time_level[m], n_trials_level_rt[m])

save_dict = { "performance": performance, "category_id": category_id, "category_level": category_level, "mouse_names": all_mice }
save_filename = "../../data/p3a_chronicimagingbehavior/performance_{}_chronic_infointegr.npy".format(data_set_name)
np.save( save_filename, save_dict )
print("Saved performance data in: {}".format(save_filename))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display colored performance plots per mouse
fig = CAplot.init_figure(fig_size=(8,8))
panel_ids,n = CAplot.grid_indices( n_mice, n_columns=4 )
for m,py,px in zip( all_mice, panel_ids[0], panel_ids[1] ):
    ax = CAplot.plt.subplot2grid( n, (py,px) )

    b_a, (x_line,y_line) = matdata.calculate_boundary( category_id[m], normalize=True, mat_size=(5,6) )
    ax.plot(x_line, y_line, color="#000000", linestyle="-")

    b_a, (x_line,y_line) = matdata.calculate_boundary( performance[m], normalize=True, mat_size=(5,6) )
    ax.plot(x_line, y_line, color="#000000", linestyle="--")

    CAplot.plt.imshow(performance[m], cmap=colormap, vmin=0.0, vmax=1.0)
    CAplot.plt.title(m)
    CAplot.plt.axis('off')

CAplot.finish_figure( filename="3d-ED4b-ChronicImagingBehavior-Performance-2Dplots-{}".format(data_set_name), wspace=0.4, hspace=0.4 )

fig = CAplot.init_figure(fig_size=(6,3))
CAplot.plt.imshow(np.full((1,1),0.0), cmap=colormap, vmin=0.0, vmax=1.0)
CAplot.plt.title("This plot is only here so that the colorbar can be used in the figure ...")
cbar = CAplot.plt.colorbar()
CAplot.plt.axis('off')
CAplot.finish_figure( filename="3d-ED4b-ChronicImagingBehavior-Colorbar-2Dplots", wspace=0.2, hspace=0.4 )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display mean performance and reaction time across mice
m_perf_mat = np.zeros((10,n_mice))
m_rt_mat = np.zeros((10,n_mice))
for m_nr,m in enumerate(all_mice):
    m_perf_mat[:,m_nr] = performance_level[m]
    m_rt_mat[:,m_nr] = reaction_time_level[m]
xvalues = np.arange(10)
xticklabels = np.concatenate((np.arange(-5,0,1),np.arange(1,6,1)))

fig = CAplot.init_figure(fig_size=(12,6))

ax = CAplot.plt.subplot2grid( (1,2), (0,0) )
mn,sem,n = CAgeneral.mean_sem( m_perf_mat, axis=1 )
for m_nr in range(n_mice):
    CAplot.plt.plot( xvalues, m_perf_mat[:,m_nr], marker="o", markersize=1,
        color='#aaaaaa', linestyle='-', linewidth=1 )
CAplot.line( xvalues, mn, sem )
CAplot.finish_panel( ax, ylabel="Left choice (p)", xlabel="Boundary distance", legend="off", y_minmax=[0.0,1.000001], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,9.0], x_margin=0.95, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)

ax = CAplot.plt.subplot2grid( (1,2), (0,1) )
mn,sem,n = CAgeneral.mean_sem( m_rt_mat, axis=1 )
for x,y,e in zip(xvalues, mn, sem):
    CAplot.bar( x, y, e )
for m_nr in range(n_mice):
    CAplot.plt.plot( xvalues, m_rt_mat[:,m_nr], marker="o", markersize=1,
        color='#aaaaaa', linestyle='-', linewidth=1 )

CAplot.finish_panel( ax, ylabel="Response time (s)", xlabel="Boundary distance", legend="off", y_minmax=[0.0,3.000001], y_step=[0.5,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[-0.0,9.0], x_margin=0.95, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)

# Finish figure layout and save
CAplot.finish_figure( filename="3e-ChronicImagingBehavior-Performance-Responsetime-1Dplot-{}".format(data_set_name), wspace=0.4, hspace=0.8 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()
