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
import CAplot, CAgeneral, CAstats, klimbic

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load all data
mdata,run_nr,all_mice = klimbic.quick_load_behavioral_box_data_run1_run2()
n_mice = len(all_mice)
colormap = "RdBu_r"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate average performance and reaction time per stimulus
n_trials = {}
stim_number = {}
performance = {}
reaction_time = {}
n_trials_level = {}
performance_level = {}
reaction_time_level = {}

# Loop mice
for m in all_mice:

    n_trials[m] = np.zeros(49)
    stim_number[m] = np.zeros(49)
    performance[m] = np.zeros(49)
    reaction_time[m] = np.zeros(49)
    n_trials_level[m] = np.zeros(13)
    performance_level[m] = np.zeros(13)
    reaction_time_level[m] = np.zeros(13)

    # Get all data of highest CT
    ctdata = mdata[m][5]

    print("Mouse {}".format(m))

    # Loop sessions
    for s_nr,x in enumerate(ctdata):

        # Loop trials and add data per trial (skipping incomplete and missed)
        for t,d,o,r in zip( x['trial target id'], x['trial dummy id'], x["trial outcome"], x["trial screen RT"] ):
            if o < 2:
                cat_level_t = int(klimbic.CATEGORY[m]["level"][t]+6)
                cat_level_d = int(klimbic.CATEGORY[m]["level"][d]+6)

                stim_number[m][t] = t
                n_trials[m][t] += 1
                reaction_time[m][t] += r

                stim_number[m][d] = d
                n_trials[m][d] += 1
                reaction_time[m][d] += r

                n_trials_level[m][cat_level_t] += 1
                n_trials_level[m][cat_level_d] += 1
                reaction_time_level[m][cat_level_t] += r
                reaction_time_level[m][cat_level_d] += r

                if o == 0:
                    performance[m][t] += 1
                    performance_level[m][cat_level_t] += 1
                else:
                    performance[m][d] += 1
                    performance_level[m][cat_level_d] += 1

    performance[m] = np.divide(performance[m], n_trials[m])
    reaction_time[m] = np.divide(reaction_time[m], n_trials[m])

    performance_level[m] = np.divide(performance_level[m], n_trials_level[m])
    reaction_time_level[m] = np.divide(reaction_time_level[m], n_trials_level[m])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display colored performance plots per mouse
fig = CAplot.init_figure(fig_size=(16,8))
panel_ids,n = CAplot.grid_indices( 8, n_columns=4 )
for m,py,px in zip( all_mice, panel_ids[0], panel_ids[1] ):
    ax = CAplot.plt.subplot2grid( n, (py,px) )

    mouse_plot = np.reshape(performance[m][:], (7,7))
    CAplot.plt.imshow(mouse_plot, cmap=colormap, vmin=0.0, vmax=1.0)
    CAplot.plt.title(m)
    CAplot.plt.axis('off')

CAplot.finish_figure( filename="1e-BehavioralChambers-Performance-2Dplots", wspace=0.0, hspace=0.4 )

fig = CAplot.init_figure(fig_size=(16,8))
CAplot.plt.imshow(np.full((1,1),0.0), cmap=colormap, vmin=0.0, vmax=1.0)
CAplot.plt.title("This plot is only here so that the colorbar can be used in the figure ...")
cbar = CAplot.plt.colorbar()
CAplot.plt.axis('off')
CAplot.finish_figure( filename="1e-BehavioralChambers-Colorbar-2Dplots", wspace=0.0, hspace=0.4 )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mean performance and reaction time across mice
m_perf_mat = np.zeros((13,n_mice))
m_rt_mat = np.zeros((13,n_mice))
m_level_mat = np.zeros((13,n_mice))
for m_nr,m in enumerate(all_mice):
    m_level_mat[:,m_nr] = np.arange(-6,7)
    m_perf_mat[:,m_nr] = performance_level[m]
    m_rt_mat[:,m_nr] = reaction_time_level[m]
xvalues = np.arange(-6,7,1)

# Convert to continuous numpy arrays for statistics
m_perf_test = np.array(m_perf_mat)
m_rt_test = np.array(m_rt_mat)
m_level_test = np.array(m_level_mat)
m_perf_test = np.delete(m_perf_test, 6, axis=0)
m_rt_test = np.delete(m_rt_test, 6, axis=0)
m_level_test = np.delete(m_level_test, 6, axis=0)

samplelist = []
for s in range(m_perf_test.shape[0]):
    samplelist.append(m_perf_test[s,:])

CAstats.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 )
statmat = np.full((len(samplelist),len(samplelist)),np.NaN)
for c1 in range(len(samplelist)):
    for c2 in range(len(samplelist)):
        if c1 < c2:
            p = CAstats.report_wmpsr_test( samplelist[c1], samplelist[c2], n_indents=2, alpha=0.05, bonferroni=1, preceding_text="{:2d} vs {:2d}: ".format(c1,c2))
            statmat[c1,c2] = p
            if p < 0.01:
                statmat[c2,c1] = 1

# Figure with performance and response time
fig = CAplot.init_figure(fig_size=(12,6))

ax = CAplot.plt.subplot2grid( (1,2), (0,0) )
mn,sem,n = CAgeneral.mean_sem( m_perf_mat, axis=1 )
for m_nr in range(n_mice):
    CAplot.plt.plot( xvalues, m_perf_mat[:,m_nr], marker="o", markersize=1,
        color='#aaaaaa', linestyle='-', linewidth=1 )
CAplot.line( xvalues, mn, sem )
CAplot.finish_panel( ax, ylabel="Fraction chosen", xlabel="Boundary distance", legend="off", y_minmax=[0.0,1.000001], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[-6.0,6.0001], x_step=[1.0,0], x_margin=0.55, x_axis_margin=0.55, despine=True)

ax = CAplot.plt.subplot2grid( (1,2), (0,1) )
mn,sem,n = CAgeneral.mean_sem( m_rt_mat, axis=1 )
for x,y,e in zip(xvalues, mn, sem):
    CAplot.bar( x, y, e )
for m_nr in range(n_mice):
    CAplot.plt.plot( xvalues, m_rt_mat[:,m_nr], marker="o", markersize=1,
        color='#aaaaaa', linestyle='-', linewidth=1 )

CAplot.finish_panel( ax, ylabel="Response time (s)", xlabel="Boundary distance", legend="off", y_minmax=[0.0,6.000001], y_step=[1.0,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=[-6.0,6.0001], x_step=[1.0,0], x_margin=0.55, x_axis_margin=0.55, despine=True)

# Finish figure layout and save
CAplot.finish_figure( filename="1c-ED1c-BehavioralChambers-Performance-Responsetime-1Dplot", wspace=0.4, hspace=0.8 )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()
