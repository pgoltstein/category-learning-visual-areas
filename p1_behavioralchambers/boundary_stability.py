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

# Settings
max_spacing = 25

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load all data
mdata,run_nr,all_mice = klimbic.quick_load_behavioral_box_data_run1_run2()
n_mice = len(all_mice)
X = klimbic.GRATING["orientation_id"]
Y = klimbic.GRATING["spatialf_id"]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate average performance and reaction time per stimulus
boundary_angle = {}
boundary_deviation = {}
max_n_sessions = 0
min_n_sessions = 999

# Loop mice
for m_nr,m in enumerate(all_mice):
    boundary_angle[m] = []
    boundary_deviation[m] = []

    # Get data of highest CT
    ctdata = mdata[m][5]
    rndata = run_nr[m][5]
    max_n_sessions = np.max((max_n_sessions,len(ctdata)))
    min_n_sessions = np.min((min_n_sessions,len(ctdata)))

    # Loop sessions
    for s_nr,x in enumerate(ctdata):
        if rndata[s_nr] == 1:
            continue

        n_trials = np.zeros(49)
        performance = np.zeros(49)

        # Loop trials and add data per trial (skipping incomplete and missed)
        for t,d,o in zip( x['trial target id'], x['trial dummy id'], x["trial outcome"] ):
            if o < 2:
                cat_level_t = int(klimbic.CATEGORY[m]["level"][t])
                if cat_level_t < 99:
                    n_trials[t] += 1
                    n_trials[d] += 1
                    if o == 0:
                        performance[t] += 1
                    else:
                        performance[d] += 1

        # Get performance matrix for this session
        performance_mat = np.divide(performance, n_trials)
        b_a, (x_line,y_line) = klimbic.calculate_boundary( performance_mat )
        boundary_angle[m].append(b_a)
        if np.mod(m_nr,2) == 0:
            boundary_deviation[m].append( np.abs(b_a-45.0) )
        else:
            boundary_deviation[m].append( np.abs(b_a+45.0) )

# Get difference over n days
boundary_ndays = np.zeros((n_mice,max_spacing))
boundary_diff = np.zeros((n_mice,max_spacing))
boundary_diff_shuffle = np.zeros((n_mice,max_spacing))
for m_nr,m in enumerate(all_mice):
    for d in range(max_spacing):
        ang_diff = []
        ang_diff_shuffle = []
        ang_vec = boundary_angle[m]
        ang_vec_shuffle = ang_vec.copy()
        np.random.shuffle(ang_vec_shuffle)
        for x in range(len(ang_vec)-(d+1)):
            ang_diff.append( np.abs( ang_vec[x+(d+1)] - ang_vec[x] ) )
            ang_diff_shuffle.append( np.abs( ang_vec_shuffle[x+(d+1)] - ang_vec_shuffle[x] ) )

        boundary_ndays[m_nr,d] = d
        boundary_diff[m_nr,d] = np.mean( np.array(ang_diff) )
        boundary_diff_shuffle[m_nr,d] = np.mean( np.array(ang_diff_shuffle) )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display boundary angle per mouse
fig = CAplot.init_figure(fig_size=(18,6))
ax = CAplot.plt.subplot2grid( (1,5), (0,0), colspan=4 )

# Plot individual mouse data points
all_ydata = np.full((n_mice,max_n_sessions),np.NaN)
first_last_dev = np.full((n_mice,2),np.NaN)
for m_nr,m in enumerate(all_mice):
    ydata = boundary_deviation[m]
    all_ydata[m_nr,:len(ydata)] = ydata
    first_last_dev[m_nr,0] = np.nanmean(ydata[0:5])
    first_last_dev[m_nr,1] = np.nanmean(ydata[(min_n_sessions-5):min_n_sessions])
    xvalues = np.arange(0,len(ydata))
    # CAplot.plt.plot( xvalues, ydata, color=CAplot.colors[m_nr], linestyle='-', linewidth=1 )
    CAplot.plt.plot( xvalues, ydata, color="#aaaaaa", linestyle='-', linewidth=1 )

mn,sem,n = CAgeneral.mean_sem( all_ydata, axis=0 )
xvalues = np.arange(0,max_n_sessions)
CAplot.line( xvalues, mn, sem )

# Finish panel layout
CAplot.finish_panel( ax, ylabel="Boundary deviation ($^\circ$)", xlabel="Training session", legend="off", y_minmax=[0.0,60.000001], y_step=[10.0,0], y_margin=1.0, y_axis_margin=0.0, x_minmax=None, x_step=None, x_margin=0.55, x_axis_margin=0.55,  despine=True)

ax = CAplot.plt.subplot2grid( (1,5), (0,4) )
xvalues = np.arange(2)
for m_nr in range(len(first_last_dev)):
    CAplot.line( xvalues, first_last_dev[m_nr,:], line_color='#aaaaaa', sem_color='#aaaaaa' )
mn,sem,n = CAgeneral.mean_sem( first_last_dev, axis=0 )
for x,y,e in zip(xvalues, mn, sem):
    CAplot.bar( x, y, e )

CAplot.finish_panel( ax, ylabel="Boundary deviation ($^\circ$)", legend="off", y_minmax=[0.0,30.000001], y_step=[5.0,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,1.0], x_step=[1.0,0], x_margin=0.75, x_axis_margin=0.55, despine=True, x_ticks=[0,1], x_ticklabels={"Last 5","First 5"} )
CAplot.plt.xticks(rotation=45)

# Finish figure layout and save
CAplot.finish_figure( filename="1ED1e-BehavioralChambers-BoundaryDeviation", wspace=0.8, hspace=0.8 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display boundary stability

fig,ax = CAplot.init_figure_axes(fig_size=(6,6))

xvalues = np.arange(1,max_spacing+1,1)

mn,sem,n = CAgeneral.mean_sem( boundary_diff, axis=0 )
CAplot.line( xvalues, mn, sem, label="data" )

mn,sem,n = CAgeneral.mean_sem( boundary_diff_shuffle, axis=0 )
CAplot.line( xvalues, mn, sem, line_color='#aaaaaa', sem_color='#aaaaaa', label="shuffle" )

CAplot.finish_panel( ax, ylabel=r"$\Delta$Boundary angle ($^\circ$)", xlabel="Session spacing", legend="off", y_minmax=[0.0,15.000001], y_step=[3,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,max_spacing+1], x_step=[5.0,0], x_margin=0.55, x_axis_margin=0.55, despine=True)

# Finish figure layout and save
CAplot.finish_figure( filename="1ED1f-BehavioralChambers-BoundaryDrift", wspace=0.8, hspace=0.8 )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Stats

# Normalize per mouse
BS = np.repeat( np.mean(boundary_diff,axis=1).reshape((-1,1)), boundary_diff.shape[1], axis=1)
boundary_diff = boundary_diff - BS

BS = np.repeat( np.mean(boundary_diff_shuffle,axis=1).reshape((-1,1)), boundary_diff_shuffle.shape[1], axis=1)
boundary_diff_shuffle = boundary_diff_shuffle - BS

print("\n -- STATS --")
CAstats.report_wmpsr_test( first_last_dev[:,0], first_last_dev[:,1], n_indents=0, alpha=0.05, bonferroni=1, preceding_text="Boundary deviation, first vs. last: " )

samplelist = []
for s in range(boundary_diff.shape[1]):
    samplelist.append(boundary_diff[:,s])

print("Boundary drift")
CAstats.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 )

# Non-param
samplelist = []
for s in range(boundary_diff_shuffle.shape[1]):
    samplelist.append(boundary_diff_shuffle[:,s])

print("Boundary drift (shuffled)")
CAstats.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()
