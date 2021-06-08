#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:35:35 2017

@author: pgoltstein
"""


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get imports
import numpy as np
import pandas as pd
import sys
sys.path.append('../xx_analysissupport')

# Add local imports
import CAplot, CAgeneral, CAstats, klimbic

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load all data
mdata,run_nr,all_mice = klimbic.quick_load_behavioral_box_data_run1_run2()
n_mice = len(all_mice)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate average performance and reaction time per stimulus
first_trials = {}
first_trials_ids = {}
last_trials = {}
last_trials_ids = {}
last_trials_first = {}
last_trials_first_ids = {}

# Loop mice
for m_nr,m in enumerate(all_mice):
    first_trials[m] = [[],[],[],[],[],[]]
    last_trials[m] = [[],[],[],[],[],[]]
    last_trials_first[m] = [[],[],[],[],[],[]]
    first_trials_ids[m] = [[],[],[],[],[],[]]
    last_trials_ids[m] = [[],[],[],[],[],[]]
    last_trials_first_ids[m] = [[],[],[],[],[],[]]

    # Loop CTs
    for lev,ct in zip(list(range(6,0,-1)),list(range(0,6))):

        # First, last session
        firstdata = mdata[m][ct][0]
        lastdata = mdata[m][ct][-1]

        # Only for prototype shaping sessions
        outcome = np.array(firstdata["trial outcome"])
        if lev==6:
            # if '3'(repeat) is followed by '2'(timeout), count as error
            for t in range(1,len(outcome)):
                if outcome[t]==2 and outcome[t-1]==3:
                    outcome[t]=1.0

        # Loop trials and add data per trial (skipping incomplete and missed)
        for t,d,o in zip( firstdata['trial target id'], firstdata['trial dummy id'], outcome ):
            if o < 2:
                cat_level_t = int(klimbic.CATEGORY[m]["level"][t])

                # This is a bug fix, because on mouse C01,3,5,7 the target id was different on the first training session
                if np.mod(m_nr,2) == 0 and lev==6:
                    cat_level_t = 6

                # Prototype sessions are skipped because the mice had prior experience
                if cat_level_t == lev and ct>0:
                    if t not in first_trials_ids[m][ct]:
                    # if t not in first_trials_ids[m][ct] or len(first_trials_ids[m][ct])<6:
                        first_trials[m][ct].append(o==0)
                        first_trials_ids[m][ct].append(t)

        # Loop trials and add data per trial (skipping incomplete and missed)
        for t,d,o in zip( lastdata['trial target id'][::-1], lastdata['trial dummy id'][::-1], lastdata["trial outcome"][::-1] ):
        # for t,d,o in zip( lastdata['trial target id'], lastdata['trial dummy id'], lastdata["trial outcome"] ):
            if o < 2:
                cat_level_t = int(klimbic.CATEGORY[m]["level"][t])
                if cat_level_t == lev:
                    if t not in last_trials_ids[m][ct]:
                    # if t not in last_trials_ids[m][ct] or len(last_trials_ids[m][ct])<6:
                        last_trials[m][ct].append(o==0)
                        last_trials_ids[m][ct].append(t)

        # Loop trials and add data per trial (skipping incomplete and missed)
        for t,d,o in zip( lastdata['trial target id'], lastdata['trial dummy id'], lastdata["trial outcome"] ):
            if o < 2:
                cat_level_t = int(klimbic.CATEGORY[m]["level"][t])
                if cat_level_t == lev:
                    if t not in last_trials_first_ids[m][ct]:
                    # if t not in last_trials_ids[m][ct] or len(last_trials_ids[m][ct])<6:
                        last_trials_first[m][ct].append(o==0)
                        last_trials_first_ids[m][ct].append(t)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mean performance for first and last
m_all_first = np.zeros(n_mice)
m_all_last = np.zeros(n_mice)
for m_nr,m in enumerate(all_mice):
    m_all_first[m_nr] = np.nanmean( first_trials[m][-2] + first_trials[m][-1] )
    m_all_last[m_nr] = np.nanmean( last_trials_first[m][-2] + last_trials_first[m][-1] )

fig,ax = CAplot.init_figure_axes(fig_size=(4,6))
xvalues = np.arange(2)
mn,sem,n = CAgeneral.mean_sem( m_all_first, axis=0 )
CAplot.bar( 0, mn, sem )
mn,sem,n = CAgeneral.mean_sem( m_all_last, axis=0 )
CAplot.bar( 1, mn, sem )
for m_nr in range(n_mice):
    CAplot.line( xvalues, [m_all_first[m_nr],m_all_last[m_nr]], line_color='#aaaaaa', sem_color='#aaaaaa' )

CAplot.finish_panel( ax, ylabel="Fraction correct", xlabel=None, legend="off", y_minmax=[0.0,1.000001], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,1.00001], x_step=[1.0,0], x_ticks=xvalues, x_ticklabels=["First","Last"], x_margin=0.75, x_axis_margin=0.55, despine=True)
CAplot.plt.xticks(rotation=45)

# Finish figure layout and save
CAplot.finish_figure( filename="1d-BehavioralChambers-FirstLast-Level56", wspace=0.8, hspace=0.8 )

CAstats.report_wmpsr_test( m_all_first, m_all_last, n_indents=0, alpha=0.05, bonferroni=1, preceding_text="All (level 5 & 6): ")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()
