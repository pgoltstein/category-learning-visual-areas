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
# Load all data of info-integr pilot
all_mice,mdata = matdata.quick_load_full_infointegr_data()

# Add data of second behavioral experiment
all_mice2,mdata2 = matdata.quick_load_second_infointegr_data()
all_mice.extend(all_mice2)
mdata.update(mdata2)

all_mice =       ['A77','C01','C04','C07','W06','W08','W09','W10']
mouse_boundary = [ -45 ,  45 ,  45 , -45 , -45 ,  45 ,  45 ,  45 ]

n_mice = len(all_mice)
n_ct = 6

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate learning curve per mouse and CT level
max_n = np.zeros(n_ct)
for m in all_mice:
    for ct_nr,ct in enumerate(mdata[m]["sessions"]):
        max_n[ct_nr] = np.max( [max_n[ct_nr], len(ct)] )

lc = []
rt = []
for ct_nr in range(n_ct):
    lc.append( np.full((n_mice,int(max_n[ct_nr])), np.NaN) )
    rt.append( np.full((n_mice,int(max_n[ct_nr])), np.NaN) )

for m_nr,m in enumerate(all_mice):
    for ct_nr,ct in enumerate(mdata[m]["sessions"]):
        for s_nr,x in enumerate(ct):

            outcome = x["outcome"]
            # reaction_time_all = x["TrRespRtFirstLickIx"]
            reaction_time_all = np.zeros_like(outcome) * np.NaN

            trial_on = x["StimOnsetIx"]
            trial_off = x["RespWinStopIx"]
            licks = np.array(x["LickIx"])
            for tr,(on,off) in enumerate(zip(trial_on,trial_off)):
                licks = licks[licks>on]
                if len(licks) == 0:
                    break
                if licks[0] < off:
                    reaction_time_all[tr] = (licks[0]-on)/500

            performance = np.sum(outcome==1) / (np.sum(outcome==1) + np.sum(outcome==0))
            reaction_time = np.nanmean(reaction_time_all[outcome==0])
            if ct_nr == 0:
                s_ix = (int(max_n[0])-len(ct))+s_nr
                lc[ct_nr][m_nr,s_ix] = performance
                rt[ct_nr][m_nr,s_ix] = reaction_time
            else:
                lc[ct_nr][m_nr,s_nr] = performance
                rt[ct_nr][m_nr,s_nr] = reaction_time

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display learning curve
fig,ax = CAplot.init_figure_axes(fig_size=(18,9))

# Loop CT
ct_offset = (-1*max_n[0])+1
real_offset = (-1*max_n[0])+1
x_ticks = []
x_ticklabels = []
for ct_nr in range(n_ct):

    # Calculate mean trace, sem trace, xvalues
    n_mice,n_datapts = lc[ct_nr].shape
    mean_data,sem_data,n = CAgeneral.mean_sem( lc[ct_nr], axis=0 )
    mean_data[n<2] = np.NaN
    sem_data[n<2] = np.NaN
    xvalues = np.arange(n_datapts) + ct_offset
    ct_offset += max_n[ct_nr]+1
    real_xvalues = np.arange(n_datapts) + real_offset
    real_offset += max_n[ct_nr]
    for xx,rx in zip(xvalues,real_xvalues):
        if xvalues.shape[0] > 6:
            if np.mod(rx,5) == 0:
                x_ticks.append(int(xx))
                x_ticklabels.append(str(int(rx)))
        elif xvalues.shape[0] > 2:
            if np.mod(rx,2) == 0:
                x_ticks.append(int(xx))
                x_ticklabels.append(str(int(rx)))
        else:
            x_ticks.append(int(xx))
            x_ticklabels.append(str(int(rx)))

    # Plot individual mouse data points
    for m_nr in range(n_mice):
        CAplot.plt.plot( xvalues, lc[ct_nr][m_nr,:], marker="o", markersize=1,
            color='#aaaaaa', linestyle='-', linewidth=1 )

    CAplot.line( xvalues, mean_data, sem_data, line_color='#000000', line_width=1, sem_color='#000000' )

# Finish panel layout
CAplot.finish_panel( ax, ylabel="Fraction correct", xlabel="Training session", legend="off", y_minmax=[0.0,1.000001], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[(-1*max_n[0]),ct_offset-1], x_step=None, x_margin=0.55, x_axis_margin=0.55, x_ticks=x_ticks, x_ticklabels=x_ticklabels,  despine=True)

# Finish figure layout and save
CAplot.finish_figure( filename="2ED2c-HeadFixed-LearningCurve", wspace=0.8, hspace=0.8 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()
