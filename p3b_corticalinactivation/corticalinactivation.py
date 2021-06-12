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
import CAplot, CAgeneral, CAstats, matdata


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load all data of info-integr pilot
datafile = matdata.get_muscimol_infointegr_data()
all_mice,mdata_acsf,mdata_musc = matdata.extract_muscimol_infointegr_data( datafile )
all_mice,mdata_acsf,mdata_musc = matdata.quick_load_muscimol_infointegr_data()
n_mice = len(all_mice)

# Get performance and reaction time per mouse
pf = np.full((n_mice,2),np.NaN)
nt = np.full((n_mice,2),np.NaN)
rt = np.full((n_mice,2),np.NaN)
for m_nr,m in enumerate(all_mice):
    x = mdata_acsf[m]["sessions"][0][0]
    outcome = x["outcome"]
    reaction_time_all = np.zeros_like(outcome) * np.NaN
    trial_on = x["StimOnsetIx"]
    trial_off = x["RespWinStopIx"]
    licks = np.array(x["LickIx"])
    for tr,(on,off) in enumerate(zip(trial_on,trial_off)):
        licks = licks[licks>on]
        if len(licks) == 0:
            break
        if licks[0] < off:
            reaction_time_all[tr] = (licks[0]-on)/x["aux-sf"]

    pf[m_nr,0] = np.sum(outcome==1) / (np.sum(outcome==1) + np.sum(outcome==0))
    nt[m_nr,0] = np.sum(outcome==1) + np.sum(outcome==0)
    rt[m_nr,0] = np.nanmean(reaction_time_all[outcome==0])

    x = mdata_musc[m]["sessions"][0][0]
    outcome = x["outcome"]
    reaction_time_all = np.zeros_like(outcome) * np.NaN
    trial_on = x["StimOnsetIx"]
    trial_off = x["RespWinStopIx"]
    licks = np.array(x["LickIx"])
    for tr,(on,off) in enumerate(zip(trial_on,trial_off)):
        licks = licks[licks>on]
        if len(licks) == 0:
            break
        if licks[0] < off:
            reaction_time_all[tr] = (licks[0]-on)/x["aux-sf"]

    pf[m_nr,1] = np.sum(outcome==1) / (np.sum(outcome==1) + np.sum(outcome==0))
    nt[m_nr,1] = np.sum(outcome==1) + np.sum(outcome==0)
    rt[m_nr,1] = np.nanmean(reaction_time_all[outcome==0])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display learning curve
fig = CAplot.init_figure(fig_size=(9,5))

ax = CAplot.plt.subplot2grid( (1,3), (0,0) )
xvalues = np.arange(2)
for m_nr in range(pf.shape[0]):
    CAplot.line( xvalues, pf[m_nr,:], line_color='#aaaaaa', sem_color='#aaaaaa' )
mn,sem,n = CAgeneral.mean_sem( pf, axis=0 )
for x,y,e in zip(xvalues, mn, sem):
    CAplot.bar( x, y, e )

CAplot.finish_panel( ax, ylabel="Fraction correct", legend="off", y_minmax=[0.0,1.000001], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,1.0], x_step=[1.0,0], x_margin=0.75, x_axis_margin=0.55, despine=True, x_ticks=[0,1], x_ticklabels=["aCSF","Muscimol"], x_tick_rotation=45 )


ax = CAplot.plt.subplot2grid( (1,3), (0,1) )
xvalues = np.arange(2)
for m_nr in range(nt.shape[0]):
    CAplot.line( xvalues, nt[m_nr,:], line_color='#aaaaaa', sem_color='#aaaaaa' )
mn,sem,n = CAgeneral.mean_sem( nt, axis=0 )
for x,y,e in zip(xvalues, mn, sem):
    CAplot.bar( x, y, e )

CAplot.finish_panel( ax, ylabel="# of trials", legend="off", y_minmax=[0.0,200.000001], y_step=[40.0,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,1.0], x_step=[1.0,0], x_margin=0.75, x_axis_margin=0.55, despine=True, x_ticks=[0,1], x_ticklabels=["aCSF","Muscimol"], x_tick_rotation=45 )


ax = CAplot.plt.subplot2grid( (1,3), (0,2) )
xvalues = np.arange(2)
for m_nr in range(rt.shape[0]):
    CAplot.line( xvalues, rt[m_nr,:], line_color='#aaaaaa', sem_color='#aaaaaa' )
mn,sem,n = CAgeneral.mean_sem( rt, axis=0 )
for x,y,e in zip(xvalues, mn, sem):
    CAplot.bar( x, y, e )

CAplot.finish_panel( ax, ylabel="Response time (s)", legend="off", y_minmax=[0.0,4.000001], y_step=[1.0,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,1.0], x_step=[1.0,0], x_margin=0.75, x_axis_margin=0.55, despine=True, x_ticks=[0,1], x_ticklabels=["aCSF","Muscimol"], x_tick_rotation=45 )

# Finish figure layout and save
CAplot.finish_figure( filename="3f-ED4cd-ChronicImagingBehavior-CorticalInactivation", wspace=0.8, hspace=0.8 )


# Stats
print("\nSTATS")
print("Performance")
CAstats.report_mean(pf[:,0],pf[:,1])
CAstats.report_wmpsr_test(pf[:,0],pf[:,1], alternative="greater")
print("# of trials")
CAstats.report_mean(nt[:,0],nt[:,1])
CAstats.report_wmpsr_test(nt[:,0],nt[:,1], alternative="greater")
print("Response time")
CAstats.report_mean(rt[:,0],rt[:,1])
CAstats.report_wmpsr_test(rt[:,0],rt[:,1], alternative="greater")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()
