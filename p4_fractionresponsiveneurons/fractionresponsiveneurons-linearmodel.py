#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30, 2018

@author: pgoltstein
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get all imports
import numpy as np
import pandas as pd
import os, glob
import warnings
import scipy.stats as scistats
import scipy.optimize
import sys
sys.path.append('../xx_analysissupport')
import CAplot, CAgeneral, CArec, CAstats
import CAanalysissupport as CAsupp

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
warnings.filterwarnings('ignore')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings
CAplot.font_size["title"] = 10
CAplot.font_size["label"] = 7
CAplot.font_size["tick"] = 7
CAplot.font_size["text"] = 6
CAplot.font_size["legend"] = 6

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find imaging location files
locations = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
xlabels = ['Bs-tc', 'Bs-tc', 'Bs-tsk', 'Bs-tsk', 'Lrn-tsk', 'Lrn-tc']
n_locs = len(locations)
n_timepoints = len(xlabels)
data_dir = "../../data/p4_fractionresponsiveneurons"
shuffle_regr = None # number 0, 1, 2, 3 or None (0=base which should not make a difference, thus result in a delta of 0.0)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load and process recordings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_all = []
data_mats_all = []
data_full = []
area_full = []
R2_full = []
beh_full = []
for l_nr,loc in enumerate(locations):

    # Load data from file
    filename = os.path.join(data_dir, 'data_frc_'+loc+'.npy')
    print("Loading data from: {}".format(filename))
    data_dict = np.load(filename, allow_pickle=True).item()
    CAplot.print_dict(data_dict["settings"])
    resp_frac = data_dict["data"]["fr_resampled"]
    behavioral_performance = data_dict["data"]["behavioral_performance"]

    # Add cumulative fraction of responsive neurons to list
    data_mat = np.sum(resp_frac[:,:,:], axis=2)
    data_mats_all.append(data_mat)

    # Define regressors
    n_recs,n_timepoints = data_mat.shape
    n_regr = 4
    X5 = np.zeros((n_timepoints-1,n_regr))
    X6 = np.zeros((n_timepoints,n_regr))
    rr = -1

    rr += 1
    X5[:,rr] = 1.0
    X6[:,rr] = 1.0
    rr += 1
    X5[:,rr] = np.array([8,4,2,1,0]) / 8
    X6[:,rr] = np.array([16,8,4,2,1,0]) / 16
    rr += 1
    X5[:,rr] = np.array([0,0,1,1,0])
    X6[:,rr] = np.array([0,0,1,1,1,0])
    rr += 1
    X5[:,rr] = np.array([0,0,0,1,1])
    X6[:,rr] = np.array([0,0,0,0,1,1])

    weights = np.full((n_recs,n_regr),np.NaN)
    R2s = np.full((n_recs,),np.NaN)
    for r in range(n_recs):
        y = data_mat[r,:]

        if np.sum(np.isnan(y)) == 0:
            (weights[r,:],_) = scipy.optimize.nnls(X6, y)
            pred = np.dot( X6, np.expand_dims(weights[r,:],axis=1) )[:,0]
            SStotal = np.sum( (y-np.mean(y))**2 )
            SSresidual = np.sum( (y-pred)**2 )
            R2s[r] = 1.0 - (SSresidual/SStotal)

            if shuffle_regr is not None:
                X6_shuff = np.array(X6)
                shuff_ixs = np.random.choice(X6.shape[0], size=X6.shape[0], replace=False)
                X6_shuff[:,shuffle_regr] = X6_shuff[shuff_ixs,shuffle_regr]
                (weights_shuff,_) = scipy.optimize.nnls(X6_shuff, y)
                pred_shuff = np.dot( X6_shuff, np.expand_dims(weights_shuff,axis=1) )[:,0]
                SStotal = np.sum( (y-np.mean(y))**2 )
                SSresidual = np.sum( (y-pred_shuff)**2 )
                R2s[r] = R2s[r] - (1.0 - (SSresidual/SStotal))

        elif np.sum(np.isnan(y)) == 1:
            y = np.concatenate([y[:3],y[4:]])
            (weights[r,:],_) = scipy.optimize.nnls(X5, y)
            pred = np.dot( X5, np.expand_dims(weights[r,:],axis=1) )[:,0]
            SStotal = np.sum( (y-np.mean(y))**2 )
            SSresidual = np.sum( (y-pred)**2 )
            R2s[r] = 1.0 - (SSresidual/SStotal)

            if shuffle_regr is not None:
                X5_shuff = np.array(X5)
                shuff_ixs = np.random.choice(X5.shape[0], size=X5.shape[0], replace=False)
                X5_shuff[:,shuffle_regr] = X6_shuff[shuff_ixs,shuffle_regr]
                (weights_shuff,_) = scipy.optimize.nnls(X5_shuff, y)
                pred_shuff = np.dot( X5_shuff, np.expand_dims(weights_shuff,axis=1) )[:,0]
                SStotal = np.sum( (y-np.mean(y))**2 )
                SSresidual = np.sum( (y-pred_shuff)**2 )
                R2s[r] = R2s[r] - (1.0 - (SSresidual/SStotal))

    data_all.append( weights )
    data_full.append( weights )
    R2_full.append( R2s )
    beh_full.append(behavioral_performance[:,4])
    area_full.append( np.zeros(resp_frac.shape[0])+l_nr )

np.set_printoptions(precision=3, suppress=True)


data_full = np.concatenate(data_full,axis=0)
data_full = CAgeneral.remove_allNaNrow(data_mat=data_full)
area_full = np.stack(area_full,axis=0)
beh_full = np.stack(beh_full,axis=0)

area_full = area_full[~np.isnan(beh_full)]
beh_full = beh_full[~np.isnan(beh_full)]

data_ww = [[] for _ in range(n_regr)]
for w in range(n_regr):
    data_ww[w] = np.full((n_recs,n_locs),np.NaN)
    for l in range(n_locs):
        data_ww[w][:,l] = data_all[l][:,w]

regr_names = ["Base","Decay","Task","LrnTsk"]

for l_nr,loc in enumerate(locations):
    print("\n{}".format(loc))
    print(np.nanmean(data_all[l_nr],axis=0))


fig = CAplot.init_figure(fig_size=(4.5,10))
for w in range(n_regr):
    ax = CAplot.plt.subplot2grid( (n_regr,1), (w,0) )
    for tp in range(X6.shape[0]):
        CAplot.bar( tp, X6[tp,w], e=0, width=0.8, edge="off", bar_color=(0.6,0.6,0.6,1.0) )

    # Finish panel layout
    CAplot.finish_panel( ax, title="", ylabel="w", xlabel="", legend="off", y_minmax=[0,1.2], y_step=[1,0], y_margin=0, y_axis_margin=0, x_minmax=[0,X6.shape[0]-0.99], x_step=[1,0], x_margin=0.55, x_axis_margin=0.55, despine=True)

# Finish figure layout and save
CAplot.finish_figure( filename="4g-FractionResponsiveNeurons-RegressionModel" )

# Plot break-down of model fit
fig = CAplot.init_figure(fig_size=(4.5,10))
l = 0
m = 1
xvalues = np.arange(6)
ytotal = np.zeros(6)
for w in range(n_regr):
    weight = data_all[l][m,w]
    yval = weight * X6[:,w]
    ytotal += yval
    ax = CAplot.plt.subplot2grid( (6,1), (w,0) )
    CAplot.plt.plot( xvalues, yval, color="#666666" )
    CAplot.finish_panel( ax, title="", ylabel="", xlabel="", legend="off", y_minmax=[0,0.61], y_step=[0.5,1], y_margin=0, y_axis_margin=0, x_minmax=[0,X6.shape[0]-0.99], x_step=[1,0], x_margin=0.01, x_axis_margin=0.02, despine=True)

ax = CAplot.plt.subplot2grid( (6,1), (4,0) )
CAplot.plt.plot( xvalues, ytotal, color="#666666" )
CAplot.finish_panel( ax, title="", ylabel="", xlabel="", legend="off", y_minmax=[0,0.61], y_step=[0.5,1], y_margin=0, y_axis_margin=0, x_minmax=[0,X6.shape[0]-0.99], x_step=[1,0], x_margin=0.01, x_axis_margin=0.02, despine=True)

ax = CAplot.plt.subplot2grid( (6,1), (5,0) )
yreal = data_mats_all[l][m,:]
CAplot.plt.plot( xvalues, yreal, color="#000000" )
CAplot.finish_panel( ax, title="", ylabel="", xlabel="", legend="off", y_minmax=[0,0.61], y_step=[0.5,1], y_margin=0, y_axis_margin=0, x_minmax=[0,X6.shape[0]-0.99], x_step=[1,0], x_margin=0.01, x_axis_margin=0.02, despine=True)

CAplot.finish_figure( filename="4g-FractionResponsiveNeurons-RegressionExample" )

fig = CAplot.init_figure(fig_size=(17.6,4.5))
for r_nr in range(n_regr):
    ax = CAplot.plt.subplot2grid( (1,n_regr), (0,r_nr) )

    # Plot learning line and stacked timepoints
    xvalues = np.arange(n_locs)

    # Plot individual data points
    df_per_mouse = pd.DataFrame(data_ww[r_nr])
    CAplot.sns.swarmplot(data=df_per_mouse, ax=ax, color="#000000", size=3, linewidth=1, edgecolor="None")

    # Plot mean bars + errorbars
    x_wd = np.mean(xvalues[1:]-xvalues[:-1]) * 0.8
    mean_data,sem_data,_ = CAgeneral.mean_sem( data_ww[r_nr], axis=0 )
    for nr,(x,y,e) in enumerate(zip( xvalues, mean_data, sem_data )):
        CAplot.bar(x, y, e, width=x_wd, edge="on", bar_color=np.array(CAplot.colors[nr]), sem_color='#000000')

    # Finish panel layout
    CAplot.finish_panel( ax, title="{}".format(regr_names[r_nr]), ylabel="Weight", xlabel="", legend="off", y_minmax=[0,0.31], y_step=[0.1,1], y_margin=0.01, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=None, x_margin=0.65, x_axis_margin=0.55, x_ticks=xvalues, x_ticklabels=locations, x_tick_rotation=45, despine=True, legendpos=0)

# Finish figure layout and save
CAplot.finish_figure( filename="4ED6b-FractionResponsiveNeurons-WeightsPerAreaAndRegressor" )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Weights in 2d color plot

fig = CAplot.init_figure(fig_size=(7,4.5))
LMmat = np.full((n_regr,n_locs),np.NaN)
for r_nr in range(n_regr):
    LMmat[r_nr,:],_,_ = CAgeneral.mean_sem( data_ww[r_nr], axis=0 )

im = CAplot.plt.imshow( LMmat, aspect="equal", cmap="Reds", vmin=0, vmax=0.2 )
CAplot.plt.colorbar(im, ticks=[0,0.2], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=6)
CAplot.finish_panel( CAplot.plt.gca(), title=None, ylabel=None, xlabel=None, legend="off", y_minmax=[-0.5,n_regr-0.5], y_step=[1,0], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,n_locs-0.5], x_step=[1,0], x_margin=0.1, x_axis_margin=0, y_ticks=np.arange(0,n_regr,1), y_ticklabels=regr_names, x_ticks=np.arange(0,n_locs,1), x_ticklabels=locations, despine=True )

# Finish figure layout and save
CAplot.finish_figure( filename="4h-FractionResponsiveNeurons-WeightsPerAreaAndRegressor-2Dplot" )


data_R2 = np.full((n_recs,n_locs),np.NaN)
for l in range(n_locs):
    data_R2[:,l] = R2_full[l][:]

# R2 in 1d bar plot
fig,ax = CAplot.init_figure_axes(fig_size=(6,5))

# Plot learning line and stacked timepoints
xvalues = np.arange(n_locs)

# Plot individual data points
df_per_mouse = pd.DataFrame(data_R2)
CAplot.sns.swarmplot(data=df_per_mouse, ax=ax, color="#aaaaaa", size=3, linewidth=1, edgecolor="None")

# Plot mean bars + errorbars
x_wd = np.mean(xvalues[1:]-xvalues[:-1]) * 0.8
mean_data,sem_data,_ = CAgeneral.mean_sem( data_R2, axis=0 )
for nr,(x,y,e) in enumerate(zip( xvalues, mean_data, sem_data )):
    CAplot.bar(x, y, e, width=x_wd, edge="on", bar_color="#000000", sem_color='#000000')

# Finish panel layout
CAplot.finish_panel( ax, ylabel="R2", xlabel="", legend="off", y_minmax=[0,1.1], y_step=[0.2,1], y_margin=0.01, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=None, x_margin=0.65, x_axis_margin=0.55, x_ticks=xvalues, x_ticklabels=locations, x_tick_rotation=45, despine=True, legendpos=0)

# Finish figure layout and save
CAplot.finish_figure( filename="Methods-FractionResponsiveNeurons-LinearModel-R2" )


m,e,n = CAstats.mean_sem(data_R2.ravel())
standarddev = np.nanstd(data_R2.ravel())

if shuffle_regr is None:
    print("Mean R2 full model across all is {} ± {} SD, n={}".format(m,standarddev,n ))
else:
    print("Mean dR2 {} across all is {} ± {} SEM, n={}".format( regr_names[shuffle_regr], m, e, n ))

print("\nDifference between dorsal and ventral areas")
locations = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
dorsal =    [2,3,4]
ventral =    [1,6,7,8]
fig = CAplot.init_figure(fig_size=(12,4.5))
for r_nr in range(n_regr):

    x = []
    for l in dorsal:
        x.append(data_all[l][:,r_nr])
    y = []
    for l in ventral:
        y.append(data_all[l][:,r_nr])
    x = np.concatenate(x,axis=0)
    y = np.concatenate(y,axis=0)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    U,p = scistats.mannwhitneyu( x, y )

    print("\n  Regressor {}:".format(regr_names[r_nr]))
    print("    - Mean weight dorsal = {} ({}) n={}".format(*CAgeneral.mean_sem(x)))
    print("    - Mean weight ventral = {} ({}) n={}".format(*CAgeneral.mean_sem(y)))
    CAstats.report_mannwhitneyu_test( x, y, n_indents=4, alpha=0.05, bonferroni=4)

    X = np.zeros((x.shape[0],2))
    Y = np.zeros((y.shape[0],2))+1
    X[:,0] = x
    Y[:,0] = y
    df_per_neuron = pd.DataFrame(np.concatenate([X,Y],axis=0), columns=["W","S"])

    ax = CAplot.plt.subplot2grid( (1,n_regr), (0,r_nr) )
    m,e,n = CAgeneral.mean_sem(x)
    CAplot.bar( 0, m, e )
    m,e,n = CAgeneral.mean_sem(y)
    CAplot.bar( 1, m, e )
    CAplot.sns.swarmplot(data=df_per_neuron, ax=ax, y="W", x="S", color="#aaaaaa", size=2, linewidth=1, edgecolor="None")
    CAplot.finish_panel( ax, ylabel="Weight", xlabel="Stream", legend="off", y_minmax=[0,0.32], y_step=[0.1,1], y_margin=0.01, y_axis_margin=0.0, x_minmax=[0,1], x_ticks=[0,1], x_ticklabels=["D", "V"], x_margin=0.75, x_axis_margin=0.55, despine=True)

CAplot.finish_figure( filename="4i-FractionResponsiveNeurons-WeightDorsalVsVentral", wspace=0.8, hspace=0.8 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
