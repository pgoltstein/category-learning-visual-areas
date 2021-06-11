#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30, 2018

@author: pgoltstein
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get all imports
import numpy as np
import os, glob
import warnings
from sklearn.cluster import KMeans
import sys
sys.path.append('../xx_analysissupport')
import CAplot, CAgeneral, CArec, CAstats
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

add_third_fake_cluster_as_control = False

max_n_cum_stimuli = 2

yrange = [0,0.701]
y_step = [0.2,1]

yrange2 = [0,0.501]
y_step2 = [0.2,1]

yrange3 = [-0.6,0.501]
y_step3 = [0.2,1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find imaging location files
locations = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
xlabels = ['Bs-tc', 'Bs-tc', 'Bs-tsk', 'Bs-tsk', 'Lrn-tsk', 'Lrn-tc']
n_locs = len(locations)
n_timepoints = len(xlabels)
data_dir = "../../data/p4_fractionresponsiveneurons"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load and process recordings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_mat_cumstm = []
data_mat_catdiff = []
data_mat_lrpn = []
data_all = []
area_all = []
data_mat_behperf = []
for l_nr,loc in enumerate(locations):

    # Load data from file
    filename = os.path.join(data_dir, 'data_frc_'+loc+'.npy')
    print("Loading data from: {}".format(filename))
    data_dict = np.load(filename, allow_pickle=True).item()
    CAplot.print_dict(data_dict["settings"])
    resp_frac = data_dict["data"]["fr_resampled"]
    catdiff = data_dict["data"]["catdiff_resampled_overall"]
    lrpn = data_dict["data"]["fr_lrpn_resampled"]
    behavioral_performance = data_dict["data"]["behavioral_performance"]

    # Add cumulative fraction of responsive neurons to list
    resp_frac = np.concatenate( [ resp_frac[:,:,:max_n_cum_stimuli-1], np.expand_dims( np.sum(resp_frac[:,:,max_n_cum_stimuli-1:], axis=2), axis=2 ) ], axis=2 )
    data_mat_cumstm.append( resp_frac )
    data_mat_catdiff.append( catdiff )
    data_mat_lrpn.append( lrpn )
    data_all.append( np.sum(resp_frac, axis=2) )
    area_all.append( np.zeros((resp_frac.shape[0],1))+l_nr )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cluter analysis of timepoint fraction responsive
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_all = np.concatenate( data_all, axis=0 )
area_all = np.concatenate( area_all, axis=0 )
area_all = CAgeneral.remove_allNaNrow(data_mat=area_all,selector_mat=data_all)
data_all = CAgeneral.remove_allNaNrow(data_mat=data_all)
data_all_nonan = np.array(data_all)
data_all_nonan = np.delete(data_all,3,axis=1)

print("size of data_all: {}".format(data_all.shape))

data_all_norm = np.zeros_like(data_all_nonan)

for y in range(data_all_norm.shape[0]):
    data_all_norm[y,:] = (data_all_nonan[y,:]-np.min(data_all_nonan[y,:])) / (np.max(data_all_nonan[y,:])-np.min(data_all_nonan[y,:]))


# -------- START TEST/CONTROL CODE --------

if add_third_fake_cluster_as_control:
    # ADD AREAS OF VENTRAL STREAM DOMINATED CLUSTER, BUT WITH REVERSED TIMECOURSE TO GET A 'THIRD' MAIN CLUSTER (of all area 'P' recordings) AND CONFIRM THAT CLUSTERING WORKS
    future_pred = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    third_cluster = data_all_norm[future_pred==0,:]
    third_cluster = third_cluster[:,::-1]
    data_all_norm = np.concatenate([data_all_norm,third_cluster],axis=0)

    third_cluster_data = data_all[future_pred==0,:]
    third_cluster_data = third_cluster_data[:,::-1]
    data_all = np.concatenate([data_all,third_cluster_data],axis=0)

    third_cluster_nonan = data_all_nonan[future_pred==0,:]
    third_cluster_nonan = third_cluster_nonan[:,::-1]
    data_all_nonan = np.concatenate([data_all_nonan,third_cluster_nonan],axis=0)

    third_areas = np.zeros_like(area_all) + 7
    third_areas = third_areas[future_pred==0]
    area_all = np.concatenate([area_all, third_areas])
    print(area_all)
    print(area_all.shape)
    n_clusters = 3
else:
    n_clusters = 2

# -------- END TEST / CONTROL CODE --------


n_max_clust = 8
inert_real = []
inert_shuffle = []
for K in range(2,n_max_clust+1):
    kmeans = KMeans(n_clusters=K)
    y_km = kmeans.fit_predict(data_all_norm)
    error = kmeans.inertia_

    error_sh = []
    for i in range(100):
        kmeans_sh = KMeans(n_clusters=K)
        data_all_sh = np.array(data_all_norm)
        for x in range(data_all_norm.shape[1]):
            np.random.shuffle(data_all_sh[:,x])
        y_km_sh = kmeans_sh.fit_predict(data_all_sh.reshape(data_all_norm.shape))
        error_sh.append( kmeans_sh.inertia_ )
    error_sh = np.mean(error_sh)
    print("K={:2.0f}, inertia_ = {:0.3f}, shuffled = {:0.3f}, difference = {:0.3f}".format(K,error,error_sh,error_sh-error))
    inert_real.append(error)
    inert_shuffle.append(error_sh)

inert_real = np.array(inert_real)
inert_shuffle = np.array(inert_shuffle)

kmeans = KMeans(n_clusters=n_clusters)
y_km = kmeans.fit_predict(data_all_norm)
print("\nCluster centers")
print(kmeans.cluster_centers_)
print("\nPrediction")
print(y_km)
error = kmeans.inertia_
print("inertia_ = {}".format(error))

loc_clust_frac = np.full((n_locs,n_clusters),np.NaN)
loc_clust_n = np.full((n_locs,n_clusters),np.NaN)
print("Cluster fraction per area")
for l_nr,loc in enumerate(locations):
    clust = np.zeros(n_clusters)
    for c_nr in range(n_clusters):
        clust[c_nr] = np.mean( y_km[area_all[:,0]==l_nr] == c_nr )
        loc_clust_frac[l_nr,c_nr] = clust[c_nr]
        loc_clust_n[l_nr,c_nr] = np.sum( y_km[area_all[:,0]==l_nr] == c_nr )
    print("Area {:3s} = {}".format(loc,clust) )


# Make 'learning' cluster be the second cluster (cluster # 2)
if np.nanmean(data_all[y_km==0,4]-data_all[y_km==0,2]) > np.nanmean(data_all[y_km==1,4]-data_all[y_km==1,2]):
    y_km = np.abs(y_km-1)


fig = CAplot.init_figure(fig_size=(12,6))
ax = CAplot.plt.subplot2grid( (1,2), (0,0) )
xvalues = np.arange(2,n_max_clust+1)
CAplot.plt.plot(xvalues,inert_real,color="#000000",label="real")
CAplot.plt.plot(xvalues,inert_shuffle,color="#888888",label="shuffle")
if add_third_fake_cluster_as_control:
    CAplot.finish_panel( ax, title="Model error (inertia)", ylabel="Inertia", xlabel="# clusters (K)", legend="on", y_minmax=[0,30], y_step=[5,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=[1,0], x_margin=0.55, x_axis_margin=0.55, x_ticks=xvalues[[0,-1]], despine=True, legendpos=0)
else:
    CAplot.finish_panel( ax, title="Model error (inertia)", ylabel="Inertia", xlabel="# clusters (K)", legend="on", y_minmax=[0,15], y_step=[2,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=None, x_margin=0.55, x_axis_margin=0.55, x_ticks=xvalues[[0,-1]], despine=True, legendpos=0)

ax = CAplot.plt.subplot2grid( (1,2), (0,1) )
xvalues = np.arange(2,n_max_clust+1)
CAplot.plt.plot(xvalues,inert_shuffle-inert_real,color="#000000",label="real")
if add_third_fake_cluster_as_control:
    CAplot.finish_panel( ax, title="Delta inertia", ylabel="Inertia difference", xlabel="# clusters (K)", legend="on", y_minmax=[0,8], y_step=[1,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=[1,0], x_margin=0.55, x_axis_margin=0.55, x_ticks=xvalues[[0,-1]], despine=True, legendpos=0)
else:
    CAplot.finish_panel( ax, title="Delta inertia", ylabel="Inertia difference", xlabel="# clusters (K)", legend="on", y_minmax=[0,4], y_step=[0.5,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=None, x_margin=0.55, x_axis_margin=0.55, x_ticks=xvalues[[0,-1]], despine=True, legendpos=0)
CAplot.finish_figure( filename="4b-FractionResponsiveNeurons-ClusterInertia" )

cluster_colors = ((0.4,0.4,0.4,1.0),(0.15,0.66,0.88,1.0),(0.88,0.66,0.15,1.0))
n_clust_timepoints = n_timepoints-1
xlabels_clust = xlabels[:3] + xlabels[4:]
fig = CAplot.init_figure(fig_size=(8*n_clusters,8))
for c_nr in range(n_clusters):
    ax = CAplot.plt.subplot2grid( (1,n_clusters), (0,c_nr) )

    # Plot learning line and stacked timepoints
    xvalues = np.arange(n_timepoints)
    CAplot.plot_timepoints( xvalues, data_all[y_km==c_nr,:], color=cluster_colors[c_nr] )
    CAplot.plot_learning_line(ax=ax, xpos=n_timepoints-2, yrange=[0,0.7])

    # Finish panel layout
    CAplot.finish_panel( ax, title="Cluster {}".format(c_nr), ylabel="Fraction", xlabel="", legend="off", y_minmax=[0,0.7], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=None, x_margin=0.55, x_axis_margin=0.55, x_ticks=xvalues, x_ticklabels=xlabels, x_tick_rotation=45, despine=True, legendpos=0)

# Finish figure layout and save
CAplot.finish_figure( filename="4c-FractionResponsiveNeurons-ClustersPerArea" )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show figure, fraction of recordings per cluster
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xvalues = np.arange(n_locs)

fig = CAplot.init_figure(fig_size=(12,6))
ax = CAplot.plt.subplot2grid( (1,2), (0,0) )

# Plot mean bars + errorbars
x_wd = np.mean(xvalues[1:]-xvalues[:-1]) * 0.8
for nr,(x,y1,y2) in enumerate(zip( xvalues,loc_clust_frac[:,0],loc_clust_frac[:,1])):
    CAplot.bar(x, y1, e=0, width=x_wd, edge="on", bar_color=cluster_colors[0], sem_color='#000000', bottom=0)
    CAplot.bar(x, y2, e=0, width=x_wd, edge="on", bar_color=cluster_colors[1], sem_color='#000000', bottom=y1)

# Finish panel layout
CAplot.finish_panel( ax, title="", ylabel="Fraction", xlabel="", legend="off", y_minmax=[0,1.1], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=None, x_margin=0.55, x_axis_margin=0.55, x_ticks=xvalues, x_ticklabels=locations, x_tick_rotation=45, despine=True, legendpos=0)

ax = CAplot.plt.subplot2grid( (1,2), (0,1) )
# Plot mean bars + errorbars
x_wd = np.mean(xvalues[1:]-xvalues[:-1]) * 0.8
for nr,(x,y1,y2) in enumerate(zip( xvalues,loc_clust_n[:,0],loc_clust_n[:,1])):
    CAplot.bar(x, y1, e=0, width=x_wd, edge="on", bar_color=cluster_colors[0], sem_color='#000000', bottom=0)
    CAplot.bar(x, y2, e=0, width=x_wd, edge="on", bar_color=cluster_colors[1], sem_color='#000000', bottom=y1)

# Finish panel layout
CAplot.finish_panel( ax, title="", ylabel="# mice", xlabel="", legend="off", y_minmax=[0,7.1], y_step=[1,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=None, x_margin=0.55, x_axis_margin=0.55, x_ticks=xvalues, x_ticklabels=locations, x_tick_rotation=45, despine=True, legendpos=0)

# Finish figure layout and save
CAplot.finish_figure( filename="4e-FractionResponsive-FractionPerCluster" )


print("\nDifference between dorsal and ventral areas")
locations = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
dorsal =    [2,3,4]
ventral =    [1,6,7,8]
d_clust = []
for l in dorsal:
    d_clust.append( y_km[area_all[:,0]==l] )
v_clust = []
for l in ventral:
    v_clust.append( y_km[area_all[:,0]==l] )

d_clust = np.concatenate(d_clust,axis=0)
v_clust = np.concatenate(v_clust,axis=0)
d_clust = d_clust[~np.isnan(d_clust)]
v_clust = v_clust[~np.isnan(v_clust)]

fig = CAplot.init_figure(fig_size=(10,7))
ax = CAplot.plt.subplot2grid( (1,2), (0,0) )

x,y1,y2,x_wd = 0,np.mean(d_clust==0),np.mean(d_clust==1),0.8
CAplot.bar(x, y1, e=0, width=x_wd, edge="on", bar_color=cluster_colors[0], sem_color='#000000', bottom=0)
CAplot.bar(x, y2, e=0, width=x_wd, edge="on", bar_color=cluster_colors[1], sem_color='#000000', bottom=y1)

x,y1,y2,x_wd = 1,np.mean(v_clust==0),np.mean(v_clust==1),0.8
CAplot.bar(x, y1, e=0, width=x_wd, edge="on", bar_color=cluster_colors[0], sem_color='#000000', bottom=0)
CAplot.bar(x, y2, e=0, width=x_wd, edge="on", bar_color=cluster_colors[1], sem_color='#000000', bottom=y1)

CAplot.finish_panel( ax, title="", ylabel="Fraction", xlabel="", legend="off", y_minmax=[0,1.1], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,1], x_ticks=[0,1], x_ticklabels=["D", "V"], x_margin=0.75, x_axis_margin=0.55, despine=True)

ax = CAplot.plt.subplot2grid( (1,2), (0,1) )

x,y1,y2,x_wd = 0,np.sum(d_clust==0),np.sum(d_clust==1),0.8
CAplot.bar(x, y1, e=0, width=x_wd, edge="on", bar_color=cluster_colors[0], sem_color='#000000', bottom=0)
CAplot.bar(x, y2, e=0, width=x_wd, edge="on", bar_color=cluster_colors[1], sem_color='#000000', bottom=y1)

x,y1,y2,x_wd = 1,np.sum(v_clust==0),np.sum(v_clust==1),0.8
CAplot.bar(x, y1, e=0, width=x_wd, edge="on", bar_color=cluster_colors[0], sem_color='#000000', bottom=0)
CAplot.bar(x, y2, e=0, width=x_wd, edge="on", bar_color=cluster_colors[1], sem_color='#000000', bottom=y1)

CAplot.finish_panel( ax, title="", ylabel="Count", xlabel="", legend="off", y_minmax=[0,17.1], y_step=[5,0], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,1], x_ticks=[0,1], x_ticklabels=["D", "V"], x_margin=0.75, x_axis_margin=0.55, despine=True)

CAplot.finish_figure( filename="4f-FractionResponsiveNeurons-Dorsal-vs-Ventral", wspace=0.8, hspace=0.8 )

print("\nSTATS")
print("Count dorsal/ventral stream areas")
CAstats.report_mean( d_clust, v_clust )
CAstats.report_chisquare_test( d_clust, v_clust )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show figure, all timepoints, cumulative for number of re-occurring tp's
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig = CAplot.init_figure(fig_size=(24,8))
panel_ids,n = CAplot.grid_indices( n_locs, n_columns=-5 )
for l_nr, py, px in zip( range(n_locs), panel_ids[0], panel_ids[1] ):
    ax = CAplot.plt.subplot2grid( n, (py,px) )

    # Plot learning line and stacked timepoints
    xvalues = np.arange(n_timepoints)
    CAplot.plot_timepoints( xvalues, np.sum(data_mat_cumstm[l_nr],axis=2), color=CAplot.colors[l_nr] )
    CAplot.plot_learning_line(ax=ax, xpos=n_timepoints-2, yrange=yrange)

    # Finish panel layout
    CAplot.finish_panel( ax, title=locations[l_nr], ylabel="Fraction", xlabel="", legend="off", y_minmax=yrange, y_step=y_step, y_margin=0.0, y_axis_margin=0.0, x_minmax=xvalues[[0,-1]], x_step=None, x_margin=0.55, x_axis_margin=0.55, x_ticks=xvalues, x_ticklabels=xlabels, x_tick_rotation=45, despine=True, legendpos=0)

# Finish figure layout and save
CAplot.finish_figure( filename="4ED6a-FractionResponsiveNeurons-AllAreas" )


# Show image with fraction of responsive neurons in color
X = []
seps = []
y = 0
for l_nr in range(n_locs):
    D = np.sum(data_mat_cumstm[l_nr],axis=2)
    yd = 0
    for m in range(D.shape[0]):
        if np.sum(~np.isnan(D[m,:])) > 0:
            X.append(D[m,:])
            yd += 1
    seps.append([y,y+yd-1.0])
    y += yd
X = np.stack(X,axis=0)
ylen = X.shape[0]

# Reverse y axis order
X = X[::-1,:]
for l_nr in range(n_locs):
    seps[l_nr][0] = (ylen-1) - seps[l_nr][0]
    seps[l_nr][1] = (ylen-1) - seps[l_nr][1]

minmax = 0.4
fig,ax = CAplot.init_figure_axes(fig_size=(6,20))
im = CAplot.plt.imshow( X, aspect="equal", cmap="Reds", vmin=0, vmax=minmax )

for l_nr in range(n_locs):
    CAplot.plt.plot([-1,-1],seps[l_nr],color='#000000',linestyle='-')
    CAplot.plt.text(-1.2, np.mean(seps[l_nr]), locations[l_nr], rotation=90, ha='right', va='center', size=CAplot.font_size["text"], color='#000000' )

CAplot.plt.colorbar(im, ticks=[0,minmax], fraction=0.1, pad=0.1, shrink=0.5, aspect=10).ax.tick_params(labelsize=8)
CAplot.finish_panel( CAplot.plt.gca(), title="Mean fraction per neuron", ylabel=None, xlabel="Timepoint", legend="off", y_minmax=[-0.5,X.shape[0]-0.5], y_margin=0.1, y_axis_margin=0, x_minmax=[-0.5,X.shape[1]-0.5], x_step=[1,0], x_margin=2.1, x_axis_margin=0, x_ticks=np.arange(0,X.shape[1],1), x_ticklabels=np.arange(0,X.shape[1],1), despine=True )
ax.get_yaxis().set_visible(False)

# Finish figure layout and save
CAplot.finish_figure( filename="4a-FractionResponsiveNeurons-2Dplot" )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
