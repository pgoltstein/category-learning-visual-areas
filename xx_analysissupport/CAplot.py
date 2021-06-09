#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr Aug 31, 2017

@author: pgoltstein
"""

########################################################################
### Imports
########################################################################

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob, sys
sys.path.append('../xx_analysissupport')

# Local
import CAgeneral, CArec


########################################################################
### Defaults
########################################################################

# settings for retaining pdf font
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

# Default settings
font_size = { "title": 12, "label": 11, "tick": 10, "text": 8, "legend": 8 }
location_coordinates = {"V1": [0,0], "LM": [-1,1], "AL": [1,1], "P": [-1.5,2], "POR": [-2.2,2], "LI": [-1,2], "RL": [2,2], "AM": [1,2], "PM": [0,2]}
location_edges = [["V1","LM"],["V1","AL"],["V1","PM"],["LM","AL"],["LM","PM"],["LM","P"],["LM","POR"],["LM","LI"],["AL","PM"],["AL","AM"],["AL","RL"]]

# seaborn color context
color_context = {   'axes.edgecolor': '#000000',
                    'axes.labelcolor': '#000000',
                    'boxplot.capprops.color': '#000000',
                    'boxplot.flierprops.markeredgecolor': '#000000',
                    'grid.color': '#000000',
                    'patch.edgecolor': '#000000',
                    'text.color': '#000000',
                    'xtick.color': '#000000',
                    'ytick.color': '#000000'}
sns.set_context("notebook")

cmap = matplotlib.cm.get_cmap('tab10')
colors = []
for x in np.arange(0,1,0.1): colors.append(cmap(x))
colors *= 36

########################################################################
### Functions
########################################################################


def init_figure( fig_size=(10,7), dpi=80, facecolor="w", edgecolor="w" ):
    # Convert fig size to inches (default is inches, fig_size argument is supposed to be in cm)
    inch2cm = 2.54
    fig_size = fig_size[0]/inch2cm,fig_size[1]/inch2cm
    with sns.axes_style(style="ticks",rc=color_context):
        fig = plt.figure(num=None, figsize=fig_size, dpi=dpi,
            facecolor=edgecolor, edgecolor=edgecolor)
        return fig

def init_figure_axes( fig_size=(10,7), dpi=80, facecolor="w", edgecolor="w" ):
    # Convert fig size to inches (default is inches, fig_size argument is supposed to be in cm)
    inch2cm = 2.54
    fig_size = fig_size[0]/inch2cm,fig_size[1]/inch2cm
    with sns.axes_style(style="ticks",rc=color_context):
        fig,ax = plt.subplots(num=None, figsize=fig_size, dpi=dpi,
            facecolor=facecolor, edgecolor=edgecolor)
        return fig,ax

def grid_indices( n_plots, n_columns ):
    """ Returns a list of x and y indices, as well as total plot number to conveniently index subplot2grid """
    if n_columns == -5:
        panel_ids_y = [0,0,0,0,1,1,1,1,1]
        panel_ids_x = [0,1,2,3,0,1,2,3,4]
        n_y=2
        n_x = 5
    else:
        panel_ids_y = np.floor(np.arange(n_plots)/n_columns).astype(int)
        panel_ids_x = np.mod(np.arange(n_plots),n_columns).astype(int)
        n_x = panel_ids_x.max()+1
        n_y = panel_ids_y.max()+1
    return (panel_ids_y,panel_ids_x),(n_y,n_x)

def area_graph_plot( locations, data_mat ):
    edge_space = 0.2
    cmap = matplotlib.cm.get_cmap('hot')
    mean_data,sem_data,_ = CAgeneral.mean_sem( data_mat, axis=0 )
    mean_data = (mean_data,np.min(mean_data)) / (np.max(mean_data)-np.min(mean_data))

    fig,ax = init_figure_axes(fig_size=(17.6,9))
    for loc in locations:
        x,y = location_coordinates[loc]
        plt.text(x, y, loc, rotation=0, ha='center', va='center', size=20, color='#000000' )
    for a1,a2 in location_edges:
        x2,y2 = location_coordinates[a2]
        x1,y1 = location_coordinates[a1]
        if y1 != y2:
            y1 += edge_space
            y2 -= edge_space
        xd = np.abs(x2-x1)*edge_space
        x1 += (-1*(x2<x1))*xd + (1*(x2>x1))*xd
        x2 += (-1*(x2>x1))*xd + (1*(x2<x1))*xd
        ax.annotate( "", xy=(x2,y2), xytext=(x1,y1), va="top", ha="left", size=20, color='#777777', arrowprops=dict( arrowstyle="<->", linewidth=1, edgecolor="#777777" ) )
    ax.set_ylim(-0.5,2.7)
    ax.set_xlim(-3,4)


def plot_learning_line(ax=None, xpos=2, yrange=[0,1], text_size=None):
    """Plots line between baseline and learning time points """
    if text_size is None: text_size=font_size['text']
    if yrange is None:
        yrange = ax.get_ylim()
    plt.plot([xpos-0.5, xpos-0.5],[yrange[0], yrange[1]*0.88],color='#777777',linestyle='--')
    plt.text(xpos-0.5, yrange[1]*0.9, 'Learning', rotation=0, ha='center', va='bottom', size=text_size, color='#777777' )


def finish_panel( ax, title="", ylabel="", xlabel="", legend="off", y_minmax=None, y_step=None, y_margin=0.02, y_axis_margin=0.01, x_minmax=None, x_step=None, x_margin=0.55, x_axis_margin=0.55, x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, x_tick_rotation=0, tick_size=None, label_size=None, title_size=None, legend_size=None, despine=True, legendpos=0):
    """ Finished axis formatting of an individual plot panel """
    if tick_size is None: tick_size=font_size['tick']
    if label_size is None: label_size=font_size['label']
    if title_size is None: title_size=font_size['title']
    if legend_size is None: legend_size=font_size['legend']

    # Set limits and trim spines
    if y_minmax is not None:
        ax.set_ylim(y_minmax[0]-y_margin,y_minmax[1]+y_margin)
    if x_minmax is not None:
        ax.set_xlim(x_minmax[0]-x_margin,x_minmax[-1]+x_margin)
    if despine:
        sns.despine(ax=ax, offset=0, trim=True)

    # Set tickmarks and labels
    if x_ticklabels is not None:
        plt.xticks( x_ticks, x_ticklabels, rotation=x_tick_rotation, fontsize=tick_size )
    elif x_minmax is not None and x_step is not None:
        plt.xticks( np.arange(x_minmax[0],x_minmax[1]+0.0000001,x_step[0]), suck_on_that_0point0(x_minmax[0], x_minmax[1]+0.0000001, step=x_step[0], format_depth=x_step[1]), rotation=x_tick_rotation, fontsize=tick_size )
    if y_ticklabels is not None:
        plt.yticks( y_ticks, y_ticklabels, fontsize=tick_size )
    elif y_minmax is not None and y_step is not None:
        plt.yticks( np.arange(y_minmax[0],y_minmax[1]+0.0000001,y_step[0]), suck_on_that_0point0(y_minmax[0], y_minmax[1]+0.0000001, step=y_step[0], format_depth=y_step[1]), rotation=0, fontsize=tick_size )

    ax.tick_params(length=3)

    # Set spine limits
    if y_minmax is not None:
        ax.spines['left'].set_bounds( y_minmax[0]-y_axis_margin, y_minmax[1]+y_axis_margin )
    if x_minmax is not None:
        ax.spines['bottom'].set_bounds( x_minmax[0]-x_axis_margin, x_minmax[1]+x_axis_margin )

    # Add title and legend
    if title != "":
        plt.title(title, fontsize=title_size)
    if ylabel != "":
        plt.ylabel(ylabel, fontsize=label_size)
    if xlabel != "":
        plt.xlabel(xlabel, fontsize=label_size)
    if legend == "on":
        lgnd = plt.legend(loc=legendpos, fontsize=legend_size, ncol=1, frameon=True)
        lgnd.get_frame().set_facecolor('#ffffff')

def finish_figure( filename=None, path=None, wspace=None, hspace=None ):
    """ Finish up layout and save to ~/figures"""
    plt.tight_layout()
    if wspace is not None or hspace is not None:
        if wspace is None: wspace = 0.6
        if hspace is None: hspace = 0.8
        plt.subplots_adjust( wspace=wspace, hspace=hspace )
    if filename is not None:
        if path is None:
            plt.savefig('../../figureout/'+filename+'.pdf', transparent=True)
        else:
            plt.savefig(path+filename+'.pdf', transparent=True)

def get_max_tm_values( tm, bs, stimulus_ids ):
    """ Calculates a 2d grid with psths at the according places """
    unique_stimulus_ixs = np.unique(stimulus_ids)
    max_values = np.zeros((tm.shape[1],len(unique_stimulus_ixs)))
    for s_ix,s in enumerate(unique_stimulus_ixs):
        if bs is not None:
            max_values[:,s_ix] = np.mean(tm[stimulus_ids==s,:],axis=0) - \
                                 np.mean(bs[stimulus_ids==s,:],axis=0)
        else:
            max_values[:,s_ix] = np.mean(tm[stimulus_ids==s,:],axis=0)
    return max_values.max(axis=1)

def get_max_psth_values( psth, bs, stimulus_ids ):
    """ Calculates a 2d grid with psths at the according places """
    unique_stimulus_ixs = np.unique(stimulus_ids)
    max_values = np.zeros((psth.shape[1],len(unique_stimulus_ixs)))
    for s_ix,s in enumerate(unique_stimulus_ixs):
        mean_curve,sem_curve,_ = \
            CAgeneral.mean_sem(psth[stimulus_ids==s,:,:],axis=0)
        if bs is not None:
            max_values[:,s_ix] = np.max(mean_curve+sem_curve,axis=1) - \
                                    np.mean(bs[stimulus_ids==s,:],axis=0)
        else:
            max_values[:,s_ix] = np.max(mean_curve+sem_curve,axis=1)
    return max_values.max(axis=1)

def psth_xvalues(sf, frame_range, psth_binning=1):
    """ Returns the x-axis values of a peri-stimulus time histogram """
    return (np.arange( frame_range[0]/psth_binning,
                      frame_range[1]/psth_binning ) / sf) *psth_binning

def plot_psth_grid( ax, xvalues, psth, bs, std, stimulus_ids, stim_id_grid,
        y_scale, prototype_stimuli, category_stimuli ):
    """ Plots an entire grid with psths at the according places """
    x_scale = 1.2*(xvalues[-1]-xvalues[0])
    n_spatialf,n_direction = stim_id_grid.shape
    x_plotted = []
    for sf in range(n_spatialf):
        for dir in range(n_direction):
            stim_id = stim_id_grid[sf,dir]
            if np.isfinite(stim_id):
                color = "#999999"
                for c_dir,c_sf in zip( category_stimuli["left"]["dir"],
                        category_stimuli["left"]["spf"] ):
                    if c_dir==CArec.directions[dir] and c_sf==CArec.spatialfs[sf]:
                        color = "#60C3DB"
                for c_dir,c_sf in zip( category_stimuli["right"]["dir"],
                        category_stimuli["right"]["spf"] ):
                    if c_dir==CArec.directions[dir] and c_sf==CArec.spatialfs[sf]:
                        color = "#F60951"
                if prototype_stimuli["left"]["dir"]==CArec.directions[dir] and \
                    prototype_stimuli["left"]["spf"]==CArec.spatialfs[sf]:
                    color = "#000099"
                if prototype_stimuli["right"]["dir"]==CArec.directions[dir] and \
                    prototype_stimuli["right"]["spf"]==CArec.spatialfs[sf]:
                    color = "#990000"
                stim_ids = np.where(stimulus_ids==stim_id)[0]
                mean_curve,sem_curve,_ = \
                    CAgeneral.mean_sem(psth[stim_ids,:],axis=0)
                if bs is not None:
                    mean_curve = mean_curve - np.mean(bs[stim_ids])
                if std is not None:
                    std_curve = np.zeros_like(mean_curve) + \
                        ( 3 * np.mean(std[stim_ids]) )
                else:
                    std_curve = None
                psth_in_grid( gx=dir, gy=sf, x=xvalues, y=mean_curve,
                    e=sem_curve, x_scale=x_scale, y_scale=y_scale,
                    color=color, std=std_curve )
                x_plotted.append(dir)
    x_left_ticklabels = (min(x_plotted)*x_scale)+xvalues[0]-(0.2*x_scale)
    for y in range(n_spatialf):
        plt.text(x_left_ticklabels, y*y_scale, "{}".format(CArec.spatialfs[y]),
            rotation=0, ha='right', va='center', size=6, color='#000000' )
    y_bottom_ticklabels = -0.2*y_scale
    for x in range(n_direction):
        if x >= min(x_plotted) and x <= max(x_plotted):
            plt.text((x*x_scale)+np.median(xvalues), y_bottom_ticklabels,
                "{}".format(CArec.directions[x]), rotation=0,
                ha='center', va='top', size=6, color='#000000' )
    ax.set_ylim(-0.6*y_scale,(n_spatialf*y_scale)+(0.2*y_scale))
    # ax.set_xlim(xvalues[0]-0.1,xvalues[-1]+0.1)


def psth_in_grid( gx, gy, x, y, e=None, x_scale=1, y_scale=1,
                    color="#0000ff", std=None ):
    # if std is not None:
    #     plt.plot( (gx*x_scale)+x, (gy*y_scale)+std, color="#000000" )
    plt.plot( (gx*x_scale)+x, (gy*y_scale)+y, color=color )
    if e is not None:
        plt.fill_between( (gx*x_scale)+x, (gy*y_scale)+(y-e), (gy*y_scale)+(y+e), facecolor=color, alpha=0.4, linewidth=0 )

def plot_psth_and_model( neuron_nr, rec, model, category_stimuli, prototype_stimuli, predictor_groups, y_scale=0.2, frame_range=[-15,60], frame_range_bs=[-15,0] ):

    # Calculate parameters and indices
    model.Y = rec.spikes[:,neuron_nr].reshape((rec.n_frames,1))
    data_mat = model.Y
    category_ids = rec.get_trial_category_id(category_stimuli)
    prototype_ids = rec.get_trial_category_id(prototype_stimuli)
    frame_ixs = rec.vis_on_frames
    stim_ids = rec.stimuli
    stim_id_grids = rec.get_1d_stimulus_ix_in_2d_grid()
    xvalues = psth_xvalues(rec.sf, frame_range )
    x_scale = 1.2*(xvalues[-1]-xvalues[0])

    # Calculate PSTH
    psth = CAgeneral.psth( data_mat, frame_ixs, frame_range )
    bs = CAgeneral.tm( data_mat, frame_ixs, frame_range_bs, return_peak=False, include_bs=False )

    # Plot individual stimuli
    ax = [None,None]
    ax[0] = plt.subplot2grid((1,2),(0,0))
    plot_psth_grid( ax[0], xvalues, psth[:,0,:], bs[:,0], None, stim_ids, stim_id_grids, y_scale, prototype_stimuli, category_stimuli )
    plt.axis('off')

    # Plot regressor PSTHs and weight-kernels
    ax[1] = plt.subplot2grid((1,2),(0,1))
    y_cnt = 0
    for group_name,group_members in predictor_groups:
        if not "lick" in group_name.lower():
            x_cnt = 0
            for name in group_members:
                if name in model.kernels:
                    frame_ixs = model.indices[name]
                    psth = CAgeneral.psth( data_mat, frame_ixs, frame_range )
                    bs = CAgeneral.tm( data_mat, frame_ixs, frame_range_bs, return_peak=False, include_bs=False )
                    y,e,_ = CAgeneral.mean_sem(psth[:,0,:],axis=0)
                    psth_in_grid( x_cnt, y_cnt, xvalues, y, e=e, x_scale=x_scale, y_scale=y_scale, color="#000000", std=None )

                    weights = np.nanmean(model.weights[name][:,:,neuron_nr],axis=0)
                    kernel_weights = model.calculate_weightkernel( model.kernels[name], weights )
                    kernel_xvalues = model.kernels[name].alltimes

                    psth_in_grid( x_cnt, y_cnt, kernel_xvalues, kernel_weights, e=None, x_scale=x_scale, y_scale=y_scale, color="#aa0000", std=None )

                    plt.text( (x_cnt*x_scale)+np.median(xvalues), (y_cnt*y_scale)-(0.05*y_scale), name, rotation=0, ha='center', va='top', size=6, color='#000000' )
                    x_cnt += 1
            y_cnt -= 1
    plt.axis('off')
    return ax


def scatter( x_data, y_data, color="#444444", size=10, edge=1, marker="o", label=None ):
    for nr,(x,y) in enumerate(zip(x_data,y_data)):
        if label is not None and nr == 0:
            plt.plot( x, y, color="None", markerfacecolor=None, markersize=size, markeredgewidth=edge, marker=marker, markeredgecolor=color, label=label )
        else:
            plt.plot( x, y, color="None", markerfacecolor=None, markersize=size, markeredgewidth=edge, marker=marker, markeredgecolor=color )

def plot_bars(xvalues, mdata, sdata, cdata):
    for x,y,e in zip( xvalues, mdata, sdata ):
        bar( x, y, e, edge="on", bar_color=np.array(cdata[x]), sem_color='#000000' )

def plot_double_bars(xvalues, mdata, sdata, cdata, mdata2, sdata2, cdata2):
    for x,y,e in zip( xvalues, mdata, sdata ):
        bar( x, y, e, edge="on", bar_color=np.array(cdata[x]), sem_color='#000000' )
    for x,y,e in zip( xvalues, mdata2, sdata2 ):
        bar( x, -1*y, e, edge="on", bar_color=0.7*np.array(cdata2[x]), sem_color='#000000' )

def plot_double_side_bars(xvalues, mdata, sdata, cdata, mdata2, sdata2, cdata2):
    for x,y,e in zip( xvalues, mdata, sdata ):
        bar( x*2, y, e, edge="on", bar_color=np.array(cdata[x]), sem_color='#000000' )
    for x,y,e in zip( xvalues, mdata2, sdata2 ):
        bar( (x*2)+1, y, e, edge="on", bar_color=np.array(cdata2[x]), sem_color='#000000' )

def plot_bootstrapped_paired_data( xvalues, paired_data, marker="o", size=5, line_color=None, plot_indiv=False, ci=95):
    marker_list = color if type(marker) is list else [marker,]*2
    for x,(data1,data2) in enumerate(paired_data):
        b1 = CAgeneral.bootstrapped(data1)
        b2 = CAgeneral.bootstrapped(data2)
        bar( x-0.21, b1.mean, e=b1.ci95 if ci==95 else b1.ci99, width=0.4, edge="on", bar_color="#666666", sem_color='#000000' )
        bar( x+0.21, b2.mean, e=b2.ci95 if ci==95 else b2.ci99, width=0.4, edge="on", bar_color="#333333", sem_color='#000000' )
        if plot_indiv:
            for y1,y2 in zip(data1,data2):
                plt.plot( x-0.21, y1, color="None", marker=marker_list[0], markerfacecolor=None, markersize=size, markeredgewidth=1, markeredgecolor="#888888" )
                plt.plot( x+0.21, y2, color="None", marker=marker_list[1], markerfacecolor=None, markersize=size, markeredgewidth=1, markeredgecolor="#888888" )
                plt.plot( [x-0.21,x+0.21], [y1,y2], color="#888888", marker=None )

def plot_bootstrapped_lsg_data( xvalues, lsg_data, color="#888888", marker="o", size=5, line_color=None, plot_indiv=False, ci=95):
    marker_list = color if type(marker) is list else [marker,]*3
    color_list = color if type(color) is list else [color,]*3
    line_color = color_list[0] if line_color is None else line_color
    for x,(dataL,dataS,dataG) in enumerate(lsg_data):
        bL = CAgeneral.bootstrapped(dataL)
        bS = CAgeneral.bootstrapped(dataS)
        bG = CAgeneral.bootstrapped(dataG)
        bar( x-0.21, bL.mean, e=bL.ci95 if ci==95 else bL.ci99, width=0.4, edge="on", bar_color="#883333", sem_color='#000000', bottom=bS.mean )
        bar( x+0.21, bG.mean, e=bG.ci95 if ci==95 else bG.ci99, width=0.4, edge="on", bar_color="#338833", sem_color='#000000', bottom=bS.mean )
        bar( x, bS.mean, e=bS.ci95 if ci==95 else bS.ci99, width=0.8, edge="on", bar_color="#666666", sem_color='#000000', error_width=0.25 )
        if plot_indiv:
            for y1,y2,y3 in zip(dataL,dataS,dataG):
                plt.plot( x-0.21, y1+bS.mean, color="None", marker=marker_list[0], markerfacecolor=None, markersize=size, markeredgewidth=1, markeredgecolor="#dd8888")
                plt.plot( x+0.21, y3+bS.mean, color="None", marker=marker_list[1], markerfacecolor=None, markersize=size, markeredgewidth=1, markeredgecolor="#88dd88" )
                plt.plot( [x-0.21,x+0.21], [y1+bS.mean,y3+bS.mean], color=line_color, marker=None )
                plt.plot( x, y2, color="None", marker=marker_list[1], markerfacecolor=None, markersize=size, markeredgewidth=1, markeredgecolor="#888888" )

def bar( x, y, e, width=0.8, edge="off", bar_color='#000000', sem_color='#000000', label=None, bottom=0, error_width=0.5 ):
    error_halfwidth = 0.5 * error_width * width
    if type(e) is list or type(e) is tuple:
        # Two sided confidence interval
        plt.plot( [x,x], [e[0]+bottom,e[1]+bottom], color=sem_color, linewidth=1 )
        plt.plot( [x-error_halfwidth,x+error_halfwidth], [e[1]+bottom,e[1]+bottom], color=sem_color, linewidth=1 )
        plt.plot( [x-error_halfwidth,x+error_halfwidth], [e[0]+bottom,e[0]+bottom], color=sem_color, linewidth=1 )
    elif e > 0:
        # One sided errorbars
        ye = y+bottom
        if y < 0:
            plt.plot( [x,x], [ye,ye-e], color=sem_color, linewidth=1 )
            plt.plot( [x-error_halfwidth,x+error_halfwidth], [ye-e,ye-e], color=sem_color, linewidth=1 )
        else:
            plt.plot( [x,x], [ye,ye+e], color=sem_color, linewidth=1 )
            plt.plot( [x-error_halfwidth,x+error_halfwidth], [ye+e,ye+e], color=sem_color, linewidth=1 )
    edgecolor,lw = (sem_color,1) if "on" in edge.lower() else ('None',0)
    if label is None:
        plt.bar( x-(0.5*width), y, width, color=bar_color, edgecolor=edgecolor, linewidth=lw, align='edge', bottom=bottom )
    else:
        plt.bar( x-0.(0.5*width), y, width, color=bar_color, edgecolor=edgecolor, linewidth=lw, align='edge', label=label, bottom=bottom )

def line( x, y, e=None, line_color='#000000', line_width=1, sem_color=None, shaded=False, label=None ):
    if e is not None:
        if shaded:
            if sem_color is None:
                sem_color = line_color
            plt.fill_between( x, y-e, y+e, facecolor=sem_color, alpha=0.4, linewidth=0 )
        else:
            if sem_color is None:
                sem_color = '#000000'
            for xx,yy,ee in zip(x,y,e):
                plt.plot( [xx,xx], [yy-ee,yy+ee], color=sem_color, linewidth=1 )
                plt.plot( [xx-0.2,xx+0.2], [yy-ee,yy-ee], color=sem_color, linewidth=1 )
                plt.plot( [xx-0.2,xx+0.2], [yy+ee,yy+ee], color=sem_color, linewidth=1 )
    if label is None:
        plt.plot( x, y, color=line_color, linewidth=line_width )
    else:
        plt.plot( x, y, color=line_color, linewidth=line_width, label=label )

def line_ci( x, y, upper_ci=None, lower_ci=None, line_color='#000000', line_width=1, label=None ):
    if upper_ci is not None and lower_ci is not None:
            plt.fill_between( x, lower_ci, upper_ci, facecolor=line_color, alpha=0.4, linewidth=0 )
    if label is None:
        plt.plot( x, y, color=line_color, linewidth=line_width )
    else:
        plt.plot( x, y, color=line_color, linewidth=line_width, label=label )

def redraw_markers( collections, marker_list, color_list, size=5,
        reduce_x_width=1, x_offset=None  ):
    # collections = ax.collections
    # ax.cla()
    for nr,col in enumerate(collections):
        for x,y in col.get_offsets():
            if x_offset is None:
                plt.plot( nr+(reduce_x_width*(x-nr)), y,
                         color="None", marker=marker_list[nr],
                         markerfacecolor=None, markersize=size,
                         markeredgewidth=1, markeredgecolor=color_list[nr] )
            else:
                plt.plot( nr+(reduce_x_width*(x-nr))+x_offset[nr], y,
                         color="None", marker=marker_list[nr],
                         markerfacecolor=None, markersize=size,
                         markeredgewidth=1, markeredgecolor=color_list[nr] )


def redraw_paired_markers( collections, marker, color, line_color="#aaaaaa", size=5 ):
    marker_list = color if type(marker) is list else [marker,]*2
    color_list = color if type(color) is list else [color,]*2
    for collection1,collection2 in zip(collections[::2],collections[1::2]):
        for (x1,y1),(x2,y2) in zip(collection1.get_offsets(),collection2.get_offsets()):
            plt.plot( x1, y1, color="None", marker=marker_list[0], markerfacecolor=None, markersize=size, markeredgewidth=1, markeredgecolor=color_list[0] )
            plt.plot( x2, y2, color="None", marker=marker_list[1], markerfacecolor=None, markersize=size, markeredgewidth=1, markeredgecolor=color_list[1] )
            plt.plot( [x1,x2], [y1,y2], color=line_color, marker=None )


def suck_on_that_0point0( start, stop, step=1, format_depth=1 ):
    values = []
    for i in np.arange( start, stop, step ):
        if i == 0:
            values.append('0')
        else:
            values.append('{:0.{dpt}f}'.format(i,dpt=format_depth))
    return values

def print_dict( d, indent=0, max_items=5 ):
    """ Functions prints the contents of a dictionary in a hierarchical way. Uses a recursive procedure. For numpy arrays it only displays the size of the array. For lists, only the length if the list is longer than 5 items.
        - Inputs -
        d:        dictionary
        indent:   current indent (mostly for internal use)
    """
    for k,v in d.items():
        if isinstance(v,dict):
            print("{}{}:".format("  "*indent,k))
            print_dict(v,indent+1)
        else:
            if isinstance(v,np.ndarray):
                if len(v)>max_items:
                    v = "Numpy array of shape {}".format(v.shape)
                else:
                    v = "Numpy array: {}".format(v)
            elif isinstance(v,list):
                if len(v) > max_items:
                    v = "List of length {}".format(len(v))
                elif isinstance(v[0],dict):
                    print("{}{}: <list of length {}>".format("  "*indent,k,len(v)))
                    indent += 1
                    for nr,vv in enumerate(v):
                        print("{}Item [{}]:".format("  "*indent,nr))
                        if vv is None:
                            print("{}".format("  "*indent,vv))
                        else:
                            print_dict(vv,indent+1)
                    continue
            print("{}{}: {}".format("  "*indent,k,v))

def print_cat( cat_dict ):
    for k, v in cat_dict.items():
        print('--{}--'.format(k))
        for kx, vx in v.items():
            print('  {} : {}'.format(kx,vx))

def plot_timepoints( xvalues, data_mouse_tp, color="#444444"):
    """ Plots timepoints in one panel """
    n_mice,n_timepoints = data_mouse_tp.shape

    # Plot individual mouse data points
    for m_nr in range(n_mice):
        yvalues_m = data_mouse_tp[m_nr,:]
        xvalues_m = xvalues[~np.isnan(yvalues_m)]
        yvalues_m = yvalues_m[~np.isnan(yvalues_m)]
        plt.plot( xvalues_m, yvalues_m,
            color='#aaaaaa', linestyle='-', linewidth=1 )

    # Plot mean bars + errorbars
    x_wd = np.mean(xvalues[1:]-xvalues[:-1]) * 0.8
    mean_data,sem_data,_ = CAgeneral.mean_sem( data_mouse_tp, axis=0 )
    for x,y,e in zip( xvalues, mean_data, sem_data ):
        bar(x, y, e, width=x_wd, edge="on", bar_color=np.array(color), sem_color='#000000')

def plot_paired_timepoints( xvalues, data_mouse_tp_pair, color=(0.3,0.3,0.3,1.0) ):
    """ Plots timepoints in one panel """
    n_mice,n_timepoints,_ = data_mouse_tp_pair.shape

    # Plot individual mouse data points
    for m_nr in range(n_mice):
        yvalues_m0 = data_mouse_tp_pair[m_nr,:,0]
        yvalues_m1 = data_mouse_tp_pair[m_nr,:,1]
        xvalues_m0 = xvalues[~np.isnan(yvalues_m0)]
        xvalues_m1 = xvalues[~np.isnan(yvalues_m1)]
        yvalues_m0 = yvalues_m0[~np.isnan(yvalues_m0)]
        yvalues_m1 = yvalues_m1[~np.isnan(yvalues_m1)]
        plt.plot( xvalues_m0, yvalues_m0,
            color='#aaaaaa', linestyle='-', linewidth=1 )
        plt.plot( xvalues_m1, yvalues_m1,
            color='#888888', linestyle='-', linewidth=1 )

    # Plot mean bars + errorbars
    x_wd = np.mean(xvalues[1:]-xvalues[:-1])/2.2
    mean_data0,sem_data0,_ = CAgeneral.mean_sem( data_mouse_tp_pair[:,:,0], axis=0 )
    mean_data1,sem_data1,_ = CAgeneral.mean_sem( data_mouse_tp_pair[:,:,1], axis=0 )
    for x,y,e in zip( xvalues-(0.52*x_wd), mean_data0, sem_data0 ):
        bar(x, y, e, width=x_wd, edge="on", bar_color=np.array(color), sem_color='#000000')
    for x,y,e in zip( xvalues+(0.52*x_wd), mean_data1, sem_data1 ):
        bar(x, y, e, width=x_wd, edge="on", bar_color=np.array(color)*0.5, sem_color='#000000')

def plot_opposing_timepoints( xvalues, data_mouse_tp_pair, color=(0.3,0.3,0.3,1.0) ):
    """ Plots timepoints in one panel """
    n_mice,n_timepoints,_ = data_mouse_tp_pair.shape

    # Plot individual mouse data points
    for m_nr in range(n_mice):
        yvalues_m0 = data_mouse_tp_pair[m_nr,:,0]
        yvalues_m1 = data_mouse_tp_pair[m_nr,:,1]
        xvalues_m0 = xvalues[~np.isnan(yvalues_m0)]
        xvalues_m1 = xvalues[~np.isnan(yvalues_m1)]
        yvalues_m0 = yvalues_m0[~np.isnan(yvalues_m0)]
        yvalues_m1 = yvalues_m1[~np.isnan(yvalues_m1)]
        plt.plot( xvalues_m0, yvalues_m0 * -1.0,
            color='#aaaaaa', linestyle='-', linewidth=1 )
        plt.plot( xvalues_m1, yvalues_m1,
            color='#aaaaaa', linestyle='-', linewidth=1 )

    # Plot mean bars + errorbars
    x_wd = np.mean(xvalues[1:]-xvalues[:-1]) * 0.8
    mean_data0,sem_data0,_ = CAgeneral.mean_sem( data_mouse_tp_pair[:,:,0], axis=0 )
    mean_data1,sem_data1,_ = CAgeneral.mean_sem( data_mouse_tp_pair[:,:,1], axis=0 )
    for x,y,e in zip( xvalues, mean_data0 * -1.0, sem_data0 ):
        bar(x, y, e, width=x_wd, edge="on", bar_color=np.array(color), sem_color='#000000')
    for x,y,e in zip( xvalues, mean_data1, sem_data1 ):
        bar(x, y, e, width=x_wd, edge="on", bar_color=np.array(color), sem_color='#000000')

def plot_stacked_timepoints(xvalues, data_mouse_tp_stack, color="#444444"):
    """ Plots cumulatively stacked timepoints in one panel """
    n_mice,n_timepoints,n_stack = data_mouse_tp_stack.shape

    # Plot cumulative stack data per mouse and timepoint
    for m_nr in range(n_mice):
        yvalues_m = data_mouse_tp_stack[m_nr,:,:].sum(axis=1)
        xvalues_m = xvalues[~np.isnan(yvalues_m)]
        yvalues_m = yvalues_m[~np.isnan(yvalues_m)]
        plt.plot( xvalues_m, yvalues_m, color='#aaaaaa', linestyle='-', linewidth=1, marker='o', markerfacecolor=None, markersize=2, markeredgewidth=1, markeredgecolor='#aaaaaa' )

    # Plot stacked bars
    x_wd = np.mean(xvalues[1:]-xvalues[:-1]) * 0.8
    bottom_data = np.zeros((n_timepoints,))
    for stck in range(n_stack):
        mean_data,sem_data,_ = CAgeneral.mean_sem( data_mouse_tp_stack[:,:,stck], axis=0 )
        # Plot bars + errorbars
        for x,y,e,b in zip( xvalues, mean_data, sem_data, bottom_data ):
            bar( x, y, e, width=x_wd, edge="on", bar_color=(stck/(n_stack-1))*np.array(color), sem_color='#000000', bottom=b )
        bottom_data += mean_data
