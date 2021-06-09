#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Saturday 24 Oct 2020

@author: pgoltstein
"""

import os, glob, sys
sys.path.append('../xx_analysissupport')
import matplotlib.pyplot as plt
import auxrec
import vidrec
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns
import cv2
from tqdm import tqdm

savefig = "../../figureout/"
base_data_path = "../../data/p6b_mouthtracking"

# settings for retaining pdf font
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

# Default settings
font_size = { "title": 10, "label": 9, "tick": 9, "text": 8, "legend": 8 }

# seaborn color context (all to black)
color_context = {   'axes.edgecolor': '#000000',
                    'axes.labelcolor': '#000000',
                    'boxplot.capprops.color': '#000000',
                    'boxplot.flierprops.markeredgecolor': '#000000',
                    'grid.color': '#000000',
                    'patch.edgecolor': '#000000',
                    'text.color': '#000000',
                    'xtick.color': '#000000',
                    'ytick.color': '#000000'}


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Arguments

mice = ["V05","V08"]
n_imaging_planes = 4
example_trial = 20
example_frame = 5

all_directions = np.arange(0,360,20)
all_spatialfs = np.array([0.04,0.06,0.08,0.12,0.16,0.24])
n_directions = all_directions.shape[0]
n_spatialfs = all_spatialfs.shape[0]

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# functions

def mean_sem( datamat, axis=0 ):
    mean = np.nanmean(datamat,axis=axis)
    n = np.sum( ~np.isnan( datamat ), axis=axis )
    sem = np.nanstd( datamat, axis=axis ) / np.sqrt( n )
    return mean,sem,n

def init_figure_axes( fig_size=(10,7), dpi=80, facecolor="w", edgecolor="w" ):
    # Convert fig size to inches (default is inches, fig_size argument is supposed to be in cm)
    inch2cm = 2.54
    fig_size = fig_size[0]/inch2cm,fig_size[1]/inch2cm
    with sns.axes_style(style="ticks",rc=color_context):
        fig,ax = plt.subplots(num=None, figsize=fig_size, dpi=dpi,
            facecolor=facecolor, edgecolor=edgecolor)
        return fig,ax

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

def suck_on_that_0point0( start, stop, step=1, format_depth=1, labels_every=None ):
    values = []
    if labels_every is None:
        values = []
        for i in np.arange( start, stop, step ):
            if i == 0:
                values.append('0')
            else:
                values.append('{:0.{dpt}f}'.format(i,dpt=format_depth))
    else:
        for i in np.arange( start, stop, step ):
            if i == 0 and np.mod(i,labels_every)==0:
                values.append('0')
            elif np.mod(i,labels_every)==0:
                values.append('{:0.{dpt}f}'.format(i,dpt=format_depth))
            else:
                values.append('')
    return values

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

def psth_in_grid( gx, gy, x, y, e=None, x_scale=1, y_scale=1,
                    color="#0000ff", std=None ):
    # if std is not None:
    #     plt.plot( (gx*x_scale)+x, (gy*y_scale)+std, color="#000000" )
    plt.plot( (gx*x_scale)+x, (gy*y_scale)+y, color=color )
    if e is not None:
        plt.fill_between( (gx*x_scale)+x, (gy*y_scale)+(y-e), (gy*y_scale)+(y+e), facecolor=color, alpha=0.4, linewidth=0 )

def plot_psth_grid_simple( ax, xvalues, psth, bs, std, stimulus_ids, stim_id_grid, y_scale, prototype_stimuli, category_stimuli, directions, spatialfs ):
    """ Plots an entire grid with psths at the according places """
    x_scale = 1.2*(xvalues[-1]-xvalues[0])
    n_spatialf,n_direction = stim_id_grid.shape
    x_plotted = []
    for sf in range(n_spatialf):
        for dir_ in range(n_direction):
            stim_id = stim_id_grid[sf,dir_]
            if np.isfinite(stim_id):
                color = "#999999"
                for c_dir,c_sf in zip( category_stimuli["left"]["dir"],
                        category_stimuli["left"]["spf"] ):
                    if c_dir==directions[dir_] and c_sf==spatialfs[sf]:
                        color = "#60C3DB"
                for c_dir,c_sf in zip( category_stimuli["right"]["dir"],
                        category_stimuli["right"]["spf"] ):
                    if c_dir==directions[dir_] and c_sf==spatialfs[sf]:
                        color = "#F60951"
                if prototype_stimuli is not None:
                    if prototype_stimuli["left"]["dir"]==directions[dir_] and \
                        prototype_stimuli["left"]["spf"]==spatialfs[sf]:
                        color = "#000099"
                    if prototype_stimuli["right"]["dir"]==directions[dir_] and \
                    prototype_stimuli["right"]["spf"]==spatialfs[sf]:
                        color = "#990000"
                stim_ids = np.where(stimulus_ids==stim_id)[0]
                mean_curve,sem_curve,_ = mean_sem(psth[stim_ids,:],axis=0)
                if bs is not None:
                    if type(bs) == float:
                        mean_curve = mean_curve - bs
                    else:
                        mean_curve = mean_curve - np.mean(bs[stim_ids])
                if std is not None:
                    std_curve = np.zeros_like(mean_curve) + \
                        ( 3 * np.mean(std[stim_ids]) )
                else:
                    std_curve = None
                psth_in_grid( gx=dir_, gy=sf, x=xvalues, y=mean_curve,
                    e=sem_curve, x_scale=x_scale, y_scale=y_scale,
                    color=color, std=std_curve )
                x_plotted.append(dir_)
    x_left_ticklabels = (min(x_plotted)*x_scale)+xvalues[0]-(0.2*x_scale)
    for y in range(n_spatialf):
        plt.text(x_left_ticklabels, y*y_scale, "{}".format(spatialfs[y]),
            rotation=0, ha='right', va='center', size=6, color='#000000' )
    y_bottom_ticklabels = -0.2*y_scale
    for x in range(n_direction):
        if x >= min(x_plotted) and x <= max(x_plotted):
            plt.text((x*x_scale)+np.median(xvalues), y_bottom_ticklabels,
                "{}".format(directions[x]), rotation=0,
                ha='center', va='top', size=6, color='#000000' )
    ax.set_ylim(-0.6*y_scale,(n_spatialf*y_scale)+(0.2*y_scale))
    # ax.set_xlim(xvalues[0]-0.1,xvalues[-1]+0.1)

def finish_figure( filename=None, wspace=None, hspace=None ):
    """ Finish up layout and save to ~/figures"""
    plt.tight_layout()
    if wspace is not None or hspace is not None:
        if wspace is None: wspace = 0.6
        if hspace is None: hspace = 0.8
        plt.subplots_adjust( wspace=wspace, hspace=hspace )
    if filename is not None:
        plt.savefig(filename+'.pdf', transparent=True)

def stimulus(file):
    return file['S']['StimIDs'][0,0].ravel()

def direction(file):
    """ Returns a list with directions of the stimuli """
    stimuli = file['S']['StimIDs'][0,0].ravel()
    direction_list = file['S']['Angles'][0,0].ravel()
    return np.array(
        [direction_list[int(s)-1] for s in stimuli])

def spatialf(file):
    """ Returns a list with spatial frequencies of the stimuli """
    stimuli = file['S']['StimIDs'][0,0].ravel()
    spatialf_list = file['S']['spatialF'][0,0].ravel()
    return np.array(
        [spatialf_list[int(s)-1] for s in stimuli])

def left_category(file, dir_offset=0):
    """ Returns a list with (dir,spf) tuples of the left category stimuli
    """
    direction = file['S']['LeftCat'][0,0]['Angles'][0,0].ravel()+dir_offset
    spatialf = file['S']['LeftCat'][0,0]['spatialF'][0,0].ravel()
    return {'dir': direction, 'spf': spatialf}

def right_category(file, dir_offset=0):
    """ Returns a list with (dir,spf) tuples of the right category stimuli
    """
    direction = file['S']['RightCat'][0,0]['Angles'][0,0].ravel()+dir_offset
    spatialf = file['S']['RightCat'][0,0]['spatialF'][0,0].ravel()
    return {'dir': direction, 'spf': spatialf}

def get_1d_stimulus_ix_in_2d_grid(file,directions,spatialfs):
    """ Returns 2d grids with stimulus, direction, orientation ids """
    stim_2d = np.full((6,18),np.NaN)
    rec_spatialf = spatialf(file)
    rec_directions = direction(file)
    rec_stimuli = stimulus(file)
    n_unique_stimuli = int(rec_stimuli.max())
    for s in range(1,n_unique_stimuli+1):
        s_ix = np.where(rec_stimuli==s)[0][0]
        dir = rec_directions[s_ix]
        sf = rec_spatialf[s_ix]
        dir_ix = np.where(directions==dir)[0][0]
        sf_ix = np.where(spatialfs==sf)[0][0]
        # print("{}: {}, dir={}, sf={}, dir_ix={}, sf_ix={}".format( \
        #     s, s_ix, dir, sf, dir_ix, sf_ix ))
        stim_2d[sf_ix,dir_ix] = s
    return stim_2d

def get_trial_category_id(file, cat_stimuli):
    """ Returns a list with the category id of each trial """
    dir_id = direction(file)
    spf_id = spatialf(file)
    n_trials = len(dir_id)
    category_id = np.zeros(n_trials)
    for t in range(n_trials):
        for d,s in zip( cat_stimuli['left']['dir'],
                        cat_stimuli['left']['spf'] ):
            if dir_id[t] == d and spf_id[t] == s:
                category_id[t] = 1
                break
        for d,s in zip( cat_stimuli['right']['dir'],
                        cat_stimuli['right']['spf'] ):
            if dir_id[t] == d and spf_id[t] == s:
                category_id[t] = 2
                break
    return category_id

def mouth_tracking_parameters( dlcdata ):
    """ Calculate eye tracking parameters """
    # Get the main index name and other summary data
    dlcscorer = dlcdata.columns[0][0]
    n_frames = dlcdata.shape[0]
    annotation_names = ["leftback", "leftmiddle", "leftfront", "rightfront", "rightmiddle", "rightback", "lowerjawfront", "lowerjawback"]

    # Get upper jaw left (top) coordinates
    leftmiddle_x = np.array(dlcdata[dlcscorer,"leftmiddle","x"])
    leftmiddle_y = np.array(dlcdata[dlcscorer,"leftmiddle","y"])

    # Get upper jaw right (bottom) coordinates
    rightmiddle_x = np.array(dlcdata[dlcscorer,"rightmiddle","x"])
    rightmiddle_y = np.array(dlcdata[dlcscorer,"rightmiddle","y"])

    # Get lower jaw front coordinates
    lowerjawfront_x = np.array(dlcdata[dlcscorer,"lowerjawfront","x"])
    lowerjawfront_y = np.array(dlcdata[dlcscorer,"lowerjawfront","y"])

    # Distance between upper and lower jaw
    mouth_open = np.sqrt( np.add( np.square( np.subtract(leftmiddle_x, lowerjawfront_x) ) , np.square( np.subtract(leftmiddle_y, lowerjawfront_y) ) ) )

    # Distance between left and right upper jaw
    mouth_total = np.sqrt( np.add( np.square( np.subtract(leftmiddle_x, rightmiddle_x) ) , np.square( np.subtract(leftmiddle_y, rightmiddle_y) ) ) )

    mouth_open_rel = mouth_open / mouth_total
    return mouth_open_rel


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Code

for m_nr,m in enumerate(mice):
    data_path = os.path.join(base_data_path,m)

    # Load mat file of this session
    matfilename = glob.glob(os.path.join(data_path,"*StimSettings.mat"))[0]
    print(matfilename)
    matfile = loadmat(matfilename)

    # Load mat file with categories of this mouse
    catfilename = glob.glob(os.path.join(data_path,"*CatSett.mat"))[0]
    print(catfilename)
    catfile = loadmat(catfilename)

    # Make dict with category stimuli
    if m == "V08":
        cat_stimuli = {"left": left_category(catfile,-10), "right": right_category(catfile,-10)}
    else:
        cat_stimuli = {"left": left_category(catfile), "right": right_category(catfile)}
    all_stimuli = stimulus(matfile)
    stim_id_grid = get_1d_stimulus_ix_in_2d_grid(matfile, all_directions, all_spatialfs)
    cat_id = get_trial_category_id(matfile, cat_stimuli)

    # Load Aux data
    auxfilestem = "*"+m+"*.lvd"
    Aux = auxrec.LvdAuxRecorder(data_path, filename=auxfilestem, nimagingplanes=n_imaging_planes)
    print(Aux)

    # Find visual stimulus onsets in aux data
    stim_onsets = Aux.stimulus_onsets

    # Load Vid synchronization indices
    search_path = os.path.join(data_path,"*"+m+"*eye1-ix.npy")
    ix_files_full = glob.glob(search_path)
    sync_ixs = np.load(ix_files_full[0], allow_pickle=True).item()["video_frame_conversion_index"].astype(int)

    # Find matching deeplabcut data
    dlcfilestem = "*"+m+"*-eye1DLC*.h5"
    dlc_h5_file = glob.glob(os.path.join(data_path,dlcfilestem))
    dlcdata = pd.read_hdf(dlc_h5_file[0])
    dlcscorer = dlcdata.columns[0][0]
    n_frames = dlcdata.shape[0]

    if Aux.imagingsf > 5:
        x_range = np.arange(-8,24)
    else:
        x_range = np.arange(-4,12)
    x_values = x_range / Aux.imagingsf
    mouthopen = mouth_tracking_parameters( dlcdata )

    fig,ax = init_figure_axes(fig_size=(10,6.6))

    for catid_ in range(3):
        ax = plt.subplot2grid( (2,3), (0,catid_) )

        this_stim_onsets = stim_onsets[cat_id==catid_]

        data_mat = np.zeros((len(this_stim_onsets),len(x_range)))
        for tr_nr,tr in enumerate(this_stim_onsets):
            frames = tr+x_range
            data_mat[tr_nr,:] = mouthopen[sync_ixs[frames]]
            plt.plot(x_values,data_mat[tr_nr,:],color="#aaaaaa")
        mn,sem,n = mean_sem( data_mat, axis=0 )

        line( x_values, mn, sem, line_color="#0000AA", sem_color="#0000AA", shaded=True )

        finish_panel( ax, ylabel="mouth-open, id={}".format(catid_), xlabel="Time (s)", legend="off", y_minmax=[0,1.5], y_step=[0.5,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[x_values[0],x_values[-1]], x_margin=0.55, x_axis_margin=0.55, despine=True )

        ax = plt.subplot2grid( (2,3), (1,catid_) )
        line( x_values, mn, sem, line_color="#0000AA", sem_color="#0000AA", shaded=True )
        finish_panel( ax, ylabel="mouth-open, id={}".format(catid_), xlabel="Time (s)", legend="off", y_minmax=[0.3,0.4], y_step=[0.05,2], y_margin=0.002, y_axis_margin=0.001, x_minmax=[x_values[0],x_values[-1]], x_margin=0.55, x_axis_margin=0.55, despine=True )

    finish_figure( filename=os.path.join(savefig,"6ED10gj-MouthMovement-PerCategory-Mouse-"+m), wspace=0.5, hspace=0.5 )

    fig,ax = init_figure_axes(fig_size=(10,6.6))

    data_mat = np.zeros((len(stim_onsets),len(x_range)))
    for tr_nr,tr in enumerate(stim_onsets):
        frames = tr+x_range
        data_mat[tr_nr,:] = mouthopen[sync_ixs[frames]]

    plot_psth_grid_simple( ax=ax, xvalues=x_values, psth=data_mat, bs=0.3, std=None, stimulus_ids=all_stimuli, stim_id_grid=stim_id_grid, y_scale=0.1, prototype_stimuli=None, category_stimuli=cat_stimuli, directions=all_directions, spatialfs=all_spatialfs )
    plt.axis('off')

    finish_figure( filename=os.path.join(savefig,"6ED10kh-MouthMovement-PerStimulus-Mouse-"+m), wspace=0.5, hspace=0.5 )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# That's all folks!
plt.show()
