#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions and settings for retinotopy analysis of revision experiment

Created on Thu Oct 22, 2020

@author: pgoltstein
"""


# Imports
import os, sys, glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scistats
import scipy.linalg
import scipy.optimize
from scipy.io import loadmat
sys.path.append('../xx_analysissupport')
import auxrec

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
warnings.filterwarnings('ignore')

# settings for retaining pdf font
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

# Default settings
font_size = { "title": 10, "label": 9, "tick": 9, "text": 8, "legend": 8 }
degree_sign = u"\N{DEGREE SIGN}"

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

# Data path
basepath="../.."
datapath = basepath+"/data/p2b_retinotopybehavior"
figpath = basepath+"/figureout"

# Settings
all_orientations = [0,20,40,60,80,100,120]
all_spatialfreqs = [0.24,0.16,0.12,0.08,0.06,0.04]
colormap = "RdBu_r"
shift_colors = { 26: "#000000", 0: "#880000", -26: "#ff0000" }

# Functions
def load_mouse_data(datapath, mice, shifts, include_eye=False, im_nplanes=None):
    print("\n ---LOADING DATA ---")
    mousedata = {}
    eyedata = {}
    for m in mice:
        print("\n{}".format(m))
        matfiles = sorted(glob.glob(os.path.join(datapath,m,"*.mat")))
        mousedata[m] = dict(zip(shifts, [[] for _ in range(len(shifts))]))
        eyedata[m] = dict(zip(shifts, [[] for _ in range(len(shifts))]))
        for f in matfiles:
            print(f)ß
            matfile_contents = loadmat(f)
            shift = int(matfile_contents["S"]["MonitorPosition"])
            print("  - shift = {} degrees".format(shift))
            if shift in shifts:
                mousedata[m][shift].append(matfile_contents)

                if not include_eye:
                    continue

                # load aux data if present and save trial onset frames
                eyedict = {"trialonsets": [], "sf": 0, "eye1": {}, "eye2": {}}
                filestem = f.split("-")[-2]
                auxfilestem = "*"+filestem+"*.lvd"
                print(auxfilestem)
                Aux = auxrec.LvdAuxRecorder(os.path.join(datapath,m), filename=auxfilestem, nimagingplanes=im_nplanes[m])
                print(Aux)
                try:
                    eyedict["trialonsets"] = np.array(Aux.stimulus_onsets)
                    eyedict["sf"] = float(Aux.imagingsf)
                except:
                    print("No trial found, skipping")
                    continue

                # Load Vid synchronization indices
                data_present = {"eye1": False, "eye2": False}
                search_path = os.path.join(datapath,m,"*"+filestem+"*-eye1-ix.npy")
                ix_files_full = glob.glob(search_path)
                if len(ix_files_full) > 0:
                    print(ix_files_full[0])
                    eyedict["eye1"]["ix"] = np.load(ix_files_full[0], allow_pickle=True).item()["video_frame_conversion_index"].astype(int)
                    data_present["eye1"] = True

                search_path = os.path.join(datapath,m,"*"+filestem+"*-eye2-ix.npy")
                ix_files_full = glob.glob(search_path)
                if len(ix_files_full) > 0:
                    print(ix_files_full[0])
                    eyedict["eye2"]["ix"] = np.load(ix_files_full[0], allow_pickle=True).item()["video_frame_conversion_index"].astype(int)
                    data_present["eye2"] = True

                # Find matching deeplabcut data
                search_path = os.path.join(datapath,m,"*"+filestem+"*-eye1DLC*.h5")
                dlc_h5_file = glob.glob(search_path)
                if len(dlc_h5_file) > 0:
                    print(dlc_h5_file[0])
                    dlcdata = pd.read_hdf(dlc_h5_file[0])
                    dlcdict = calc_eye_tracking_parameters( dlcdata )
                    for k,v in dlcdict.items():
                        eyedict["eye1"][k] = v

                search_path = os.path.join(datapath,m,"*"+filestem+"*-eye2DLC*.h5")
                dlc_h5_file = glob.glob(search_path)
                if len(dlc_h5_file) > 0:
                    print(dlc_h5_file[0])
                    dlcdata = pd.read_hdf(dlc_h5_file[0])
                    dlcdict = calc_eye_tracking_parameters( dlcdata )
                    for k,v in dlcdict.items():
                        eyedict["eye2"][k] = v
                if data_present["eye1"] and data_present["eye2"]:
                    eyedata[m][shift].append(eyedict)

    return mousedata, eyedata

def calc_eye_tracking_parameters( dlcdata ):
    """ Calculate eye tracking parameters """
    # Get the main index name and other summary data
    dlcscorer = dlcdata.columns[0][0]
    n_frames = dlcdata.shape[0]

    # Get x and y coords of pupil center
    pupil_names = ["pupil_left", "pupil_LT", "pupil_top", "pupil_RT", "pupil_right", "pupil_RB", "pupil_bottom", "pupil_LB"]
    pupil_coords_x = np.zeros((n_frames,len(pupil_names)))
    pupil_coords_y = np.zeros((n_frames,len(pupil_names)))
    for point_nr, point_name in enumerate(pupil_names):
        pupil_coords_x[:,point_nr] = dlcdata[dlcscorer,point_name,"x"]
        pupil_coords_y[:,point_nr] = dlcdata[dlcscorer,point_name,"y"]

    # Get eye left coordinates
    eye_left_x = np.array(dlcdata[dlcscorer,"eye_left","x"])
    eye_left_y = np.array(dlcdata[dlcscorer,"eye_left","y"])

    # Get eye right coordinates
    eye_right_x = np.array(dlcdata[dlcscorer,"eye_right","x"])
    eye_right_y = np.array(dlcdata[dlcscorer,"eye_right","y"])

    # Get eye top coordinates
    eye_top_x = np.array(dlcdata[dlcscorer,"eye_top","x"])
    eye_top_y = np.array(dlcdata[dlcscorer,"eye_top","y"])

    # Get eye bottom coordinates
    eye_bottom_x = np.array(dlcdata[dlcscorer,"eye_bottom","x"])
    eye_bottom_y = np.array(dlcdata[dlcscorer,"eye_bottom","y"])

    # calculate pupil center
    pupil_center_x = np.mean(pupil_coords_x,axis=1)
    pupil_center_y = np.mean(pupil_coords_y,axis=1)

    # calculate pupil diameter
    n_opps = int(len(pupil_names)/4)
    pupil_diameter = np.zeros((n_frames,n_opps))
    for p_nr in range(n_opps):
        pupil_diameter[:,p_nr] = np.sqrt( np.add( np.square( np.subtract(pupil_coords_x[:,p_nr],pupil_coords_x[:,p_nr+n_opps]) ) , np.square( np.subtract(pupil_coords_y[:,p_nr],pupil_coords_y[:,p_nr+n_opps]) ) ) )
    pupil_diameter = np.mean(pupil_diameter,axis=1)

    # calculate distance from eye left to pupil center
    pupil_shift_x = pupil_center_x-eye_left_x
    pupil_shift_y = pupil_center_y-eye_left_y

    # normalize everything to distance between right and left of eye
    pupil_diameter = pupil_diameter / (eye_right_x-eye_left_x)
    pupil_shift_x = pupil_shift_x / (eye_right_x-eye_left_x)
    pupil_shift_y = pupil_shift_y / (eye_top_y-eye_bottom_y)

    return {"x": pupil_shift_x, "y": pupil_shift_y, "d": pupil_diameter}

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

def running_average( data, n):
    dlen = data.shape[0]
    half_w_len = np.floor(n/2)
    run_avg = np.zeros_like(data)
    for t in range(dlen):
        window = t + np.arange(-half_w_len,half_w_len,1)
        if np.min(window) < 0:
            window = window[window>=0]
        if np.max(window) > dlen-1:
            window = window[window<dlen]
        run_avg[t] = np.nanmean(data[window.astype(int)])
    return run_avg

def mean_sem( datamat, axis=0 ):
    mean = np.nanmean(datamat,axis=axis)
    n = np.sum( ~np.isnan( datamat ), axis=axis )
    sem = np.nanstd( datamat, axis=axis ) / np.sqrt( n )
    return mean,sem,n

def bounded_sigmoid(p,x):
    # x0: time point of maximum steepness
    # k: steepness
    x0,k=p
    y = 1 / (1.0 + np.exp(-k*(x-x0)))
    return y


def get_bounded_sigmoid_fit( x, y ):
    # returns parameters (x0,k), see bounded sigmoid for explanation
    def residuals(p,x,y):
        return y - bounded_sigmoid(p,x)
    p_guess=(np.median(x),1.0)
    p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
                    residuals,p_guess,args=(x,y),full_output=1)
    return p

def calculate_boundary( performance_matrix, normalize=False, mat_size=(6,7), return_grid=False ):
    """ Calulates the boundary equation and angle using the 6x7 performance matrix as input, matrix should include np.NaN on not-used entries """

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    if normalize:
        performance_matrix = (performance_matrix - np.nanmin(performance_matrix)) / (np.nanmax(performance_matrix)-np.nanmin(performance_matrix))

    # Prepare matrix with x,y,z values
    n_datapts = mat_size[0]*mat_size[1]
    X,Y = np.meshgrid(np.arange(mat_size[1]),np.arange(mat_size[0]))
    X = np.reshape(X,(n_datapts,))
    Y = np.reshape(Y,(n_datapts,))
    P = performance_matrix.ravel()
    notNanIx = ~np.isnan(P)
    data = np.stack( (X[notNanIx], Y[notNanIx], P[notNanIx]) ).T
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]

    # Get coefficients of fitted plane
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

    # Calculate line that divides fitted plane at Z=0.5
    x_line = np.arange(-1,mat_size[1]+1)
    y_line = (((-1*C[0]*x_line) -C[2])+0.5)/C[1]

    # Calculate angle of line
    boundary_angle = np.rad2deg(np.arctan2(y_line[-1] - y_line[0], x_line[-1] - x_line[0]))

    if return_grid:
        # Calculate plane-grid with applied boundary
        X,Y = np.meshgrid(np.arange(mat_size[1]),np.arange(mat_size[0]))
        grid = np.zeros(mat_size)
        grid[ C[0]*X + C[1]*Y + C[2] > 0.5 ] = 1.0
        return boundary_angle, (x_line,y_line), grid
    else:
        return boundary_angle, (x_line,y_line)

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

def finish_figure( filename=None, wspace=None, hspace=None ):
    """ Finish up layout and save to ~/figures"""
    plt.tight_layout()
    if wspace is not None or hspace is not None:
        if wspace is None: wspace = 0.6
        if hspace is None: hspace = 0.8
        plt.subplots_adjust( wspace=wspace, hspace=hspace )
    if filename is not None:
        plt.savefig(filename+'.pdf', transparent=True)

def barplot_shifts( datamat, shifts, ylabel, y_minmax, y_step, savename ):
    fig,ax = init_figure_axes(fig_size=(4.5,6))
    m,e,n = mean_sem(datamat,axis=0)
    position_labels = ["{}{}".format(sh,degree_sign) for sh in shifts]
    for sh,shift in enumerate(shifts):
        bar( sh, m[sh], e[sh] )
    for m in range(datamat.shape[0]):
        plt.plot( datamat[m,:], color="#aaaaaa", marker="o", linewidth=1,
        markerfacecolor="#aaaaaa", markersize=2,
        markeredgewidth=None, markeredgecolor=None )
    finish_panel( ax, ylabel=ylabel, xlabel="Stimulus position", legend="off", y_minmax=y_minmax, y_step=y_step, y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,len(shifts)-1], x_ticks=list(range(len(shifts))), x_ticklabels=position_labels, x_margin=0.75, x_axis_margin=0.55, despine=True)
    finish_figure( filename=os.path.join(figpath,savename), wspace=0.8, hspace=0.8 )

def curveplots_shifts( datamat, shifts, savename ):
    xvalues = np.arange(12)
    xticklabels = np.concatenate([np.arange(-6,0,1),np.arange(1,7,1)])
    fig,ax = init_figure_axes(fig_size=(22.5,4.5))
    for sh_nr,sh in enumerate(shifts):
        ax = plt.subplot2grid( (1,5), (0,sh_nr) )
        mn,sem,n = mean_sem( datamat[:,sh_nr,:], axis=0 )
        for m_nr in range(datamat.shape[0]):
            plt.plot( xvalues, datamat[m_nr,sh_nr,:], marker="o", markersize=1, color='#aaaaaa', linestyle='-', linewidth=1 )
        line( xvalues, mn, sem, line_color=shift_colors[sh], sem_color=shift_colors[sh] )
        finish_panel( ax, ylabel="Fraction chosen", xlabel="Boundary distance", legend="off", y_minmax=[0,1], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,11], x_margin=0.55, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)

    ax = plt.subplot2grid( (1,5), (0,3) )
    for sh_nr,sh in enumerate(shifts):
        mn,sem,n = mean_sem( datamat[:,sh_nr,:], axis=0 )
        line( xvalues, mn, sem, line_color=shift_colors[sh], sem_color=shift_colors[sh] )
    finish_panel( ax, ylabel="Fraction chosen", xlabel="Boundary distance", legend="off", y_minmax=[0,1], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,11], x_margin=0.55, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)

    ax = plt.subplot2grid( (1,5), (0,4) )
    zorders = np.random.randint(low=20,high=(len(shifts)*datamat.shape[0])+20, size=(len(shifts)*datamat.shape[0],))
    zcnt = 0
    for sh_nr,sh in enumerate(shifts):
        for m_nr in range(datamat.shape[0]):
            plt.plot( xvalues, datamat[m_nr,sh_nr,:], marker="", markersize=0, color=shift_colors[sh], linestyle='-', linewidth=1, zorder=zorders[zcnt] )
            zcnt += 1
    finish_panel( ax, ylabel="Fraction chosen", xlabel="Boundary distance", legend="off", y_minmax=[0,1], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,11], x_margin=0.55, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)
    finish_figure( filename=os.path.join(figpath,savename), wspace=0.8, hspace=0.8 )

def plot_grids_mouse_shift(gridmat, mice, shifts):
    fig,ax = init_figure_axes(fig_size=(8,14))
    for m_nr,m in enumerate(mice):
        for sh_nr,sh in enumerate(shifts):
            ax = plt.subplot2grid( (len(mice),len(shifts)), (m_nr,sh_nr) )
            category_grid = gridmat[m_nr,sh_nr,:,:]

            if m == "C04":
                category_grid = np.flip(category_grid, axis=0)
                category_grid = np.flip(category_grid, axis=1)
            if m == "V05" or m == "V09":
                category_grid = np.flip(category_grid, axis=0)

            performance_mat = (category_grid+1) / 2
            b_a, (x_line,y_line) = calculate_boundary( performance_mat, normalize=True )
            plt.imshow(category_grid, cmap=colormap, vmin=-1.0, vmax=1.0)
            ax.plot(x_line, y_line, color="#000000")
            plt.axis('off')
            plt.title("{}, {}{}".format(m,sh,degree_sign), fontsize=font_size['title'])
    finish_figure( filename=os.path.join(figpath,"2Performance-2Dplots-permouse"), wspace=0.25, hspace=0.0 )

def plot_grids_shift(gridmat, mice, shifts):
    fig,ax = init_figure_axes(fig_size=(8,4))
    for sh_nr,sh in enumerate(shifts):
        ax = plt.subplot2grid( (1,len(shifts)), (0,sh_nr) )
        category_grid = np.zeros((6,7,len(mice)))
        for m_nr,m in enumerate(mice):
            mouse_grid = gridmat[m_nr,sh_nr,:,:]
            if m == "C04":
                mouse_grid = np.flip(mouse_grid, axis=0)
                mouse_grid = np.flip(mouse_grid, axis=1)
            if m == "V05" or m == "V09":
                mouse_grid = np.flip(mouse_grid, axis=0)
            category_grid[:,:,m_nr] = mouse_grid
        category_grid = np.nanmean(category_grid,axis=2)
        performance_mat = (category_grid+1) / 2
        b_a, (x_line,y_line) = calculate_boundary( performance_mat, normalize=True )
        plt.imshow(category_grid, cmap=colormap, vmin=-1.0, vmax=1.0)
        ax.plot(x_line, y_line, color="#000000")
        plt.axis('off')
        plt.title("{}{}".format(sh,degree_sign), fontsize=font_size['title'])
    finish_figure( filename=os.path.join(figpath,"2c-Performance-2Dplots"), wspace=0.25, hspace=0.0 )

def kruskalwallis_across_shifts(data_per_shift, shifts, y_variable, posthoc_wmpsr_alternative="two-sided"):
    print("\n{}:".format(y_variable))
    report_kruskalwallis( [*data_per_shift.T], n_indents=0, alpha=0.05 )
    for sh1,shift1 in enumerate(shifts):
        for sh2,shift2 in enumerate(shifts):
            if sh1 < sh2:
                print("  {}{} vs {}{}".format(shift1, degree_sign, shift2, degree_sign), end="")
                report_wmpsr_test( data_per_shift[:,sh1], data_per_shift[:,sh2], n_indents=4, alpha=0.05, alternative=posthoc_wmpsr_alternative, bonferroni=1 )

def report_wmpsr_test( sample1, sample2, n_indents=2, alpha=0.05, alternative="two-sided", bonferroni=1 ):
    p,Z,n = wilcoxon_matched_pairs_signed_rank_test( sample1, sample2, alternative=alternative )
    print('{}WMPSR test, Z={:0.0f}, p={:0.4f}, n={:0.0f}{}'.format( " "*n_indents, Z, p, n, "  >> significant" if p<(alpha/bonferroni) else "." ))
    return p

def report_mannwhitneyu_test( sample1, sample2, n_indents=2, alpha=0.05, bonferroni=1 ):
    p,U,r,n1,n2 = mann_whitney_u_test( sample1, sample2 )
    print('{}Mann-Whitney U test, U={:0.0f}, p={:0.4f}, r={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, U, p, r, n1, n2, "  >> significant" if p<(alpha/bonferroni) else "." ))
    return p

def report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 ):
    p,H,DFbetween,DFwithin,n = kruskalwallis( samplelist )
    print("{}Kruskal-Wallis test, X^2 = {:0.3f}, df = {:0.0f} p = {:0.4f}, n={:0.0f} {}".format( " "*n_indents, H, DFbetween, p, n, "  >> significant" if p<alpha else "." ))

def wilcoxon_matched_pairs_signed_rank_test( sample1, sample2, alternative="two-sided" ):
    sample1 = sample1[~np.isnan(sample1)].ravel()
    sample2 = sample2[~np.isnan(sample2)].ravel()
    if np.count_nonzero(sample1)==0 and np.count_nonzero(sample2)==0:
        return 1.0,np.NaN,np.NaN
    elif len(sample1) != len(sample2):
        return 1.0,np.NaN,np.NaN
    else:
        Z,p = scistats.wilcoxon(sample1, sample2, alternative=alternative)
        n = len(sample1)
        return p,Z,n

def mann_whitney_u_test( sample1, sample2 ):
    sample1 = sample1[~np.isnan(sample1)].ravel()
    sample2 = sample2[~np.isnan(sample2)].ravel()
    U,p = scistats.mannwhitneyu(sample1, sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    r = U / np.sqrt(n1+n2)
    return p,U,r,n1,n2

def kruskalwallis( samplelist ):
    # Clean up sample list and calculate N
    N = 0
    no_nan_samplelist = []
    for b in range(len(samplelist)):
        no_nan_samples = samplelist[b][~np.isnan(samplelist[b])]
        if len(no_nan_samples) > 0:
            no_nan_samplelist.append(no_nan_samples)
            N += len(no_nan_samples)

    # Calculate degrees of freedom
    k = len(samplelist)
    DFbetween = k - 1
    DFwithin = N - k
    DFtotal = N - 1
    H,p = scistats.kruskal( *no_nan_samplelist )
    return p,H,DFbetween,DFwithin,N


def calculate_category_gridid_levelid(data, trial_select):

    # Get the category angles and spatialF's
    left_angles = data["S"]["LeftCat"][0,0]["Angles"][0,0].ravel()
    right_angles = data["S"]["RightCat"][0,0]["Angles"][0,0].ravel()
    left_spatfs = data["S"]["LeftCat"][0,0]["spatialF"][0,0].ravel()
    right_spatfs = data["S"]["RightCat"][0,0]["spatialF"][0,0].ravel()
    left_levels = data["S"]["LeftCat"][0,0]["CatLevel"][0,0].ravel()
    right_levels = data["S"]["RightCat"][0,0]["CatLevel"][0,0].ravel()
    angles = np.unique(np.concatenate([left_angles,right_angles]))
    all_angles_mouse = all_orientations + np.min(angles)

    # Calculate the index in the grid, for each category stimulus
    left_angle_ix = []
    right_angle_ix = []
    left_spatf_ix = []
    right_spatf_ix = []
    for l_ang, l_spf in zip(left_angles,left_spatfs):
        left_angle_ix.append( int( np.argwhere( all_angles_mouse==l_ang ) ) )
        left_spatf_ix.append( int( np.argwhere( all_spatialfreqs==l_spf ) ) )
    for r_ang, r_spf in zip(right_angles,right_spatfs):
        right_angle_ix.append( int( np.argwhere( all_angles_mouse==r_ang ) ) )
        right_spatf_ix.append( int( np.argwhere( all_spatialfreqs==r_spf ) ) )

    # Get the category level and grid index per trial
    outcome_raw = data["Outcome"].ravel()[trial_select].astype(np.float)
    category_id = data["CategoryId"].ravel()[trial_select]
    category_id = category_id[~np.isnan(outcome_raw)]
    stimulus_id = data["StimulusId"].ravel()[trial_select]
    stimulus_id = stimulus_id[~np.isnan(outcome_raw)]
    stimulus_id -= 1 # Convert to zero based indices
    grid_ids = []
    catlevels = np.zeros_like(outcome_raw)
    for tr_nr,(cat_id,stim_id) in enumerate( zip( category_id, stimulus_id ) ):
        if cat_id == 1:
            grid_ids.append([left_spatf_ix[stim_id], left_angle_ix[stim_id]])
            catlevels[tr_nr] = int((7-left_levels[stim_id]) * -1)
        elif cat_id == 2:
            grid_ids.append([right_spatf_ix[stim_id], right_angle_ix[stim_id]])
            catlevels[tr_nr] = int((7-right_levels[stim_id]))

    return grid_ids, catlevels

def make_category_grid(response_side, grid_ids):
    correct_trials = np.zeros((6,7))
    all_trials = np.zeros((6,7))
    for tr in range(len(response_side)):
        spf,ang = grid_ids[tr]
        all_trials[spf,ang] += 1
        if response_side[tr] == 1:
            correct_trials[spf,ang] -= 1
        if response_side[tr] == 2:
            correct_trials[spf,ang] += 1
    category_grid = correct_trials / all_trials
    category_grid[correct_trials==0] = 0
    category_grid[all_trials==0] = np.NaN
    return category_grid

def make_category_level_curve(response_side, outcome, catlevels):
    choice_trials = np.zeros((13,))
    correct_trials = np.zeros((13,))
    all_trials = np.zeros((13,))
    for tr in range(len(response_side)):
        lev = int(catlevels[tr])+6
        all_trials[lev] += 1
        if response_side[tr] == 1:
            choice_trials[lev] -= 1
        if response_side[tr] == 2:
            choice_trials[lev] += 1
        if outcome[tr] == 1:
            correct_trials[lev] += 1
    category_curve = choice_trials / all_trials
    category_curve[choice_trials==0] = 0
    category_curve[all_trials==0] = np.NaN
    category_curve = np.delete(category_curve,6)
    category_curve = (category_curve -1) / -2
    category_performance = correct_trials / all_trials
    category_performance[correct_trials==0] = 0
    category_performance[all_trials==0] = np.NaN
    category_performance = np.delete(category_performance,6)
    xlabels = np.concatenate([np.arange(-6,0,1),np.arange(1,7,1)])
    return category_curve, category_performance, xlabels

def mean_per_shift( mousedata, mice, shifts ):
    p_per_shift = np.full( (len(mice),len(shifts)), np.NaN )
    for m_nr,m in enumerate(mice):
        for sh_nr,sh in enumerate(shifts):
            p_per_ses = []
            print(">{}  {}: {}".format(m,sh,len(mousedata[m][sh])))
            for s in mousedata[m][sh]:
                outcomes = s["Outcome"].ravel().astype(np.float)
                if m[0] == "V":
                     outcomes = outcomes[12:]
                p_per_ses.append( np.nanmean( outcomes ) )
            p_per_shift[m_nr,sh_nr] = np.nanmean(p_per_ses)
    return p_per_shift

def eye_mean_per_shift( eyedata, mice, shifts, eye, param, period=[0,1.5] ):
    # eyedict = {"trialonsets": [], "sf": 0, "eye1-ix": [], "eye2-ix": [], "eye1": {}, "eye2": {}}
    m_per_shift = np.full( (len(mice),len(shifts)), np.NaN )
    for m_nr,m in enumerate(mice):
        for sh_nr,sh in enumerate(shifts):
            m_per_ses = []
            print(">{}  {}: {}".format(m,sh,len(eyedata[m][sh])))
            for e in eyedata[m][sh]:

                sf = e["sf"]
                fr_range = np.arange(int(period[0]*sf),int(period[1]*sf))
                data_mat = np.zeros((len(e["trialonsets"]),))

                for tr_nr,tr in enumerate(e["trialonsets"]):
                    trialframes = e[eye]["ix"][tr+fr_range]
                    data_mat[tr_nr] = np.nanmean(e[eye][param][trialframes])

                if m[0] == "V":
                     data_mat = data_mat[12:]
                m_per_ses.append( np.nanmean( data_mat ) )
            m_per_shift[m_nr,sh_nr] = np.nanmean(m_per_ses)
    return m_per_shift

def mean_grid_per_shift( mousedata, mice, shifts ):
    grid_per_shift = np.full( (len(mice), len(shifts), len(all_spatialfreqs), len(all_orientations)), np.NaN )
    for m_nr,m in enumerate(mice):
        for sh_nr,sh in enumerate(shifts):
            grid_per_ses = []
            for s in mousedata[m][sh]:
                outcomes_raw = s["Outcome"].ravel().astype(np.float)
                respsides_raw = s["ResponseSide"].ravel().astype(np.float)
                n_trials = len(outcomes_raw)
                if m[0] == "V":
                     outcomes_raw = outcomes_raw[12:]
                     respsides_raw = respsides_raw[12:]
                     trial_select = np.arange(12,n_trials,1).astype(int)
                else:
                     trial_select = np.arange(0,n_trials,1).astype(int)
                outcomes = outcomes_raw[~np.isnan(outcomes_raw)]
                respsides = respsides_raw[~np.isnan(outcomes_raw)]

                grid_ids, catlevels = calculate_category_gridid_levelid(s, trial_select)
                grid_per_ses.append( make_category_grid(respsides, grid_ids) )
            grid_per_shift[m_nr,sh_nr,:,:] = np.nanmean( np.stack(grid_per_ses,axis=2), axis=2)
    return grid_per_shift

def calculate_boundary_angles_grid( gridmat, mice, shifts):
    boundary_angle = np.full((len(mice),len(shifts)), np.NaN)
    for m_nr,m in enumerate(mice):
        for sh_nr,sh in enumerate(shifts):
            category_grid = gridmat[m_nr,sh_nr,:,:]
            performance_mat = (category_grid+1) / 2
            bound_ang, (x_line,y_line) = calculate_boundary( performance_mat, normalize=True )
            boundary_angle[m_nr,sh_nr] = bound_ang
    return boundary_angle

def mean_curve_per_shift( mousedata, mice, shifts ):
    curve_per_shift = np.full( (len(mice), len(shifts), 12), np.NaN )
    for m_nr,m in enumerate(mice):
        for sh_nr,sh in enumerate(shifts):
            curve_per_ses = []
            for s in mousedata[m][sh]:
                outcomes_raw = s["Outcome"].ravel().astype(np.float)
                respsides_raw = s["ResponseSide"].ravel().astype(np.float)
                n_trials = len(outcomes_raw)
                if m[0] == "V":
                     outcomes_raw = outcomes_raw[12:]
                     respsides_raw = respsides_raw[12:]
                     trial_select = np.arange(12,n_trials,1).astype(int)
                else:
                     trial_select = np.arange(0,n_trials,1).astype(int)
                outcomes = outcomes_raw[~np.isnan(outcomes_raw)]
                respsides = respsides_raw[~np.isnan(outcomes_raw)]

                grid_ids, catlevels = calculate_category_gridid_levelid(s, trial_select)
                category_curve, category_performance, xlabels = make_category_level_curve(respsides, outcomes, catlevels)

                curve_per_ses.append( category_curve )
            curve_per_shift[m_nr,sh_nr,:] = np.nanmean( np.stack(curve_per_ses,axis=1), axis=1)
    return curve_per_shift

def calculate_curve_fits( curvemat, mice, shifts):
    xvalues = np.arange(12)
    curve_steepness = np.full((len(mice),len(shifts)), np.NaN)
    curve_fits = np.full((len(mice),len(shifts),12), np.NaN)
    for m_nr,m in enumerate(mice):
        for sh_nr,sh in enumerate(shifts):
            category_curve = curvemat[m_nr,sh_nr,:]
            params = get_bounded_sigmoid_fit(xvalues, category_curve)
            curve_fits[m_nr,sh_nr,:] = bounded_sigmoid(params, xvalues)
            curve_steepness[m_nr,sh_nr] = params[1]
    return curve_fits,curve_steepness
