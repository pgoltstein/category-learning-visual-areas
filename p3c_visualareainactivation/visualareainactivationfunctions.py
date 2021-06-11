#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions and settings for inactivation analysis of revision experiment

Created on Thu Oct 22, 2020

@author: pgoltstein
"""


# Imports
import os, sys, glob
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg
import scipy.optimize
# import statsmodels.api as statsmodels
from scipy.io import loadmat

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
warnings.filterwarnings('ignore')

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

# Data path
datapath = "../../data/p3c_visualareainactivation"
figpath = "../../figureout/"

# Settings
all_orientations = [0,20,40,60,80,100,120]
all_spatialfreqs = [0.24,0.16,0.12,0.08,0.06,0.04]
colormap = "RdBu_r"
inact_colors = { "aCSF": "#000000", "Muscimol": "#FF0000", "Control": "#888888" }

# Functions
def load_mouse_data(datapath, mice, areas, conditions, conditions_names):
    print("\n ---LOADING DATA ---")
    mousedata = {}
    datedata = {}
    for m in mice:
        print("\n{}".format(m))
        matfiles = sorted(glob.glob(os.path.join(datapath,m,"*.mat")))
        datedata[m] = []
        mousedata[m] = {}
        for a in areas:
            mousedata[m][a] = {}
            for c in conditions:
                mousedata[m][a][c] = []

        for f_nr,f in enumerate(matfiles):
            print(f)
            matfile_contents = loadmat(f)
            condition = matfile_contents["S"]["Exp_Injection"][0,0][0]
            area = matfile_contents["S"]["Exp_Injection_Loc"][0,0][0]
            print("  - inactivation condition = {}".format(condition))
            print("  - inactivation area = {}".format(area))
            datedata[m].append(matfile_contents)
            if condition == "aCSF" or condition == "Muscimol":
                mousedata[m][area][condition].append(matfile_contents)
                mousedata[m][area]["Control"] = [loadmat(matfiles[f_nr-1]), loadmat(matfiles[f_nr+1])]

    return mousedata, datedata

def mean_per_area_condition( mousedata, mice, areas, conditions, conditions_names ):
    p_per_area_cond = np.full( (len(mice),len(areas),len(conditions)), np.NaN )
    for m_nr,m in enumerate(mice):
        print("  Mouse {}".format(m))
        for a_nr,a in enumerate(areas):
            print("  - Area {}".format(a))
            for c_nr,c in enumerate(conditions):
                print("    .{}: n={}".format(conditions_names[c_nr], len(mousedata[m][a][c])))
                p_per_ses = []
                for s in mousedata[m][a][c]:
                    outcomes = s["Outcome"].ravel().astype(np.float)
                    outcomes = outcomes[12:]
                    p_per_ses.append( np.nanmean( outcomes ) )
                p_per_area_cond[m_nr,a_nr,c_nr] = np.nanmean(p_per_ses)
    return p_per_area_cond

def calculate_curve_fits( curvemat, mice, areas, conditions):
    xvalues = np.arange(12)
    curve_steepness = np.full((len(mice),len(areas),len(conditions)), np.NaN)
    curve_fits = np.full((len(mice),len(areas),len(conditions),12), np.NaN)
    for m_nr,m in enumerate(mice):
        for a_nr,a in enumerate(areas):
            for c_nr,c in enumerate(conditions):
                category_curve = curvemat[m_nr,a_nr,c_nr,:]
                params = get_bounded_sigmoid_fit(xvalues, category_curve)
                curve_fits[m_nr,a_nr,c_nr,:] = bounded_sigmoid(params, xvalues)
                curve_steepness[m_nr,a_nr,c_nr] = params[1]
    return curve_fits,curve_steepness

def barplot_area_cond( datamat, areas, conditions, conditions_names, ylabel, y_minmax, y_step, savename ):
    fig,ax = init_figure_axes(fig_size=(14,6))
    all_data = {"Control": [], "aCSF": [], "Muscimol": []}
    all_data_red = []
    for a_nr,a in enumerate(areas):
        print("\nArea {}".format(a))
        ax = plt.subplot2grid( (1,3), (0,a_nr) )
        m,e,n = mean_sem(datamat[:,a_nr,:],axis=0)
        for c_nr,c in enumerate(conditions):
            bar( c_nr, m[c_nr], e[c_nr] )
            mn = np.nanmean(datamat[:,a_nr,c_nr])
            sd = np.nanstd(datamat[:,a_nr,c_nr])
            all_data[c].append(datamat[:,a_nr,c_nr]-mn)
            print('{} mean={:7.5f}, std={:7.5f}'.format( conditions_names[c_nr], mn, sd ))
        reduction = datamat[:,a_nr,1]-datamat[:,a_nr,2]
        mn = np.nanmean(reduction)
        sd = np.nanstd(reduction)
        all_data_red.append(reduction-mn)
        for m in range(datamat.shape[0]):
            plt.plot( datamat[m,a_nr,:], color="#aaaaaa", marker="o", linewidth=1, markerfacecolor="#aaaaaa", markersize=2, markeredgewidth=None, markeredgecolor=None )
        finish_panel( ax, ylabel=ylabel, xlabel="Condition", legend="off", y_minmax=y_minmax, y_step=y_step, y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,len(conditions)-1], x_ticks=list(range(len(conditions))), x_ticklabels=conditions_names, x_margin=0.75, x_axis_margin=0.55, x_tick_rotation=-45, despine=True)
    finish_figure( filename=os.path.join(figpath,savename), wspace=0.8, hspace=0.8 )

def mean_grid_per_area_cond( mousedata, mice, areas, conditions ):
    grid_per_area_cond = np.full( (len(mice), len(areas), len(conditions), len(all_spatialfreqs), len(all_orientations)), np.NaN )
    for m_nr,m in enumerate(mice):
        for a_nr,a in enumerate(areas):
            for c_nr,c in enumerate(conditions):
                grid_per_ses = []
                for s in mousedata[m][a][c]:
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
                grid_per_area_cond[m_nr,a_nr,c_nr,:,:] = np.nanmean( np.stack(grid_per_ses,axis=2), axis=2)
    return grid_per_area_cond

def mean_curve_per_area_cond( mousedata, mice, areas, conditions ):
    curve_per_area_cond = np.full( (len(mice), len(areas), len(conditions), 12), np.NaN )
    for m_nr,m in enumerate(mice):
        for a_nr,a in enumerate(areas):
            for c_nr,c in enumerate(conditions):
                curve_per_ses = []
                for s in mousedata[m][a][c]:
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
                curve_per_area_cond[m_nr,a_nr,c_nr,:] = np.nanmean( np.stack(curve_per_ses,axis=1), axis=1)
    return curve_per_area_cond

def curveplots_area_cond( datamat, areas, conditions, conditions_names, savename ):
    xvalues = np.arange(12)
    xticklabels = []
    for x in range(-6,7,1):
        if x == 0:
            pass
        elif x%2 == 0:
            xticklabels.append(x)
        else:
            xticklabels.append("")
    fig,ax = init_figure_axes(fig_size=(3.5*(len(conditions)+2),4*len(areas)))
    for a_nr,a in enumerate(areas):
        for c_nr,c in enumerate(conditions):
            ax = plt.subplot2grid( (len(areas),len(conditions)+2), (a_nr,c_nr) )
            mn,sem,n = mean_sem( datamat[:,a_nr,c_nr,:], axis=0 )
            for m_nr in range(datamat.shape[0]):
                plt.plot( xvalues, datamat[m_nr,a_nr,c_nr,:], marker="o", markersize=1, color='#aaaaaa', linestyle='-', linewidth=1 )
            line( xvalues, mn, sem, line_color=inact_colors[c], sem_color=inact_colors[c] )
            finish_panel( ax, title="{}".format(conditions_names[c_nr]), ylabel="Fraction chosen", xlabel="Boundary distance", legend="off", y_minmax=[0,1], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,11], x_margin=0.55, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)

        ax = plt.subplot2grid( (len(areas),len(conditions)+2), (a_nr,3) )
        for c_nr,c in enumerate(conditions):
            mn,sem,n = mean_sem( datamat[:,a_nr,c_nr,:], axis=0 )
            line( xvalues, mn, sem, line_color=inact_colors[c], sem_color=inact_colors[c] )
        finish_panel( ax, title="{}".format(a), ylabel="Fraction chosen", xlabel="Boundary distance", legend="off", y_minmax=[0,1], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,11], x_margin=0.55, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)

        ax = plt.subplot2grid( (len(areas),len(conditions)+2), (a_nr,4) )
        for c_nr,c in enumerate(conditions):
            for m_nr in range(datamat.shape[0]):
                plt.plot( xvalues, datamat[m_nr,a_nr,c_nr,:], marker="", markersize=0, color=inact_colors[c], linestyle='-', linewidth=1 )
        finish_panel( ax, title="{}".format(a), ylabel="Fraction chosen", xlabel="Boundary distance", legend="off", y_minmax=[0,1], y_step=[0.25,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,11], x_margin=0.55, x_axis_margin=0.55, despine=True, x_ticks=xvalues, x_ticklabels=xticklabels)
    finish_figure( filename=os.path.join(figpath,savename), wspace=0.8, hspace=1.0 )

def plot_grids_mouse_area_cond(gridmat, mice, areas, conditions, conditions_names):
    fig,ax = init_figure_axes( fig_size=(3.5*len(areas)*len(conditions),3.5*len(mice)) )
    for m_nr,m in enumerate(mice):
        for a_nr,a in enumerate(areas):
            for c_nr,c in enumerate(conditions):
                ax = plt.subplot2grid( (len(mice),len(areas)*len(conditions)), (m_nr,c_nr+(a_nr*len(conditions))) )
                category_grid = gridmat[m_nr,a_nr,c_nr,:,:]

                if m == "V05" or m == "V09":
                    category_grid = np.flip(category_grid, axis=0)

                performance_mat = (category_grid+1) / 2
                b_a, (x_line,y_line) = calculate_boundary( performance_mat, normalize=True )
                plt.imshow(category_grid, cmap=colormap, vmin=-1.0, vmax=1.0)
                ax.plot(x_line, y_line, color="#000000")
                ax.set_xlim(-0.5,6.5)
                ax.set_ylim(-0.5,5.5)
                plt.axis('off')
                plt.title("{}-{}".format(a,conditions_names[c_nr][:4]), fontsize=font_size['title'])
    finish_figure( filename=os.path.join(figpath,"3ED5d-gridspermouse"), wspace=0.4, hspace=0.4 )

def plot_grids_area_cond(gridmat, mice, areas, conditions, conditions_names):
    fig,ax = init_figure_axes(fig_size=(3.5*len(conditions),3.5*len(areas)))
    for a_nr,a in enumerate(areas):
        for c_nr,c in enumerate(conditions):
            ax = plt.subplot2grid( (len(areas),len(conditions)), (a_nr,c_nr) )
            category_grid = np.zeros((6,7,len(mice)))
            for m_nr,m in enumerate(mice):
                mouse_grid = gridmat[m_nr,a_nr,c_nr,:,:]
                if m == "V05" or m == "V09":
                    mouse_grid = np.flip(mouse_grid, axis=0)
                category_grid[:,:,m_nr] = mouse_grid
            category_grid = np.nanmean(category_grid,axis=2)
            performance_mat = (category_grid+1) / 2
            b_a, (x_line,y_line) = calculate_boundary( performance_mat, normalize=True )
            plt.imshow(category_grid, cmap=colormap, vmin=-1.0, vmax=1.0)
            ax.plot(x_line, y_line, color="#000000")
            ax.set_xlim(-0.5,6.5)
            ax.set_ylim(-0.5,5.5)
            plt.axis('off')
            plt.title("{}-{}".format(a,conditions_names[c_nr][:4]), fontsize=font_size['title'])
    finish_figure( filename=os.path.join(figpath,"3ED5d-grids"), wspace=0.25, hspace=0.0 )

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
