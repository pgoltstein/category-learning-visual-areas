#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13, 2020

@author: pgoltstein
"""

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Get all imports
import numpy as np
import pandas as pd
import os, sys
sys.path.append('../xx_analysissupport')
import CAplot, CAgeneral, CAencodingmodel, CArec, CAstats
import CAanalysissupport as CAsupp
import CAorientationtuning as CAori
import CAneuronaltuning as CAtun
import matdata
import warnings

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings, which are due to columns/rows full with NaN's in numpy matrices
warnings.filterwarnings('ignore')

figpath = "../../figureout/"
base_path = "../../data/chronicrecordings"
model_basepath = "../../data/p5_encodingmodel/"
data_path = '../../data/p6a_encmodelcategorytuning/'


CAplot.font_size["title"] = 6
CAplot.font_size["label"] = 6
CAplot.font_size["tick"] = 6
CAplot.font_size["text"] = 6
CAplot.font_size["legend"] = 6

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Supporting functions
def realign_according_to_cat( tm, cat_grid ):
    # remove nan row
    if np.sum(np.isnan(cat_grid[-1,:])) == cat_grid.shape[1]:
        cat_grid = cat_grid[:-1,:]
        tm = tm[:-1,:]
    if np.sum(np.isnan(cat_grid[0,:])) == cat_grid.shape[1]:
        cat_grid = cat_grid[1:,:]
        tm = tm[1:,:]

    # Move to center
    ixs_list = np.array( list(range(20)) + list(range(20)) + list(range(20)) + list(range(20)) )
    cat_start = int(np.argwhere( np.nanmax(cat_grid,axis=0)>0 )[0])
    shift_amount = 20 - (7 - cat_start)
    shift_ixs = ixs_list[shift_amount:shift_amount+20]
    cat_grid = cat_grid[:,shift_ixs]
    tm = tm[:,shift_ixs]

    # Flip left category to left side
    if np.nanmax(cat_grid[:,7]) == 2:
        cat_grid = cat_grid[:,::-1]
        tm = tm[:,::-1]

    # Flip left category to be left top
    if cat_grid[0,7] == 1:
        cat_grid = cat_grid[::-1,:]
        tm = tm[::-1,:]

    # Return aligned tuning curves
    return tm,cat_grid

class mean_stderr(object):
        def __init__(self, data, axis=0):
            """ Sets the data sample and calculates mean and stderr """
            n = np.sum( ~np.isnan( data ), axis=axis )
            self._mean = np.nanmean( data, axis=axis )
            self._stderr = np.nanstd( data, axis=axis ) / np.sqrt( n )

        @property
        def mean(self):
            return self._mean

        @property
        def stderr(self):
            return self._stderr



#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Basic settings
load_instead_of_calc = True
areas = ['V1','LM','AL','RL','AM','PM','LI','POR']
# areas = ['POR']
mice = ['21a','21b','F02','F03','F04','K01','K02','K03','K06','K07']
directions = CArec.directions
spatialfs = CArec.spatialfs
orientations = directions[5:15]
n_areas = len(areas)
n_mice = len(mice)
n_timepoints = 5

cmap = CAplot.matplotlib.colors.ListedColormap(['#60c3db', '#f60951'])
bounds=[0,1.5,3]
norm = CAplot.matplotlib.colors.BoundaryNorm(bounds, cmap.N)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Load data
tm_data = np.load( os.path.join( data_path, "tuningcurvedata_spike_Stimulus.npy" ), allow_pickle=True ).item()
all_tm = tm_data["all_tm"]
all_selectivity = tm_data["all_selectivity"]
all_cti = tm_data["all_cti"]
all_cat_grid = tm_data["all_cat_grid"]


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Plot how preferred dir/spf shifts between baseline and learned
CatPDSFdist = {}
CatCV = {}
CatSA = {}
for a_nr,area in enumerate(areas):
    print("Loading area {}".format(area))

    tunmat = np.delete(all_tm[area], 3, 1)
    catgrid = np.delete(all_cat_grid[area], 3, 1)
    n_neurons,n_timepoints,n_spf,n_dir = tunmat.shape

    if n_neurons < 1:
        continue

    incl_tps = [0,1,4]
    ntp = len(incl_tps)
    PD = np.full((n_neurons,ntp), np.NaN)
    SP = np.full((n_neurons,ntp), np.NaN)
    CAT = np.full((n_neurons,ntp), np.NaN)
    CV = np.full((n_neurons,ntp), np.NaN)
    SA = np.full((n_neurons,ntp), np.NaN)
    PDSFdist = np.full((n_neurons,ntp), np.NaN)
    CM = []
    for nr in range(n_neurons):
        for tpnr,tp in enumerate(incl_tps):

            # If all-NaN -> continue
            if np.sum(np.isnan(tunmat[nr,tp,:,:])) == tunmat.shape[2] * tunmat.shape[3]:
                continue

            # Get realigned 2d tuning curve
            tc2d, C = realign_according_to_cat( tunmat[nr,tp,:,:], catgrid[nr,tp,:,:] )

            tc2d = tc2d[:,5:15] + np.concatenate([tc2d[:,:5],tc2d[:,15:]],axis=1)
            C = C[:,5:15] + np.concatenate([C[:,:5],C[:,15:]],axis=1)
            CM.append(C)

            # Flatten tuning curve to 1d
            tc_dir = CAori.orientation_tc( tc2d )
            tc_spf = CAori.orientation_tc( tc2d.T )

            # Calculate preferred direction and spatial frequency
            _,pdix = CAtun.preferreddirection( tc_dir, orientations )
            _,spix = CAtun.preferredspatialf( tc_spf, spatialfs )

            # Calculate circular variance and sparseness
            cv,_ = CAtun.resultant( tc_dir, "direction", angles=None )
            sparseness = CAtun.sparseness( tc2d.ravel() )

            # Calculate the distance of the PD to the category center
            PDSFdist[nr,tpnr] = np.sqrt( np.add( np.power(pdix-4.5,2), np.power(spix-2,2) ) )

            PD[nr,tpnr] = pdix
            SP[nr,tpnr] = spix
            CV[nr,tpnr] = cv
            SA[nr,tpnr] = sparseness
            CAT[nr,tpnr] = C[spix,pdix]

    if ntp == 3:
        PD = np.stack([ np.nanmean(PD[:,[0,1]],axis=1), PD[:,2] ], axis=1)
        SP = np.stack([ np.nanmean(SP[:,[0,1]],axis=1), SP[:,2] ], axis=1)
        CV = np.stack([ np.nanmean(CV[:,[0,1]],axis=1), CV[:,2] ], axis=1)
        SA = np.stack([ np.nanmean(SA[:,[0,1]],axis=1), SA[:,2] ], axis=1)
        CAT = np.stack([ np.nanmean(CAT[:,[0,1]],axis=1), CAT[:,2] ], axis=1)
        PDSFdist = np.stack([ np.nanmean(PDSFdist[:,[0,1]],axis=1), PDSFdist[:,2] ], axis=1)

    CatPDSFdist[area] = PDSFdist
    CatCV[area] = CV
    CatSA[area] = SA


fig,ax = CAplot.init_figure_axes(fig_size=(6,4))
x = []
y = []
for a_nr,area in enumerate(areas):
    data = CatPDSFdist[area][:,1]
    data = data - CatPDSFdist[area][:,0]
    data = data[~np.isnan(data)]
    x.append(data.ravel())
    y.append(np.zeros_like(data)+a_nr)
    d = mean_stderr(data)
    CAplot.bar( a_nr, d.mean, d.stderr )
x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
mat = np.stack([x,y],axis=1)
df = pd.DataFrame(mat, columns=["ddPDSF","Area"])
CAplot.sns.swarmplot(data=df, ax=ax, y="ddPDSF", x="Area", color="#888888", size=1, linewidth=0.5, edgecolor="None")
CAplot.finish_panel( ax, title="", ylabel="ddPDSF", xlabel="Area", legend="off", y_minmax=[-4,4], y_step=[2,0], y_margin=1.1, y_axis_margin=1.0, x_minmax=[0,7], x_step=None, x_margin=0.75, x_axis_margin=0.55, x_ticks=np.arange(8), x_ticklabels=areas, despine=True)
CAplot.finish_figure( filename="6ED10c-Tuningparameters-ChangeBsLrn-PrefDirPrefSpatf", path=figpath, wspace=0.5, hspace=0.5 )


fig,ax = CAplot.init_figure_axes(fig_size=(6,4))
x = []
y = []
for a_nr,area in enumerate(areas):
    data = CatSA[area][:,1]
    data = data - CatSA[area][:,0]
    data = data[~np.isnan(data)]
    x.append(data.ravel())
    y.append(np.zeros_like(data)+a_nr)
    d = mean_stderr(data)
    CAplot.bar( a_nr, d.mean, d.stderr )
x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
mat = np.stack([x,y],axis=1)
df = pd.DataFrame(mat, columns=["dSparseness","Area"])
CAplot.sns.swarmplot(data=df, ax=ax, y="dSparseness", x="Area", color="#888888", size=1, linewidth=0.5, edgecolor="None")
CAplot.finish_panel( ax, title="", ylabel="dSparseness", xlabel="Area", legend="off", y_minmax=[-0.4,0.4], y_step=[0.2,1], y_margin=0.11, y_axis_margin=0.1, x_minmax=[0,7], x_step=None, x_margin=0.75, x_axis_margin=0.55,x_ticks=np.arange(8), x_ticklabels=areas, despine=True)
CAplot.finish_figure( filename="6ED10d-Tuningparameters-ChangeBsLrn-Sparseness", path=figpath, wspace=0.5, hspace=0.5 )


fig,ax = CAplot.init_figure_axes(fig_size=(6,4))
x = []
y = []
for a_nr,area in enumerate(areas):
    data = CatCV[area][:,1]
    data = data - CatCV[area][:,0]
    data = data[~np.isnan(data)]
    x.append(data.ravel())
    y.append(np.zeros_like(data)+a_nr)
    d = mean_stderr(data)
    CAplot.bar( a_nr, d.mean, d.stderr )
x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
mat = np.stack([x,y],axis=1)
df = pd.DataFrame(mat, columns=["dCircVar","Area"])
CAplot.sns.swarmplot(data=df, ax=ax, y="dCircVar", x="Area", color="#888888", size=1, linewidth=0.5, edgecolor="None")
CAplot.finish_panel( ax, title="", ylabel="dCircVar", xlabel="Area", legend="off", y_minmax=[-0.4,0.4], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,7], x_step=None, x_margin=0.75, x_axis_margin=0.55,x_ticks=np.arange(8), x_ticklabels=areas, despine=True)
CAplot.finish_figure( filename="6ED10e-Tuningparameters-ChangeBsLrn-CircVar", path=figpath, wspace=0.5, hspace=0.5 )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show
CAplot.plt.show()

# That's all folks!
