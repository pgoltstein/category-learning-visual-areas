#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28, 2018

Loaded data is in dict like this:
data_dict = {"R2": bR2, "weight": bweight, "gweight": bgweight, "mouse_id": mouse_id, "settings": settings }
bR2[area][group] = {'mean': np.array(bd.mean), 'ci95low': np.array(bd.ci95[0]), 'ci95up': np.array(bd.ci95[1]), 'ci99low': np.array(bd.ci99[0]), 'ci99up': np.array(bd.ci99[1])}

@author: pgoltstein
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get all imports
import numpy as np
import pandas as pd
import collections
import sys
sys.path.append('../xx_analysissupport')
import CAplot, CAgeneral, CAencodingmodel, CArec, CAstats
import CAanalysissupport as CAsupp
import warnings

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings, which are due to columns/rows full with NaN's in numpy matrices
warnings.filterwarnings('ignore')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Local functions

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


def test_before_after( data_bf, area, min_n_samples=5, paired=False, bonferroni=1, name1="bs", name2="lrn", suppr_out=False, datatype="Data"):
    if type(data_bf) is list:
        data1 = data_bf[0]
        data2 = data_bf[1]
    else:
        data1 = np.array(data_bf[:,0])
        data2 = np.array(data_bf[:,1])
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    meansem1 = mean_stderr(data1, axis=0)
    meansem2 = mean_stderr(data2, axis=0)
    if not suppr_out:
        print("    {}: {}={:5.3f} (+-{:5.3f}), {}={:5.3f} (+-{:5.3f}) (n={},{})".format( area, name1, meansem1.mean, meansem1.stderr, name2, meansem2.mean, meansem2.stderr, data1.size, data2.size))
        try:
            if paired:
                if datatype == "Fraction":
                    p = CAstats.report_chisquare_test( data1, data2, n_indents=6, bonferroni=bonferroni )
                else:
                    p = CAstats.report_wmpsr_test( data1, data2, n_indents=6, bonferroni=bonferroni )
            else:
                if datatype == "Fraction":
                    p = CAstats.report_chisquare_test( data1, data2, n_indents=6, bonferroni=bonferroni )
                else:
                    p = CAstats.report_mannwhitneyu_test( data1, data2, n_indents=6, bonferroni=bonferroni )
        except ValueError as e:
            print(e)
            p = 1.0
    return meansem1, meansem2, 1.0 if p<0.05 else 0.0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic settings
figpath = "../../figureout/"
areas = ['V1','LM','AL','RL','AM','PM','LI','P','POR']
mice = ['21a','21b','F02','F03','F04','K01','K02','K03','K06','K07']
n_areas = len(areas)
n_mice = len(mice)
min_n_samples = 10
CAplot.font_size = { "title": 10, "label": 8, "tick": 8, "text": 8, "legend": 8 }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load model data
model_data = CAsupp.model_load_data("cat-trained", R2="m")
model_data_sh = CAsupp.model_load_data("shuffled-trials", R2="m")
CAplot.print_dict(model_data["settings"])

R2 = model_data["R2"]
R2_sh = model_data_sh["R2"]
mouse_id = model_data["mouse_id"]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Prepare data dictionaries
full_R2_per_neuron_LG = []
full_R2_per_neuron_ST = []

# Loop areas and get R2s
for a_nr,area in enumerate(areas):
    print("Processing area {}".format(area))

    # Get significantly predictive neurons
    full_signtuned_lost = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="lost", n_timepoints=2, component="Full")
    full_signtuned_stable = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="stable", n_timepoints=2, component="Full")
    full_signtuned_gained = CAsupp.model_get_sign_neurons(R2[area], R2_sh[area], which="gained", n_timepoints=2, component="Full")

    # Get R2 of tuned neurons
    full_R2lost = CAsupp.model_get_R2( R2[area]["Full"]["mean"], full_signtuned_lost, mouse_id[area], permouse=False, negatives=0.0 )
    full_R2stable = CAsupp.model_get_R2( R2[area]["Full"]["mean"], full_signtuned_stable, mouse_id[area], permouse=False, negatives=0.0 )
    full_R2gained = CAsupp.model_get_R2( R2[area]["Full"]["mean"], full_signtuned_gained, mouse_id[area], permouse=False, negatives=0.0 )
    full_R2_per_neuron_LG += [np.stack([full_R2lost[:,0],full_R2gained[:,1]], axis=1),]
    full_R2_per_neuron_ST += [np.stack([full_R2stable[:,0],full_R2stable[:,1]], axis=1),]


fig,ax = CAplot.init_figure_axes(fig_size=(11,5))

ax = CAplot.plt.subplot2grid( (1,4), (0,0) )
x = []
y = []
d = []
v = []
for a_nr,area in enumerate(areas):
    data = (~np.isnan(full_R2_per_neuron_ST[a_nr]))*1.0
    frac = CAgeneral.mean_per_mouse( data[:,1], mouse_id[area] )
    if area in ["AL","RL","AM"]:
        x.append(frac.ravel())
        y.append(np.zeros_like(frac))
        d.append(frac.ravel())
    if area in ["LM","P","LI","POR"]:
        x.append(frac.ravel())
        y.append(np.zeros_like(frac)+1.0)
        v.append(frac.ravel())

x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
d = np.concatenate(d,axis=0)
v = np.concatenate(v,axis=0)
mat = np.stack([x,y],axis=1)

d,v,sign = test_before_after( [d,v], "Full model, stable fraction (dorsal vs ventral)", min_n_samples=min_n_samples, paired=False, bonferroni=1, name1="dorsal", name2="ventral", datatype="Data" )

CAplot.bar( 0, d.mean, d.stderr )
CAplot.bar( 1, v.mean, v.stderr )

df = pd.DataFrame(mat, columns=["D","S"])
CAplot.sns.swarmplot(data=df, ax=ax, y="D", x="S", color="#aaaaaa", size=2, linewidth=1, edgecolor="None")
CAplot.finish_panel( ax, title="Stable", ylabel="Fraction", xlabel="Stream", legend="off", y_minmax=[0,0.6], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55,x_ticks=np.arange(2), x_ticklabels=["D","V"], despine=True)


ax = CAplot.plt.subplot2grid( (1,4), (0,1) )
x = []
y = []
d = []
v = []
for a_nr,area in enumerate(areas):
    data = (~np.isnan(full_R2_per_neuron_LG[a_nr]))*1.0
    frac = CAgeneral.mean_per_mouse( data[:,0], mouse_id[area] )
    if area in ["AL","RL","AM"]:
        x.append(frac.ravel())
        y.append(np.zeros_like(frac))
        d.append(frac.ravel())
    if area in ["LM","P","LI","POR"]:
        x.append(frac.ravel())
        y.append(np.zeros_like(frac)+1.0)
        v.append(frac.ravel())

x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
d = np.concatenate(d,axis=0)
v = np.concatenate(v,axis=0)
mat = np.stack([x,y],axis=1)

d,v,sign = test_before_after( [d,v], "Full model, lost fraction (dorsal vs ventral)", min_n_samples=min_n_samples, paired=False, bonferroni=1, name1="dorsal", name2="ventral", datatype="Data" )

CAplot.bar( 0, d.mean, d.stderr )
CAplot.bar( 1, v.mean, v.stderr )

df = pd.DataFrame(mat, columns=["D","S"])
CAplot.sns.swarmplot(data=df, ax=ax, y="D", x="S", color="#aaaaaa", size=2, linewidth=1, edgecolor="None")
CAplot.finish_panel( ax, title="Lost", ylabel="Fraction", xlabel="Stream", legend="off", y_minmax=[0,0.6], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55,x_ticks=np.arange(2), x_ticklabels=["D","V"], despine=True)


ax = CAplot.plt.subplot2grid( (1,4), (0,2) )
x = []
y = []
d = []
v = []
for a_nr,area in enumerate(areas):
    data = (~np.isnan(full_R2_per_neuron_LG[a_nr]))*1.0
    frac = CAgeneral.mean_per_mouse( data[:,1], mouse_id[area] )
    if area in ["AL","RL","AM"]:
        x.append(frac.ravel())
        y.append(np.zeros_like(frac))
        d.append(frac.ravel())
    if area in ["LM","P","LI","POR"]:
        x.append(frac.ravel())
        y.append(np.zeros_like(frac)+1.0)
        v.append(frac.ravel())

x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
d = np.concatenate(d,axis=0)
v = np.concatenate(v,axis=0)
mat = np.stack([x,y],axis=1)

d,v,sign = test_before_after( [d,v], "Full model, gained fraction (dorsal vs ventral)", min_n_samples=min_n_samples, paired=False, bonferroni=1, name1="dorsal", name2="ventral", datatype="Data" )

CAplot.bar( 0, d.mean, d.stderr )
CAplot.bar( 1, v.mean, v.stderr )

df = pd.DataFrame(mat, columns=["D","S"])
CAplot.sns.swarmplot(data=df, ax=ax, y="D", x="S", color="#aaaaaa", size=2, linewidth=1, edgecolor="None")
CAplot.finish_panel( ax, title="Gained", ylabel="Fraction", xlabel="Stream", legend="off", y_minmax=[0,0.6], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55,x_ticks=np.arange(2), x_ticklabels=["D","V"], despine=True)


ax = CAplot.plt.subplot2grid( (1,4), (0,3) )
x = []
y = []
d = []
v = []
for a_nr,area in enumerate(areas):
    data = (~np.isnan(full_R2_per_neuron_LG[a_nr]))*1.0
    data_st = (~np.isnan(full_R2_per_neuron_ST[a_nr]))*1.0
    frac0 = CAgeneral.mean_per_mouse( data[:,0], mouse_id[area] )
    frac1 = CAgeneral.mean_per_mouse( data[:,1], mouse_id[area] )
    frac_st = CAgeneral.mean_per_mouse( data_st[:,1], mouse_id[area] )
    frac = (frac1+frac0) / frac_st
    if area in ["AL","RL","AM"]:
        x.append(frac.ravel())
        y.append(np.zeros_like(frac))
        d.append(frac.ravel())
    if area in ["LM","P","LI","POR"]:
        x.append(frac.ravel())
        y.append(np.zeros_like(frac)+1.0)
        v.append(frac.ravel())

x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
d = np.concatenate(d,axis=0)
v = np.concatenate(v,axis=0)
mat = np.stack([x,y],axis=1)

d,v,sign = test_before_after( [d,v], "Full model, (lost+gained)/stable fraction (dorsal vs ventral)", min_n_samples=min_n_samples, paired=False, bonferroni=1, name1="dorsal", name2="ventral", datatype="Data" )

CAplot.bar( 0, d.mean, d.stderr )
CAplot.bar( 1, v.mean, v.stderr )

df = pd.DataFrame(mat, columns=["D","S"])
CAplot.sns.swarmplot(data=df, ax=ax, y="D", x="S", color="#aaaaaa", size=2, linewidth=1, edgecolor="None")
CAplot.finish_panel( ax, title="Turnover ratio", ylabel="Fraction", xlabel="Stream", legend="off", y_minmax=[0,9], y_step=[2,0], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,1], x_step=None, x_margin=0.75, x_axis_margin=0.55,x_ticks=np.arange(2), x_ticklabels=["D","V"], despine=True)

CAplot.finish_figure( filename="5f-ED7ef-Encodingmodel-Fraction-DorsalVentral", wspace=0.8, hspace=0.8 )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fraction of significant neurons for full model, per area
fig,ax = CAplot.init_figure_axes(fig_size=(21,6))

ax = CAplot.plt.subplot2grid( (1,3), (0,0) )
x = []
y = []
for a_nr,area in enumerate(areas):
    data = (~np.isnan(full_R2_per_neuron_ST[a_nr]))*1.0
    frac = CAgeneral.mean_per_mouse( data[:,1], mouse_id[area] )
    x.append(frac.ravel())
    y.append(np.zeros_like(frac)+a_nr)
    d = mean_stderr(frac, axis=0)
    CAplot.bar( a_nr, d.mean, d.stderr )
x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
mat = np.stack([x,y],axis=1)
df = pd.DataFrame(mat, columns=["D","S"])
CAplot.sns.swarmplot(data=df, ax=ax, y="D", x="S", color="#aaaaaa", size=2, linewidth=1, edgecolor="None")
CAplot.finish_panel( ax, title="Stable", ylabel="Fraction", xlabel="Area", legend="off", y_minmax=[0,0.6], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,8], x_step=None, x_margin=0.75, x_axis_margin=0.55,x_ticks=np.arange(9), x_ticklabels=areas, despine=True)

ax = CAplot.plt.subplot2grid( (1,3), (0,1) )
x = []
y = []
for a_nr,area in enumerate(areas):
    data = (~np.isnan(full_R2_per_neuron_LG[a_nr]))*1.0
    frac = CAgeneral.mean_per_mouse( data[:,0], mouse_id[area] )
    x.append(frac.ravel())
    y.append(np.zeros_like(frac)+a_nr)
    d = mean_stderr(frac, axis=0)
    CAplot.bar( a_nr, d.mean, d.stderr )
x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
mat = np.stack([x,y],axis=1)
df = pd.DataFrame(mat, columns=["D","S"])
CAplot.sns.swarmplot(data=df, ax=ax, y="D", x="S", color="#aaaaaa", size=2, linewidth=1, edgecolor="None")
CAplot.finish_panel( ax, title="Lost", ylabel="Fraction", xlabel="Area", legend="off", y_minmax=[0,0.6], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,8], x_step=None, x_margin=0.75, x_axis_margin=0.55,x_ticks=np.arange(9), x_ticklabels=areas, despine=True)

ax = CAplot.plt.subplot2grid( (1,3), (0,2) )
x = []
y = []
for a_nr,area in enumerate(areas):
    data = (~np.isnan(full_R2_per_neuron_LG[a_nr]))*1.0
    frac = CAgeneral.mean_per_mouse( data[:,1], mouse_id[area] )
    x.append(frac.ravel())
    y.append(np.zeros_like(frac)+a_nr)
    d = mean_stderr(frac, axis=0)
    CAplot.bar( a_nr, d.mean, d.stderr )
x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
mat = np.stack([x,y],axis=1)
df = pd.DataFrame(mat, columns=["D","S"])
CAplot.sns.swarmplot(data=df, ax=ax, y="D", x="S", color="#aaaaaa", size=2, linewidth=1, edgecolor="None")
CAplot.finish_panel( ax, title="Gained", ylabel="Fraction", xlabel="Area", legend="off", y_minmax=[0,0.6], y_step=[0.2,1], y_margin=0.02, y_axis_margin=0.01, x_minmax=[0,8], x_step=None, x_margin=0.75, x_axis_margin=0.55,x_ticks=np.arange(9), x_ticklabels=areas, despine=True)

CAplot.finish_figure( filename="5ED7f-Encodingmodel-Fraction-Area", wspace=0.6, hspace=0.7 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()

# That's all folks!
