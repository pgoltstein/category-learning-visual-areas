#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script that analyzes retinotopy shift experiment, including eye-tracking

Created on Thu Oct 22, 2020

@author: pgoltstein
"""


# Imports
import warnings
import numpy as np
import retinotopyfunctions as rtpf

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
warnings.filterwarnings('ignore')

# Include these data
mice = ["C01","C04","V05","V08","V09"]
im_nplanes = { "C01": 2, "C04": 2, "V05": 4, "V08": 4, "V09": 4 }

basepath="../.."
datapath = basepath+"/data/p2b_retinotopybehavior"

# For eye analysis, use two shifts
shifts = [-26, 26]
mousedata, eyedata = rtpf.load_mouse_data(datapath, mice, shifts, include_eye=True, im_nplanes=im_nplanes)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show mean performance per shift, as sanity-check
# p_per_shift = rtpf.mean_per_shift( mousedata, mice, shifts )
# rtpf.barplot_shifts( p_per_shift, shifts, ylabel="Fraction correct", y_minmax=[0,1], y_step=[0.25,2], savename="performance" )

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show mean pupil diameter per shift
d_per_shift = rtpf.eye_mean_per_shift( eyedata, mice, shifts, "eye1", "d", period=[0,1.5] )
rtpf.barplot_shifts( d_per_shift, shifts, ylabel="Pupil diam left", y_minmax=[0,1], y_step=[0.25,2], savename="2ED2i-pup-diam-left" )
rtpf.kruskalwallis_across_shifts(d_per_shift, shifts, "Pupil diam left, per shift", posthoc_wmpsr_alternative="less")

d_per_shift = rtpf.eye_mean_per_shift( eyedata, mice, shifts, "eye2", "d", period=[0,1.5] )
rtpf.barplot_shifts( d_per_shift, shifts, ylabel="Pupil diam right", y_minmax=[0,1], y_step=[0.25,2], savename="2ED2i-pup-diam-right" )
rtpf.kruskalwallis_across_shifts(d_per_shift, shifts, "Pupil diam right, per shift", posthoc_wmpsr_alternative="greater")

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show mean x-position per shift
d_per_shift = rtpf.eye_mean_per_shift( eyedata, mice, shifts, "eye1", "x", period=[0,1.5] )
rtpf.barplot_shifts( d_per_shift, shifts, ylabel="Pupil x-pos left", y_minmax=[0,1], y_step=[0.25,2], savename="2ED2h-pup-x-pos-left" )
rtpf.kruskalwallis_across_shifts(d_per_shift, shifts, "Pupil x-pos left, per shift", posthoc_wmpsr_alternative="less")

d_per_shift = rtpf.eye_mean_per_shift( eyedata, mice, shifts, "eye2", "x", period=[0,1.5] )
rtpf.barplot_shifts( d_per_shift, shifts, ylabel="Pupil x-pos right", y_minmax=[0,1], y_step=[0.25,2], savename="2ED2h-pup-x-pos-right" )
rtpf.kruskalwallis_across_shifts(d_per_shift, shifts, "Pupil x-pos right, per shift", posthoc_wmpsr_alternative="less")

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# That's all folks!
rtpf.plt.show()
