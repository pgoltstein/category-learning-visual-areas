#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script that analyzes retinotopy shift experiment, behavior only

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

# For behavior analysis use three shifts
shifts = [-26, 0, 26]
mousedata, eyedata = rtpf.load_mouse_data(datapath, mice, shifts, include_eye=False, im_nplanes=im_nplanes)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show mean performance per shift
p_per_shift = rtpf.mean_per_shift( mousedata, mice, shifts )
rtpf.barplot_shifts( p_per_shift, shifts, ylabel="Fraction correct", y_minmax=[0,1], y_step=[0.25,2], savename="2e-performance" )
rtpf.kruskalwallis_across_shifts(p_per_shift, shifts, "Performance per shift", posthoc_wmpsr_alternative="less")

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show the 2d stim grids
grid_per_shift = rtpf.mean_grid_per_shift( mousedata, mice, shifts )
# rtpf.plot_grids_mouse_shift(grid_per_shift, mice, shifts)
rtpf.plot_grids_shift(grid_per_shift, mice, shifts)

# Calculate the boundary angle
boundary_angle_per_shift = rtpf.calculate_boundary_angles_grid( grid_per_shift, mice, shifts)
boundary_angle_per_shift = np.abs(boundary_angle_per_shift)
delta_boundary_angle_per_shift = 45-boundary_angle_per_shift
rtpf.barplot_shifts( delta_boundary_angle_per_shift, shifts, ylabel="dBoundary angle", y_minmax=[0,45], y_step=[15,0], savename="2g-boundaryangle" )
rtpf.kruskalwallis_across_shifts(delta_boundary_angle_per_shift, shifts, "Boundary angle per shift", posthoc_wmpsr_alternative="greater")

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show the 1d categorization curves
category_curve_per_shift = rtpf.mean_curve_per_shift( mousedata, mice, shifts )
rtpf.curveplots_shifts( category_curve_per_shift, shifts, "2ED2f-categorycurves" )

# Fit the curves and calculate the steepness
curve_fits_per_shift,curve_steepness_per_shift = rtpf.calculate_curve_fits( category_curve_per_shift, mice, shifts)
rtpf.curveplots_shifts( curve_fits_per_shift, shifts, "2d-categorycurvefits" )
rtpf.barplot_shifts( curve_steepness_per_shift, shifts, ylabel="Curve steepness", y_minmax=[-1,0], y_step=[0.25,2], savename="2f-curvesteepness" )
rtpf.kruskalwallis_across_shifts(curve_steepness_per_shift, shifts, "Curve steepness per shift", posthoc_wmpsr_alternative="greater")


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# That's all folks!
rtpf.plt.show()
