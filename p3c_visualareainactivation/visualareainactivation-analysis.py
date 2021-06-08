#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Oct 22, 2020

@author: pgoltstein
"""


# Imports
import warnings
import numpy as np
import visualareainactivationfunctions as vaif

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
warnings.filterwarnings('ignore')

# Include these data
mice = ["V05","V08","V09"] # Only mice that learned the task
areas = ["V1","AL","POR"]
conditions = ["Control","aCSF","Muscimol"]

# Load data
mousedata,datedata = vaif.load_mouse_data( vaif.datapath, mice, areas, conditions )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show mean performance per shift
p_per_area_cond = vaif.mean_per_area_condition( mousedata, mice, areas, conditions )
vaif.barplot_area_cond( p_per_area_cond, areas, conditions, ylabel="Fraction correct", y_minmax=[0,1], y_step=[0.25,2], savename="3ED5d-performance" )

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show the 2d stim grids
grid_per_area_cond = vaif.mean_grid_per_area_cond( mousedata, mice, areas, conditions )
# vaif.plot_grids_mouse_area_cond(grid_per_area_cond, mice, areas, conditions)
vaif.plot_grids_area_cond(grid_per_area_cond, mice, areas, conditions)

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show the 1d categorization curves
category_curve_per_area_cond = vaif.mean_curve_per_area_cond( mousedata, mice, areas, conditions )
vaif.curveplots_area_cond( category_curve_per_area_cond, areas, conditions, "3ED5d-categorycurves" )

# Fit the curves and calculate the steepness
curve_fits_per_area_cond,curve_steepness_per_area_cond = vaif.calculate_curve_fits( category_curve_per_area_cond, mice, areas, conditions)
vaif.curveplots_area_cond( curve_fits_per_area_cond, areas, conditions, "3ED5d-categorycurvefits" )



#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# That's all folks!
vaif.plt.show()
