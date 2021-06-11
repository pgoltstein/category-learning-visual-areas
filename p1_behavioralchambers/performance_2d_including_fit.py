#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:35:35 2017

@author: pgoltstein
"""


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get imports
import numpy as np
import scipy.linalg
import sys
sys.path.append('../xx_analysissupport')

# Add local imports
import CAplot, CAgeneral
import klimbic

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load all data
mdata,run_nr,all_mice = klimbic.quick_load_behavioral_box_data_run1_run2()
n_mice = len(all_mice)
X = klimbic.GRATING["orientation_id"]
Y = klimbic.GRATING["spatialf_id"]
colormap = "RdBu_r"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate average performance and reaction time per stimulus
n_trials = {}
data = {}
data_mat = {}

# Loop mice
for m in all_mice:

    n_trials = np.zeros(49)
    performance = np.zeros(49)

    # Get data of highest CT
    ctdata = mdata[m][5]

    # Loop sessions
    for s_nr,x in enumerate(ctdata):

        # Loop trials and add data per trial (skipping incomplete and missed)
        for t,d,o in zip( x['trial target id'], x['trial dummy id'], x["trial outcome"] ):
            if o < 2:
                cat_level_t = int(klimbic.CATEGORY[m]["level"][t])
                if cat_level_t < 99:
                    n_trials[t] += 1
                    n_trials[d] += 1
                    if o == 0:
                        performance[t] += 1
                    else:
                        performance[d] += 1
    performance = np.divide(performance, n_trials)
    notNanIx = ~np.isnan(performance)
    data[m] = np.stack( (X[notNanIx], Y[notNanIx], performance[notNanIx]) ).T
    data_mat[m] = np.reshape(performance, (7,7))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display 2D of mice
perf_colors = CAplot.plt.get_cmap(colormap)
fig_mice = CAplot.init_figure(fig_size=(12,6))
panel_ids,n = CAplot.grid_indices( 8, n_columns=4 )
for m,py,px in zip( all_mice, panel_ids[0], panel_ids[1] ):

    # Fit plane and get line
    X = np.reshape(klimbic.GRATING["orientation_id"], (7,7))
    Y = np.reshape(klimbic.GRATING["spatialf_id"], (7,7))
    A = np.c_[data[m][:,0], data[m][:,1], np.ones(data[m].shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[m][:,2])    # coefficients
    # x_line = X[0,:]
    # y_line = (((-1*C[0]*x_line) -C[2])+0.5)/C[1]
    b_a, (x_line,y_line) = klimbic.calculate_boundary( data_mat[m] )

    # 2D plot
    ax = CAplot.plt.subplot2grid( n, (py,px) )
    ax.plot(x_line-1, y_line-1, color="#000000", linewidth=1)

    CAplot.plt.imshow(data_mat[m], cmap=colormap, vmin=0.0, vmax=1.0)
    CAplot.plt.title(m, fontsize=10)
    CAplot.plt.axis('off')

CAplot.finish_figure( filename="1e-BehavioralChambers-Colorbar-2Dplots-incl-fittedboundary", wspace=0.0, hspace=0.4 )


fig = CAplot.init_figure(fig_size=(16,8))
CAplot.plt.imshow(np.full((1,1),0.0), cmap=colormap, vmin=0.0, vmax=1.0)
CAplot.plt.title("This plot is only here\nso that the colorbar can be used in the figure ...")
cbar = CAplot.plt.colorbar()
CAplot.plt.axis('off')
CAplot.finish_figure( filename="1e-BehavioralChambers-Colorbar-2Dplots", wspace=0.0, hspace=0.4 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()
