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
import matdata

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load all data of info-integr pilot
all_mice,mdata = matdata.quick_load_full_infointegr_data()

# Add data of second behavioral experiment
all_mice2,mdata2 = matdata.quick_load_second_infointegr_data()
all_mice.extend(all_mice2)
mdata.update(mdata2)

all_mice = ['A77','C01','C04','C07','W06','W08','W09','W10']

colormap = "RdBu_r"
n_mice = len(all_mice)
X,Y = np.meshgrid(np.arange(7),np.arange(6))
X = np.reshape(X,(42,))
Y = np.reshape(Y,(42,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate average performance and reaction time per stimulus
n_trials = {}
data = {}
data_mat = {}

# Loop mice
for m in all_mice:

    n_trials = np.zeros((6,7))
    performance = np.zeros((6,7))

    # Get data of highest CT
    ctdata = []
    for ct in range(1,6):
        ctdata.extend(mdata[m]["sessions"][ct])

    print("Mouse {}".format(m))

    # Loop sessions
    for s_nr,x in enumerate(ctdata):

        if x["angleset"] == 0 and m[0] != 'W':
            continue

        outcome = x["outcome"]
        categoryid = x["categoryid"]
        angleix = x["angleix"]
        spatialfix = x["spatialfix"]

        # Loop trials and add data per trial (skipping incomplete and missed)
        for ou,ang,spf,cat in zip( outcome, angleix, spatialfix, categoryid ):
            if ou < 2:
                spf = int(spf)
                ang = int(ang)
                n_trials[spf,ang] += 1
                if ou == 1 and cat == 1: # Left
                    performance[spf,ang] += 1
                if ou == 0 and cat == 2: # Left
                    performance[spf,ang] += 1

    performance = np.divide(performance, n_trials)
    performance = np.reshape(performance,(42,))
    notNanIx = ~np.isnan(performance)
    len_perf = np.sum(notNanIx)
    data_mat[m] = performance
    data[m] = np.stack( (X[notNanIx], Y[notNanIx], performance[notNanIx]) ).T
    data_mat[m] = np.reshape(performance, (6,7))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display 2D of mice
perf_colors = CAplot.plt.get_cmap(colormap)
fig_mice = CAplot.init_figure(fig_size=(12,6))
panel_ids,n = CAplot.grid_indices( 8, n_columns=4 )
for m,py,px in zip( all_mice, panel_ids[0], panel_ids[1] ):

    # Fit plane and get line
    X,Y = np.meshgrid(np.arange(7),np.arange(6))
    A = np.c_[data[m][:,0], data[m][:,1], np.ones(data[m].shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[m][:,2])    # coefficients
    # x_line = X[0,:]
    # y_line = (((-1*C[0]*x_line) -C[2])+0.5)/C[1]
    b_a, (x_line,y_line) = matdata.calculate_boundary( data_mat[m], normalize=True )

    # 2D plot
    ax = CAplot.plt.subplot2grid( n, (py,px) )
    ax.plot(x_line, y_line, color="#000000", linewidth=1)

    CAplot.plt.imshow(1-data_mat[m], cmap=colormap, vmin=0.0, vmax=1.0)
    CAplot.plt.title(m, fontsize=10)
    CAplot.plt.axis('off')

CAplot.finish_figure( filename="2b-Headfixed-Performance-2Dplots-incl-fittedboundary", wspace=0.0, hspace=0.4 )


fig = CAplot.init_figure(fig_size=(12,6))
CAplot.plt.imshow(np.full((1,1),0.0), cmap=colormap, vmin=0.0, vmax=1.0)
CAplot.plt.title("This plot is only here so that the colorbar can be used in the figure ...")
cbar = CAplot.plt.colorbar()
CAplot.plt.axis('off')
CAplot.finish_figure( filename="2b-HeadFixed-Colorbar-2Dplots", wspace=0.2, hspace=0.4 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show
CAplot.plt.show()
