#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23, 2017

@author: pgoltstein
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports

import numpy as np
import os, glob, sys
sys.path.append('../xx_analysissupport')
import CAgeneral, CAplot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# More functions to support data analysis


def model_load_data(name, R2="m", basepath="../../"):
    if name == "cat-trained":
        datafile = basepath+"data/p5_encodingmodel/full_model/alldata-wi-flexK-combMdl-cat-trained-trialrange-R2{}.npy".format(R2)
    if name == "shuffled-trials":
        datafile = basepath+"data/p5_encodingmodel/full_model/alldata-wi-flexK-combMdl-cat-trained-trialrange-R2{}-shtr.npy".format(R2)
    if name == "prot":
        datafile = basepath+"data/p5_encodingmodel/full_model/alldata-wi-flexK-catMdl-prot-trained-trialrange-R2{}.npy".format(R2)
    if name == "prot-shuffled-trials":
        datafile = basepath+"data/p5_encodingmodel/full_model/alldata-wi-flexK-catMdl-prot-trained-trialrange-R2{}-shtr.npy".format(R2)
    print("Loading data: {}".format(datafile))
    model_data = np.load(datafile, allow_pickle=True).item()
    return model_data


def model_load_data_unique_max_R2( predictor_group_names, basepath="../../" ):

    # Full model
    full_model_file = basepath+"data/p5_encodingmodel/delta_model/alldata-wi-flexK-combMdl-cat-trained-trialrange-R2m-shgr-Shuffle-no-groups.npy"
    print("Loading full model data: {}".format(full_model_file))
    full_model_data = np.load(full_model_file, allow_pickle=True).item()

    # Shuffled model
    shuffled_model_file = basepath+"data/p5_encodingmodel/delta_model/alldata-wi-flexK-combMdl-cat-trained-trialrange-R2m-shabgr-Shuffle-all-groups.npy"
    print("Loading shuffled model data: {}".format(shuffled_model_file))
    shuffled_model_data = np.load(shuffled_model_file, allow_pickle=True).item()

    # Loop predictor groups and calculate maximum and unique R2
    unique_R2 = {}
    maximum_R2 = {}
    unique_sign = {}
    maximum_sign = {}
    for predname in predictor_group_names:
        print("Loading predictor group: {}".format(predname))
        shgr_model_file = basepath+"data/p5_encodingmodel/delta_model/alldata-wi-flexK-combMdl-cat-trained-trialrange-R2m-shgr-{}.npy".format(predname)
        shabgr_model_file = basepath+"data/p5_encodingmodel/delta_model/alldata-wi-flexK-combMdl-cat-trained-trialrange-R2m-shabgr-{}.npy".format(predname)
        shgr_model_data = np.load(shgr_model_file, allow_pickle=True).item()
        shabgr_model_data = np.load(shabgr_model_file, allow_pickle=True).item()

        unique_R2[predname] = {}
        maximum_R2[predname] = {}
        unique_sign[predname] = {}
        maximum_sign[predname] = {}
        for area in shgr_model_data["R2"].keys():

            # Get means and confidence intervals per cell for this area
            full_model_R2 = full_model_data["R2"][area]["Full"]["mean"]
            full_model_R2_low95 = full_model_data["R2"][area]["Full"]["ci95low"]
            shuffled_model_R2 = full_model_data["R2"][area]["Full"]["mean"]
            shuffled_model_R2_high95 = full_model_data["R2"][area]["Full"]["ci95up"]

            shgr_R2 = shgr_model_data["R2"][area]["Full"]["mean"]
            shgr_R2_high95 = shgr_model_data["R2"][area]["Full"]["ci95up"]
            shabgr_R2 = shabgr_model_data["R2"][area]["Full"]["mean"]
            shabgr_R2_low95 = shabgr_model_data["R2"][area]["Full"]["ci95low"]

            # Calculate the unique and maximum predictor group R2
            unique_R2[predname][area] = full_model_R2-shgr_R2
            maximum_R2[predname][area] = shabgr_R2

            # significance of uniqueR2 -> shuffled group should be below the 95 lower confidence interval of the full model, and the 95 percent upper bound of the shuffled groups should be below the mean of the full model
            unique_sign[predname][area] = np.logical_and( shgr_R2<full_model_R2_low95, shgr_R2_high95<full_model_R2 )

            # significance of maximumR2 -> shuffle all but group should have a lower 95 confidence interval that is above the mean of the shuffle and the mean of the shuffle all but group should be above the upper 95 percent bound of the shuffle all groups
            maximum_sign[predname][area] = np.logical_and( shabgr_R2_low95>shuffled_model_R2, shabgr_R2>shuffled_model_R2_high95 )

    return maximum_R2, unique_R2, maximum_sign, unique_sign


def model_get_sign_neurons(R2, R2_sh, which="any", n_timepoints=3, component="Full"):
    if component == "Category":
        component = "Stimulus"
    if component == "First lick sequence":
        component = "Choice"
    bootR2 = np.array(R2[component]["mean"])
    bootR295low = np.array(R2[component]["ci95low"])
    shuffleR2 = np.array(R2_sh[component]["mean"])
    shuffleR295up = np.array(R2_sh[component]["ci95up"])
    sign_from_shuffle = np.logical_and( shuffleR2<bootR295low, bootR2>shuffleR295up )
    sign_pos_from_shuffle = np.logical_and( sign_from_shuffle, bootR2>0.0 )

    if n_timepoints == 2:
        sign_pos_from_shuffle = sign_pos_from_shuffle[:,[0,2]]

    if which == "individual":
        signtunedmat = sign_pos_from_shuffle
    else:
        if which == "any":
            signtuned = np.nansum(sign_pos_from_shuffle*1.0,axis=1) > 0
        if which == "stable":
            signtuned = np.nansum(sign_pos_from_shuffle*1.0,axis=1) == n_timepoints
        if which == "gained":
            signtuned = np.logical_and( sign_pos_from_shuffle[:,0]==False, sign_pos_from_shuffle[:,-1]==True )
        if which == "lost":
            signtuned = np.logical_and( sign_pos_from_shuffle[:,0]==True, sign_pos_from_shuffle[:,-1]==False )

        signtunedmat = np.zeros_like(sign_pos_from_shuffle)
        for tp in range(n_timepoints):
            signtunedmat[:,tp] = signtuned
    return signtunedmat


def model_get_fraction( signtuned, mouse_id, mergebs=False ):
    if signtuned.shape[1] == 2:
        fraction = np.array( (CAgeneral.mean_per_mouse(signtuned[:,0]*1.0,mouse_id),  CAgeneral.mean_per_mouse(signtuned[:,1]*1.0,mouse_id)) ).T
    elif signtuned.shape[1] == 3:
        fraction = np.array( (CAgeneral.mean_per_mouse(signtuned[:,0]*1.0,mouse_id), CAgeneral.mean_per_mouse(signtuned[:,1]*1.0,mouse_id), CAgeneral.mean_per_mouse(signtuned[:,2]*1.0,mouse_id)) ).T
        if mergebs:
            fraction = np.stack([np.nanmean(fraction[:,0:2],axis=1), fraction[:,2]])
    return fraction


def model_get_R2( R2, signtuned, mouse_id, permouse=False, negatives=0.0, mergebs=False ):
    R2 = np.array(R2)
    R2[R2<0] = negatives
    if signtuned.shape[1] == 2:
        R2 = R2[:,[0,2]]
    if permouse:
        if signtuned.shape[1] == 2:
            R2 = np.array( (CAgeneral.mean_per_mouse( R2[signtuned[:,0],0], mouse_id[signtuned[:,0]] ), CAgeneral.mean_per_mouse( R2[signtuned[:,1],1], mouse_id[signtuned[:,1]] )) ).T
        elif signtuned.shape[1] == 3:
            R2 = np.array( (CAgeneral.mean_per_mouse( R2[signtuned[:,0],0], mouse_id[signtuned[:,0]] ), CAgeneral.mean_per_mouse( R2[signtuned[:,1],1], mouse_id[signtuned[:,1]] ), CAgeneral.mean_per_mouse( R2[signtuned[:,2],2], mouse_id[signtuned[:,2]] )) ).T
            if mergebs:
                R2 = np.stack([np.nanmean(R2[:,0:2],axis=1), R2[:,2]])
    else:
        for tp in range(signtuned.shape[1]):
            R2[~signtuned[:,tp],tp] = np.NaN
        if mergebs and signtuned.shape[1] == 3:
            R2 = np.stack([np.nanmean(R2[:,0:2],axis=1), R2[:,2]])
    return R2


def load_resp_and_ampl_files( loc_name, dir_fr, dir_ampl ):
    # Load significant response thresholds from file
    filename = os.path.join(dir_fr, 'data_resp_'+loc_name+'.npy')
    print("Loading data from: {}".format(filename))
    data_dict = np.load(filename, allow_pickle=True).item()
    resp_left = data_dict["data"]["resp_left"]
    resp_right = data_dict["data"]["resp_right"]
    behavioral_performance = data_dict["data"]["behavioral_performance"]

    # Load amplitudes from file
    filename = os.path.join(dir_ampl, 'data_ampl_'+loc_name+'.npy')
    print("Loading data from: {}".format(filename))
    data_dict = np.load(filename, allow_pickle=True).item()
    ampl_left = data_dict["data"]["ampl_left"]
    ampl_right = data_dict["data"]["ampl_right"]

    return (resp_left,resp_right),(ampl_left,ampl_right),behavioral_performance

def clip_amplitude_outliers(ampl_mat,clip_neg_ampl_below,clip_pos_ampl_above):
    print("  - Clipped {} outliers (response < {}) from {} datapoints".format(np.nansum(ampl_mat<clip_neg_ampl_below),clip_neg_ampl_below, np.nansum(~np.isnan(ampl_mat))))
    ampl_mat[ampl_mat<clip_neg_ampl_below] = clip_neg_ampl_below
    print("  - Clipped {} outliers (response > {}) from {} datapoints".format(np.nansum(ampl_mat>clip_pos_ampl_above),clip_pos_ampl_above, np.nansum(~np.isnan(ampl_mat))))
    ampl_mat[ampl_mat>clip_pos_ampl_above] = clip_pos_ampl_above
    return ampl_mat

def responsive_per_timepoint( resp_all_stimuli, ampl_all_stimuli, min_resp_ampl, more_than_x_stimuli=None, equal_to_x_stimuli=None ):
    # Set all responsive detections with amplitude below threshold to zero
    resp_all_stimuli_local = resp_all_stimuli.copy()
    resp_all_stimuli_local[resp_all_stimuli_local < min_resp_ampl] = 0.0
    # Get a matrix which is True when neurons still significantly respond to more than x stimuli
    if equal_to_x_stimuli is not None:
        is_resp_anystim = np.sum(resp_all_stimuli_local>0.5, axis=2) == equal_to_x_stimuli
    else:
        is_resp_anystim = np.sum(resp_all_stimuli_local>0.5, axis=2) > (more_than_x_stimuli+0.5)
    return is_resp_anystim

def group_data_for_swarmplot( plot_data_mat, plot_data_mat2=None):
    n_locs = len(plot_data_mat)

    # Predefine output lists
    xdata = np.array([])
    ydata = np.array([])
    mdata = []
    sdata = []
    cdata = []
    if plot_data_mat2 is not None:
        mdata2 = []
        sdata2 = []
        cdata2 = []

    # Loop data list, calculate mean trace, sem trace, xvalues
    for l_nr in range(n_locs):
        n_mice = plot_data_mat[l_nr].shape[0]
        mean_data,sem_data,_ = CAgeneral.mean_sem( plot_data_mat[l_nr], axis=0 )
        xdata = np.append( xdata, np.zeros(n_mice) + l_nr, axis=0 )
        ydata = np.append( ydata, plot_data_mat[l_nr].ravel(), axis=0 )
        mdata.append( mean_data )
        sdata.append( sem_data )
        cdata.append( CAplot.colors[int(l_nr)] )

        if plot_data_mat2 is not None:
            n_mice2 = plot_data_mat2[l_nr].shape[0]
            mean_data2,sem_data2,_ = CAgeneral.mean_sem( plot_data_mat2[l_nr], axis=0 )
            xdata = np.append( xdata, np.zeros(n_mice) + l_nr, axis=0 )
            ydata = np.append( ydata, -1*plot_data_mat2[l_nr].ravel(), axis=0 )
            mdata2.append( mean_data2 )
            sdata2.append( sem_data2 )
            cdata2.append( CAplot.colors[int(l_nr)] )
    if plot_data_mat2 is None:
        return xdata,ydata,mdata,sdata,cdata
    else:
        return xdata,ydata,mdata,sdata,cdata,mdata2,sdata2,cdata2
