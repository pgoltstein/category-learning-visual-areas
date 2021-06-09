#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23, 2017

@author: pgoltstein
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports

import numpy as np
import pandas as pd


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Class for handling bootstrapped statistics

class bootstrapped(object):
    """ This class provides bootstrapped statistics for the data sample that is supplied to the initialization routine """

    def __init__(self, data, axis=0, n_samples=1000, sample_size=None, multicompare=1):
        """ Sets the data sample and runs bootstrapping """

        # Get shape of output matrices
        shape = data.shape
        n = shape[axis]
        self._shape = tuple(np.delete(shape, axis))

        if len(data.shape) == 1 and data.shape[0] == 0:
            print("Warning: No data was supplied, bootstrap results in NaN's")
            self._mean = np.NaN
            self._stderr = np.NaN
            self._upper95 = np.NaN
            self._lower95 = np.NaN
            self._upper99 = np.NaN
            self._lower99 = np.NaN

        else:

            # If no sample size supplied, set it to the size of data
            if sample_size is None:
                sample_size = n

            # Get the bootstrap samples
            bootstraps = []
            for r in range(n_samples):
                random_sample = np.random.choice(n, size=sample_size, replace=True)
                bootstraps.append( np.nanmean( np.take(data, random_sample, axis=axis), axis=axis ) )
            bootstraps = np.stack(bootstraps,axis=0)

            # Correct thresholds for multiple comparison (Bonferroni)
            low95 = 2.5/multicompare
            up95 = 100.0 - low95
            low99 = 0.5/multicompare
            up99 = 100.0 - low95

            # Calculate the statistics
            self._mean = np.nanmean(bootstraps,axis=0)
            self._stderr = np.nanstd(bootstraps,axis=0)
            self._upper95 = np.nanpercentile(bootstraps,up95,axis=0)
            self._lower95 = np.nanpercentile(bootstraps,low95,axis=0)
            self._upper99 = np.nanpercentile(bootstraps,up99,axis=0)
            self._lower99 = np.nanpercentile(bootstraps,low99,axis=0)

    @property
    def shape(self):
        return self._shape

    @property
    def mean(self):
        return self._mean

    @property
    def stderr(self):
        return self._stderr

    @property
    def ci95(self):
        return self._lower95,self._upper95

    @property
    def ci99(self):
        return self._lower99,self._upper99


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# More general functions for data analysis

def mean_sem( datamat, axis=0 ):
    mean = np.nanmean(datamat,axis=axis)
    n = np.sum( ~np.isnan( datamat ), axis=axis )
    sem = np.nanstd( datamat, axis=axis ) / np.sqrt( n )
    return mean,sem,n

def remove_allNaNrow( data_mat, selector_mat=None ):
    if selector_mat is None:
        selector_mat = data_mat
    is_NaN_row = np.sum(np.isnan(selector_mat), axis=1) == selector_mat.shape[1]
    return np.delete(data_mat,np.argwhere(is_NaN_row),axis=0)

def zscore_1d( datavec ):
    mean = np.nanmean( datavec )
    std = np.nanstd( datavec )
    if std == 0:
        return datavec
    else:
        return (datavec-mean) / std

def add_per_mouse( df, datavec, baselinevec, data_name, area, mouse_ids, mouse_names, timepoint ):
    for m_id in np.unique(mouse_ids):
        m_df = pd.DataFrame()
        if baselinevec is None:
            m_df[data_name] = [ np.nanmean( datavec[mouse_ids==m_id] ), ]
        else:
            m_df[data_name] = [ np.nanmean( datavec[mouse_ids==m_id] - np.nanmean(baselinevec[mouse_ids==m_id]) ), ]
        m_df["Area"] = area
        m_df["Timepoint"] = timepoint
        m_df["Mouse"] = mouse_names[m_id]
        df = df.append( m_df )
    return df

def mean_per_mouse( datavec, mouse_ids, baselinevec=None, max_mice=10 ):
    data = np.full( (max_mice,), np.NaN )
    mouse_id_list = np.unique(mouse_ids)
    for m_id in mouse_id_list:
        if np.sum( (mouse_ids==m_id)*1.0 ) > 0:
            if baselinevec is None:
                data[int(m_id)] = np.nanmean( datavec[mouse_ids==m_id] )
            else:
                data[int(m_id)] = np.nanmean( datavec[mouse_ids==m_id] - baselinevec[mouse_ids==m_id] )
        else:
            data[int(m_id)] = np.NaN
    return data

def beh_per_mouse( data_dict, mouse_no_dict, timepoint=2, max_mice=10 ):
    beh_data = np.full( (max_mice,), np.NaN )
    mouse_name_list = np.unique(list(data_dict.keys()))
    for m_name in mouse_name_list:
        m_id = mouse_no_dict[m_name]
        beh_data[int(m_id)] = data_dict[m_name][timepoint]
    return beh_data

def add_per_neuron( df, datavec, baselinevec, data_name, area, mouse_ids, mouse_names, timepoint ):
    for m_id in np.unique(mouse_ids):
        m_df = pd.DataFrame()
        if baselinevec is None:
            m_df[data_name] = datavec[mouse_ids==m_id]
        else:
            m_df[data_name] = datavec[mouse_ids==m_id]-np.nanmean(baselinevec[mouse_ids==m_id])
        m_df["Area"] = area
        m_df["Timepoint"] = timepoint
        m_df["Mouse"] = mouse_names[m_id]
        df = df.append( m_df )
    return df

def recode_to_integer( vector1, vector2 ):
    """ Recodes two input vectors with arbitrary values to integers where the same arbitrary values get converted in the same integeres """
    if type(vector1) == list:
        vector1 = np.array(vector1)
    if type(vector2) == list:
        vector2 = np.array(vector2)
    concat_flat_unique = np.unique(np.concatenate([vector1.ravel(),vector2.ravel()]))
    r_vec1 = np.zeros_like(vector1)
    r_vec2 = np.zeros_like(vector2)
    for x,v in enumerate(vector1):
        r_vec1[x] = np.argwhere(concat_flat_unique==v)
    for x,v in enumerate(vector2):
        r_vec2[x] = np.argwhere(concat_flat_unique==v)
    return (r_vec1,r_vec2)

def rank_transform( matrix1, matrix2 ):
    """ Rank-transforms data, returns original shape """
    concat_flattened = np.concatenate([matrix1.ravel(),matrix2.ravel()])
    rank_data = np.arange(1,concat_flattened.shape[0]+1)
    concat_index = np.concatenate( [np.zeros_like(matrix1.ravel()), np.zeros_like(matrix2.ravel())+1] )
    sort_index = np.argsort(concat_flattened)
    sorted_rank_data = rank_data[sort_index]
    rank_matrix1 = np.reshape( sorted_rank_data[concat_index==0], matrix1.shape)
    rank_matrix2 = np.reshape( sorted_rank_data[concat_index==1], matrix2.shape)
    return (rank_matrix1,rank_matrix2)

def running_fractional_average( trace, fraction, window_len, sf ):
    window_size = int(window_len*sf)
    overall_median = np.median(trace)
    n_frames = trace.shape[0]

    # Pad data array
    padded_trace = np.concatenate( [ np.ones((int(window_size/2),))*overall_median, trace, np.ones((int(window_size/2),))*overall_median, ] )

    # set indices for frames around each frame
    baseline_values_per_frame_col = np.zeros((n_frames,window_size))
    for w in range(window_size):
        bs_ixs = np.arange(0,n_frames).astype(int) + w
        baseline_values_per_frame_col[:,w] = padded_trace[bs_ixs]
    sorted_trace_per_frame = np.sort(baseline_values_per_frame_col)
    perc_ix = np.round(fraction*window_size).astype(int)
    return sorted_trace_per_frame[:,perc_ix]

def recode_classes( datavec ):
    return ( 2 * ( (datavec-np.min(datavec)) / (np.max(datavec)-np.min(datavec)) ) ) - 1

def reduce_baseline_mouserec_matrix(mouserec_mat, normalize=False, bs_ixs=[2,4]):
    """ Reduces the mouse x recording matrix to 5 time points (averaging
        possible double in-task baseline time points)
        mouserec_mat:   Data matrix (mice x recordings)
        baseline:       False / subtract / divide
        bs_ixs:         Indices of baseline [from, to+1]
        returns a reduced, baselined mouserec_mat
    """
    n_mice,n_recs = mouserec_mat.shape
    if bs_ixs[1]-bs_ixs[0] > 1:
        n_recs_reduced = n_recs-1
        mouserec_reduced = np.full((n_mice,n_recs_reduced),np.NaN)
        for m in range(n_mice):
            mouserec_reduced[m,0:bs_ixs[0]] = mouserec_mat[m,0:bs_ixs[0]]
            mouserec_reduced[m,bs_ixs[0]:bs_ixs[1]] = \
                np.nanmean(mouserec_mat[m,bs_ixs[0]:bs_ixs[1]])
            mouserec_reduced[m,(bs_ixs[0]+1):] = mouserec_mat[m,bs_ixs[1]:]
    else:
        n_recs_reduced = n_recs
        mouserec_reduced = mouserec_mat

    if normalize is not None:
        if normalize.lower() in "subtract":
            BS = np.repeat( mouserec_reduced[:,bs_ixs[0]].reshape((-1,1)), \
                            n_recs_reduced, axis=1)
            mouserec_reduced = mouserec_reduced - BS
        if normalize.lower() in "divide":
            BS = np.repeat( mouserec_reduced[:,bs_ixs[0]].reshape((-1,1)), \
                            n_recs_reduced, axis=1)
            mouserec_reduced = (mouserec_reduced - BS) / BS

    # Appropriate labels for recordings
    xlabels = ['GR-BS1', 'GR-BS2', 'TSK-BS', 'TSK-LRN', 'GR-LRN']

    # Mean across mice
    return mouserec_reduced, xlabels


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Selectivity measurements

def calc_prot_select( left_tc, right_tc ):
    left_mean = np.nanmean(left_tc)
    right_mean = np.nanmean(right_tc)
    return np.minimum( np.maximum( \
        np.abs((left_mean-right_mean) / (left_mean+right_mean)),
        -1 ), 1 )

def calc_cat_select( left_tc, right_tc ):
    if np.sum(left_tc)+np.sum(right_tc) == 0:
        return 0
    D_left = np.abs(left_tc[:,None] - left_tc)
    D_right = np.abs(right_tc[:,None] - right_tc)
    within_diff = np.nanmean( np.concatenate( (
        D_left[np.triu_indices(D_left.shape[1],k=1)],
        D_right[np.triu_indices(D_right.shape[1],k=1)] ) ) )
    between_diff = np.nanmean(np.abs(left_tc[:,None] - right_tc))
    return np.maximum((between_diff-within_diff) / (between_diff+within_diff),0)

def sparseness( tuningcurve ):
    """ Returns the sparseness of the response distribution of an array / tuningcurve. According to B. Willmore & D.J. Tolhurst (2001). Characterizing the sparseness of neural codes. Neural Systems, 12:3, 255-270.
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        returns sparseness value (np.float)
    """

    # Number of stimuli
    n_stimuli = tuningcurve.shape[0]

    # Calculate the sum off all responses normalized to total sum, then squared
    sumR_pow2 = np.sum(np.abs(tuningcurve) / n_stimuli) ** 2

    # Calculate the sum over all squared responses, then normalize to total sum
    sum_Rpow2 = np.sum(tuningcurve ** 2) / n_stimuli

    # Return sparseness: 1 - (divide squared-sum over sum-squared)
    return 1 - (sumR_pow2 / sum_Rpow2)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Trial-response matrices etc.

def tm(data_mat, frame_ixs, frame_range,
        return_peak=False, include_bs=False, frame_range_bs=None):
    """ Returns an average response per trial matrix of period frame_ix """
    frame_indices = np.arange( frame_range[0], frame_range[1] ).astype(np.int)
    tm_mat = np.zeros((len(frame_ixs),data_mat.shape[1]))
    if not include_bs:
        for trial,frame_ix in enumerate(frame_ixs.astype(np.int)):
            if return_peak:
                tm_mat[trial,:] = data_mat[frame_ix+frame_indices,:].max(axis=0)
            else:
                tm_mat[trial,:] = data_mat[frame_ix+frame_indices,:].mean(axis=0)
        return tm_mat
    else:
        frame_indices_bs = np.arange( \
            frame_range_bs[0], frame_range_bs[1] ).astype(np.int)
        bs_mat = np.zeros((len(frame_ixs),data_mat.shape[1]))
        bs_std_mat = np.zeros((len(frame_ixs),data_mat.shape[1]))
        for trial,frame_ix in enumerate(frame_ixs.astype(np.int)):
            if return_peak:
                tm_mat[trial,:] = data_mat[frame_ix+frame_indices,:].max(axis=0)
            else:
                tm_mat[trial,:] = data_mat[frame_ix+frame_indices,:].mean(axis=0)
            bs_mat[trial,:] = data_mat[frame_ix+frame_indices_bs,:].mean(axis=0)
            bs_std_mat[trial,:] = data_mat[frame_ix+frame_indices_bs,:].std(axis=0)
        return tm_mat, bs_mat, bs_std_mat

def tuningcurves(tm, stimulus_ids):
    """ Returns a matrix (stimulus x neuron) """
    unique_stimulus_ixs = np.unique(stimulus_ids)
    n_stimuli = len(unique_stimulus_ixs)
    tuningcurves = np.zeros((n_stimuli,tm.shape[1]))
    for s_ix,s in enumerate(unique_stimulus_ixs):
        tuningcurves[s_ix,:] = tm[stimulus_ids==s,:].mean(axis=0)
    return tuningcurves

def tuningcurve_to_2d_grid( tc, stim_id_grid ):
    """ Map 1d response array/tuning curve to 2d grid """
    n_spatialf,n_direction = stim_id_grid.shape
    tm_2d = np.full((n_spatialf,n_direction),np.NaN)
    for f in range(n_spatialf):
        for d in range(n_direction):
            if np.isfinite(stim_id_grid[f,d]):
                tm_2d[f,d] = tc[int(stim_id_grid[f,d])-1]
    return tm_2d

def stm(tm, stimulus_ids):
    """ Returns a matrix (stimulus x trial x neuron) """
    unique_stimulus_ixs, stim_counts = \
        np.unique(stimulus_ids,return_counts=True)
    n_stimuli = len(unique_stimulus_ixs)
    max_n_trials = stim_counts.max()
    stm = np.full((n_stimuli,max_n_trials,tm.shape[1]),np.NaN)
    for s_ix,s in enumerate(unique_stimulus_ixs):
        stm[s_ix,0:stim_counts[s_ix],:] = tm[stimulus_ids==s,:]
    return stm

def psth(data_mat, frame_ixs, frame_range):
    """ Returns a peri-stimulus time histogram (trial,frame,neuron) """
    frame_indices = np.arange( frame_range[0], frame_range[1] ).astype(np.int)
    psth_mat = np.zeros((len(frame_ixs),data_mat.shape[1],len(frame_indices)))
    for trial,frame_ix in enumerate(frame_ixs.astype(np.int)):
        psth_mat[trial,:,:] = data_mat[frame_ix+frame_indices,:].transpose()
    return psth_mat

def tca_tensor(data_mat, frame_ixs, frame_range):
    """ Returns a peri-stimulus time histogram (neuron,frame,trial) """
    frame_indices = np.arange( frame_range[0], frame_range[1] ).astype(np.int)
    psth_mat = np.zeros((data_mat.shape[1],len(frame_indices),len(frame_ixs)))
    for trial,frame_ix in enumerate(frame_ixs.astype(np.int)):
        psth_mat[:,:,trial] = data_mat[frame_ix+frame_indices,:].transpose()
    return psth_mat
