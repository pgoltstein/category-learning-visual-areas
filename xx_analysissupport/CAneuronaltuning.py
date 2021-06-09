#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17, 2018

@author: pgoltstein
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports

import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for quantifying tuning

def istuned( tuningmatrix, alpha=0.01 ):
    """ Function compares if any of the rows in tuningcurve is significantly different from any other row using an anova
        - Inputs -
        tuningmatrix: 2D matrix of neuronal responses [stimulus x trial]
        alpha:        Significance threshold
        returns a tuple (True/False, p, F)
    """
    (F,p) = stats.f_oneway( *tuningmatrix )
    return True if p < alpha else False, p, F


def preferreddirection( tuningcurve, angles=None ):
    """ Returns the angle and index of the preferred direction
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        angles:      Array with angles (equal sampling across 360 degrees)
        returns tuple (angle, angle_ix)
    """
    # Calculate angles if not supplied
    if angles is None:
        angles = np.arange(0,360,360/tuningcurve.shape[0])

    # Find index of largest value
    pref_ix = np.argmax(tuningcurve)

    # Return angle and index of largest value
    return angles[pref_ix],pref_ix


def preferredorientation( tuningcurve, angles=None ):
    """ Returns the angle and index of the preferred orientation
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        angles:      Array with angles (equal sampling across 360 degrees)
        returns tuple (angle, angle_ix)
    """
    # Calculate angles if not supplied
    if angles is None:
        angles = np.arange(0,360,360/tuningcurve.shape[0])

    # Average across opposite directions to get orientation curve
    half_range = int(tuningcurve.shape[0]/2)
    orientationcurve = tuningcurve[:half_range]+tuningcurve[half_range:]

    # Find index of largest value
    pref_ix = np.argmax(orientationcurve)

    # Return angle and index of largest value
    return angles[pref_ix],pref_ix


def preferredspatialf( tuningcurve, spatialfs ):
    """ Returns the frequency and index of the preferred spatial frequency
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        angles:      Array with spatial frequencies
        returns tuple (spatial frequency, angle_ix)
    """
    # Find index of largest value
    pref_ix = np.argmax(tuningcurve)

    # Return angle and index of largest value
    return spatialfs[pref_ix],pref_ix


def resultant( tuningcurve, resultant_type, angles=None ):
    """ Calculates the resultant length and angle (using complex direction space, normalized to a range from 0.0 to 1.0, angle in degrees)
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        resultant_type: "direction" (default) or "orientation"
        angles:      Array with angles (equal sampling across 360 degrees)
        returns normalized resultant length,angle (np.float,np.float)
    """

    # Calculate angles if not supplied
    if angles is None:
        angles = np.arange(0,360,360/tuningcurve.shape[0])

    # Assign orientation multiplier
    ori_mult = 2.0 if resultant_type.lower()=="orientation" else 1.0

    # Set values below 0.0 to 0.0
    tuningcurve[tuningcurve<0.0] = 0.0

    # Initialize a list for our vector representation
    vector_representation = []

    # Iterate over response amplitudes and directions (in radians)
    for r,ang in zip(tuningcurve,np.radians(angles)):
        vector_representation.append( r * np.exp(ori_mult*complex(0,ang)) )

    # Convert the list to a numpy array
    vector_representation = np.array(vector_representation)

    # Mean resultant vector
    mean_vector = np.sum(vector_representation) / np.sum(tuningcurve)

    # Length of resultant
    res_length = np.abs(mean_vector)

    # Angle of resultant
    res_angle = np.mod(np.degrees(np.angle(mean_vector)), 360) / ori_mult

    # Return the length (absolute) of the resultant vector
    return res_length,res_angle


def orientation_selectivity_index( tuningcurve ):
    """ Returns a ratio-like orientation selectivity index (osi). Assumes equal sampling across angles.
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        returns osi (np.float)
    """

    # Average across opposite directions to get orientation curve
    half_range = int(tuningcurve.shape[0]/2)
    orientationcurve = tuningcurve[:half_range]+tuningcurve[half_range:]

    # Find index of largest value
    pref_ix = np.argmax(orientationcurve)

    # Find index of orthogonal
    orth_ix = int(np.mod( pref_ix + orientationcurve.shape[0]/2, orientationcurve.shape[0] ))

    # Calulate and return osi
    osi = (orientationcurve[pref_ix]-orientationcurve[orth_ix]) / (orientationcurve[pref_ix]+orientationcurve[orth_ix])

    # Return osi, but bound between 0 and 1
    return np.max([0.0, np.min([1.0, osi]) ])


def direction_selectivity_index( tuningcurve ):
    """ Returns a ratio-like direction selectivity index (dsi). Assumes equal sampling across angles.
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        returns dsi (np.float)
    """

    # Find index of largest value
    pref_ix = np.argmax(tuningcurve)

    # Find index of the null (opposite) direction
    null_ix = int(np.mod( pref_ix + tuningcurve.shape[0]/2, tuningcurve.shape[0] ))

    # Calulate dsi
    dsi = (tuningcurve[pref_ix]-tuningcurve[null_ix]) / (tuningcurve[pref_ix]+tuningcurve[null_ix])

    # Return dsi, but bound between 0 and 1
    return np.max([0.0, np.min([1.0, dsi]) ])


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


def twopeakgaussianfit( tuningcurve, angles ):
    """ Returns an array of size (360,) with the fitted tuning curve at 1 degree resolution
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        angles:      Array with angles (equal sampling across 360 degrees)
        returns fitted tuning curve (array of np.float)
    """

    # Function that wraps the xvalues in the Gaussian function to 0-180 degrees
    def wrap_x(x):
        return np.abs(np.abs(np.mod(x,360)-180)-180)

    # Function that returns the two peaked Gaussian
    def twopeakgaussian(x, Rbaseline, Rpref, Rnull, thetapref, sigma):
        return Rbaseline + Rpref*np.exp(-wrap_x(x-thetapref)**2/(2*sigma**2)) + Rnull*np.exp(-wrap_x(x+180-thetapref)**2/(2*sigma**2))

    # Calculate angles if not supplied
    if angles is None:
        angles = np.arange(0,360,360/tuningcurve.shape[0])

    # Get x-value range to consider
    x_values = np.arange(0,360,1)

    # -- Estimate parameters --

    # Baseline level of tuning curve
    Rbaseline = np.min(tuningcurve)

    # Preferred direction
    thetapref,pref_ix = preferreddirection(tuningcurve,angles)

    # Response amplitude to preferred direction
    Rpref = tuningcurve[pref_ix]

    # Response amplitude to null direction
    Rnull = tuningcurve[np.mod(pref_ix+int(angles.shape[0]/2),angles.shape[0])]

    # Estimate of tuning curve width
    sigma = halfwidthhalfmax(tuningcurve)

    # Merge all parameters in a tuple
    param_estimate = (Rbaseline, Rpref, Rnull, thetapref, sigma),

    # Fit parameters
    fitted_params,pcov = optimize.curve_fit( twopeakgaussian, angles, tuningcurve, p0=param_estimate)

    # Return fitted curve
    return twopeakgaussian(x_values,*fitted_params)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for getting data matices

def tuningmatrix(trace, stim_on, stim_id, frame_range, frame_range_bs):
    """ This function will create a 2D matrix that contains the mean single trial response, organized by stimulus id and trial number
        - inputs -
        trace:       1D array of data
        stim_on:     Array containing the frame indices for stimulus onsets
        stim_id:     Array containing the stimulus index per trial
        frame_range: Tuple containing the range of frames across which the stimulus response is averaged. Second value is not included anymore by itself. As in array[frame_range(0):frame_range(1)]
        frame_range_bs: Same as 'frame_range', but for baseline range that will be averaged and subtracted from the trial-trace-sections
    """

    stf = stimtrialframe(trace, stim_on, stim_id, frame_range, frame_range_bs)
    return np.mean(stf,axis=2)

def stimtrialframe(trace, stim_on, stim_id, frame_range, frame_range_bs):
    """ This function will create a 3D matrix that contains a short section of the data organized by stimulus id and trial number
        - inputs -
        trace:       1D array of data
        stim_on:     Array containing the frame indices for stimulus onsets
        stim_id:     Array containing the stimulus index per trial
        frame_range: Tuple containing the range of frames to include in the data section that is cut out. Second value is not included anymore by itself. As in array[frame_range(0):frame_range(1)]
        frame_range_bs: Same as 'frame_range', but for baseline range that will be averaged and subtracted from the trial-trace-sections
    """

    # Get an array with unique stimuli and counts of each unique stimulus
    unique_stimuli, unique_counts = np.unique(stim_id, return_counts=True)
    n_unique_stimuli = unique_stimuli.shape[0]
    n_trials = np.max(unique_counts)

    # Calculate the indices of frames around stimulus onset
    frame_indices = np.arange(frame_range[0],frame_range[1],1).astype(int)
    n_frames = frame_indices.shape[0]

    # Calculate the indices of frames of the baseline period
    baseline_indices = np.arange(frame_range_bs[0],frame_range_bs[1],1).astype(int)

    # Predefine the stf matrix
    stf = np.full( (n_unique_stimuli, n_trials, n_frames), np.NaN )

    # Define an array to count trials
    trial_count = np.zeros(n_unique_stimuli)

    # Loop all trials
    for fr_ix,st_ix in zip(stim_on,stim_id):

        # Find the trial count index
        tr_ix = trial_count[st_ix].astype(int)

        # Get a slice of the trace, offset by stimulus onset frame index
        stim_trace = trace[fr_ix+frame_indices]

        # Get a baseline slice of the trace, offset by stim onset frame index
        baseline_trace = trace[fr_ix+baseline_indices]

        # Subtract baseline value from the trace and store in stf
        stf[st_ix,tr_ix,:] = stim_trace - np.mean(baseline_trace)

        # Increase the trial counter for this stimulus
        trial_count[st_ix] += 1

    # Return the stf [stimulus,trial,frame] variable
    return stf
