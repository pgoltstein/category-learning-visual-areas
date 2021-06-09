#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 4, 2018

@author: pgoltstein
"""

########################################################################
### Imports
########################################################################

import numpy as np
import sys
sys.path.append('../xx_analysissupport')
import CAgeneral


########################################################################
### Functions for quantifying orientation tuningcurves
########################################################################

def orientation_tc( tuningcurve2d ):
    """ Converts the 2d tuning curve in a 1d orientation tuning curve """
    n_rows = tuningcurve2d.shape[0]
    max_rows = np.nanmax(tuningcurve2d, axis=1)
    max_row_ix = np.nanargmax(max_rows)
    max_row_range = max_row_ix + np.array([-1,0,1])
    max_row_range = max_row_range[max_row_range>=0]
    max_row_range = max_row_range[max_row_range<n_rows]
    return np.nanmean(tuningcurve2d[max_row_range,:],axis=0)

def preferred_direction( tuningcurve, angles ):
    """ returns the preferred direction in degrees (0-359)
        Angles: The angle per tuning curve entry (in degrees).
    """
    max_ix = np.argmax(tuningcurve)
    return angles[max_ix]

def preferred_orientation( tuningcurve, angles ):
    """ returns the preferred orientation in degrees (0-179)
        Angles: The angle per tuning curve entry (in degrees).
    """
    max_ix = np.argmax(tuningcurve)
    return np.mod(angles[max_ix],180)

def bandwidth( tuningcurve, angles ):
    """ returns the tuning curve half width at 1/sqrt(2) maximum
        Angles: The angle per tuning curve entry (in degrees).
    """

def OSI( tuningcurve, angles ):
    """ returns the tuning curve orientation selectivity index
        Angles: The angle per tuning curve entry (in degrees).
    """
    tc_halflen = int(len(tuningcurve)/2)
    tuningcurve = tuningcurve[:tc_halflen] + tuningcurve[tc_halflen:]
    angles = angles[:tc_halflen]
    max_ix = np.argmax(tuningcurve)
    null_ix = int(np.mod( max_ix+(0.5*len(angles)), len(angles) ))
    PREF = tuningcurve[max_ix]
    NULL = tuningcurve[null_ix]
    return max([ 0, min([1, (PREF-NULL)/(PREF+NULL)]) ])

def DSI( tuningcurve, angles ):
    """ returns the tuning curve direction selectivity index
        Angles: The angle per tuning curve entry (in degrees).
    """
    max_ix = np.argmax(tuningcurve)
    null_ix = int(np.mod( max_ix+(0.5*len(angles)), len(angles) ))
    PREF = tuningcurve[max_ix]
    NULL = tuningcurve[null_ix]
    return max([ 0, min([1, (PREF-NULL)/(PREF+NULL)]) ])

def CV( tuningcurve, angles=None ):
    """ returns the tuning curve circular variance
        Angles: The angle per tuning curve entry (in degrees).
    """

def image_grating( image_size=45, image_res=100, orientation=0, spatialf=0.04, mask_offset=None, mask_width=10, return_mask=False ):
    """ returns a matrix holding the image of an oriented grating
    image_size:     Size in degrees
    image_res:      Resolution in pixels
    orientation:    Angle of the bars in degrees
    spatialf:       Spatial frequency
    mask_offset:    Number of pixels to offset the mask (None = no mask)
    """

    orientation_rad = (orientation*np.pi)/180
    x_spf = np.cos(orientation_rad) * spatialf * 2 * np.pi
    y_spf = np.sin(orientation_rad) * spatialf * 2 * np.pi

    (x_deg,y_deg) = np.meshgrid( \
        np.linspace(-0.5*image_size, 0.5*image_size, image_res),
        np.linspace(-0.5*image_size, 0.5*image_size, image_res) )

    image = (1+np.sin( (x_spf*x_deg) + (y_spf*y_deg) ))/2

    if mask_offset is not None:
        mask = np.sqrt( np.power(x_deg,2) + np.power(y_deg,2) )
        im_middle_ix = int(np.round(mask.shape[0]/2))
        threshold_value1 = mask[im_middle_ix,mask_offset]
        threshold_value2 = mask[im_middle_ix,mask_offset+mask_width]
        mask = (mask-threshold_value1) / (threshold_value2-threshold_value1)
        mask[mask<0] = 0
        mask[mask>1] = 1
        image = image * mask
        image = np.abs(image-1)

    if return_mask:
        return image,mask
    else:
        return image
