#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23, 2017

@author: pgoltstein
"""

########################################################################
### Imports
########################################################################

import numpy as np
import h5py
import os, glob, sys
from scipy import stats
from sklearn import svm
import CAgeneral
import CAplot
sys.path.append('../xx_analysissupport')

BEHAVIOR_CATEGORY_CHRONIC_INFOINT = "../../data/p3a_chronicimagingbehavior/performance_category_chronic_infointegr.npy"

import matdata


########################################################################
### Defaults
########################################################################

directions = np.arange(9,360,18)
spatialfs = np.array([0.04,0.06,0.08,0.12,0.16,0.24])
n_directions = directions.shape[0]
n_spatialfs = spatialfs.shape[0]

mouse_no = {"F02": 0, "F03": 1, "F04": 2, "21a": 3, "21b": 4, "K01": 5, "K02": 6, "K03": 7, "K06": 8, "K07": 9}
mouse_name = {0: "F02", 1: "F03", 2: "F04", 3: "21a", 4: "21b", 5: "K01", 6: "K02", 7: "K03", 8: "K06", 9: "K07"}
n_mice = 10

AREAMOUSE_BY_ID = {
    1: ("V1","21a"), 2: ("V1","21b"), 3: ("V1","F02"), 4: ("V1","F03"), 5: ("V1","F04"), 6: ("V1","K01"), 7: ("V1","K02"),
    8: ("LM","21a"), 9: ("LM","F03"), 10: ("LM","F04"), 11: ("LM","K03"),
    12: ("AL","21a"), 13: ("AL","F02"), 14: ("AL","F03"), 15: ("AL","F04"),
    16: ("RL","21b"), 17: ("RL","K01"), 18: ("RL","K02"), 19: ("RL","K03"), 20: ("RL","K06"), 21: ("RL","K07"),
    22: ("AM","21a"), 23: ("AM","21b"), 24: ("AM","K06"),
    25: ("PM","F02"), 26: ("PM","F03"), 27: ("PM","F04"), 28: ("PM","K03"), 29: ("PM","K06"), 30: ("PM","K07"),
    31: ("LI","21a"), 32: ("LI","21b"), 33: ("LI","K02"), 34: ("LI","K07"),
    35: ("P","21a"), 36: ("P","F02"), 37: ("P","K02"),
    38: ("POR","F02"), 39: ("POR","F03"), 40: ("POR","F04"), 41: ("POR","K01"), 42: ("POR","K03"), 43: ("POR","K06"), 44: ("POR","K07")
    }

########################################################################
### Functions for rec and chronicrecording classes
########################################################################

def create_boxcar_vector( indices, n_frames, amplitude=1.0):
    """ This function creates a 1 dimensional vector of length 'n_frames' that is 'high' at each index indicated by 'indices' """
    boxcar_vector = np.zeros(n_frames)
    if np.sum(indices>=n_frames) > 0:
        print("Detected {} boxcar indices > number of frames ({}); clipping these..".format(np.sum(indices>=n_frames), n_frames))
        indices = indices[indices<n_frames]
    for ix in indices:
        boxcar_vector[int(ix)] = amplitude
    return boxcar_vector

def create_nonmax_boxcar_vector( indices, n_frames, amplitude=1.0):
    """ This function creates a 1 dimensional vector of length 'n_frames' that is 'high' at each index indicated by 'indices' """
    boxcar_vector = np.zeros(n_frames)
    for ix in indices:
        boxcar_vector[int(ix)] += amplitude
    return boxcar_vector

def get_boundary_type(category_stimuli):
    """ Returns string describing type of boundary as: "orientation", "rule-bias" or "info-int"
    """
    min_left = np.min(category_stimuli['left']['dir'])
    max_left = np.max(category_stimuli['left']['dir'])
    min_right = np.min(category_stimuli['right']['dir'])
    max_right = np.max(category_stimuli['right']['dir'])
    if (max_left-min_left) < 45:
        return "orientation"
    elif (max_left-min_left) > 75:
        return "info-int"
    else:
        return "rule-bias"

def recode_category_stimuli(category_stimuli):
    """ Returns list of tuples (spatial frequency, orientation) with
        category stimuli recoded to integer space
    """
    ldir,rdir = CAgeneral.recode_to_integer( category_stimuli['left']['dir'], category_stimuli['right']['dir'] )
    lspf,rspf = CAgeneral.recode_to_integer( category_stimuli['left']['spf'], category_stimuli['right']['spf'] )
    return { 'left': { 'dir': ldir, 'spf': lspf }, 'right': { 'dir': rdir, 'spf': rspf } }

def calculate_boundary_distance(category_stimuli):
    """ Returns for each category stimulus the distance to the category boundary as fitted in 'fit_boundary_vector'
    """

    # Get recoded category stimuli
    cat_stims = recode_category_stimuli(category_stimuli)

    # Get points defining the boundary vector
    bound_vec = fit_boundary_vector(cat_stims)
    p1, p2 = bound_vec[0,:], bound_vec[1,:]

    # Calculate the distance for left category stimuli
    l_dist = np.zeros_like(cat_stims['left']['dir'])
    for nr,(d,s) in enumerate( zip( cat_stims['left']['dir'], cat_stims['left']['spf'] ) ):
        p3 = np.array([d,s])
        l_dist[nr] = np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)

    # Calculate the distance for right category stimuli
    r_dist = np.zeros_like(cat_stims['right']['dir'])
    for nr,(d,s) in enumerate( zip( cat_stims['right']['dir'], cat_stims['right']['spf'] ) ):
        p3 = np.array([d,s])
        r_dist[nr] = np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)

    # Return values in dictionary
    return {'left': l_dist, 'right': r_dist}

def fit_boundary_vector(cat_stimuli, C=10.0):
    """ returns two points defining a boundary vector, works well with integer recoded category stimuli """

    # Recode directions and spatial frequencies to integer numbers
    ldir,rdir = np.array(cat_stimuli['left']['dir']), np.array(cat_stimuli['right']['dir'])
    lspf,rspf = np.array(cat_stimuli['left']['spf']), np.array(cat_stimuli['right']['spf'])

    # Get minimum and maximum of integer space +- 1
    x_minmax = np.min(np.concatenate([ldir,rdir]))-1, np.max(np.concatenate([ldir,rdir]))+1
    y_minmax = np.min(np.concatenate([lspf,rspf]))-1, np.max(np.concatenate([lspf,rspf]))+1

    # Group data and category id]
    X = np.stack([np.concatenate([ldir,rdir]), np.concatenate([lspf,rspf])]).T
    Y = np.concatenate([np.zeros_like(ldir),np.zeros_like(rdir)+1])

    # Build svm
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X,Y)
    w = clf.coef_[0]

    # Calculate vector outer coordinates
    xx = np.linspace(x_minmax[0],x_minmax[1])
    yy = ((-w[0]*xx) - clf.intercept_[0]) / w[1]
    if yy.min() < y_minmax[0] or yy.max() > y_minmax[1]:
        yy = np.linspace(y_minmax[0],y_minmax[1])
        xx = ((w[1]*yy) + clf.intercept_[0]) / -w[0]

    # Return matrix with vectors in rows [[x1,y1],[x2,y2]]
    return np.array([ [xx[0],yy[0]], [xx[-1],yy[-1]] ])

def get_mouse_recording_sets( location_dir, check_list, n_missing_allowed=0 ):
    """ Returns a list with directories of complete recordings that
    have not more than n_missing_allowed missing time points """

    # Loop mice and check for presence of required recordings
    mouse_dirs = glob.glob(location_dir)
    mouse_recording_sets = []
    for m_nr,m_dir in enumerate(mouse_dirs):

        # Load chronic imaging recording from directory
        print("Checking: {}".format(m_dir))
        crec = chronicrecording(m_dir)

        # Select recordings to include
        checks_out = True
        n_missing = 0
        for r_name,r_nr in check_list:
            if len(crec.recs[r_name]) <= r_nr:
                n_missing += 1
                if n_missing > n_missing_allowed:
                    checks_out = False
        if checks_out:
            print(" -> Checks out, incuded recording-set ({} missing)".format(n_missing))
            mouse_recording_sets.append(m_dir)
        else:
            print(" !! Does not check out.")
    n_mice = len(mouse_recording_sets)
    n_recs = len(check_list)
    print("A total of {} mouse-recording sets check out".format(n_mice))
    return mouse_recording_sets,n_mice,n_recs


def complete_groups( rec_list ):
    """ Returns a list with group numbers of complete groups across recordings
    that are supplied in the rec_list """

    # Remove absent (None) recordings from list
    rec_list = [rec for rec in rec_list if rec is not None]

    # Get number of recordings
    n_recs = len(rec_list)

    # Get groups and maximum group number
    grp_lists = []
    grp_max = np.zeros((n_recs))
    for nr,rec in enumerate(rec_list):
        grp_lists.append( np.array(rec.groups) )
        grp_max[nr] = grp_lists[nr].max()
    max_grp_nmbr = grp_max.max().astype(int)

    # create matrix that contains ones when group is present
    grp_present_mat = np.zeros((max_grp_nmbr+1,n_recs))
    for nr,grp_list in enumerate(grp_lists):
        grp_present_mat[grp_list.astype(int),nr] = 1

    # Set the first row to zero to discard the '0' group
    grp_present_mat[0,:] = 0

    # Return list with numbers of complete groups
    return np.argwhere( np.sum(grp_present_mat,axis=1) == n_recs )


########################################################################
### Class that holds a chronic recording
########################################################################

class chronicrecording(object):
    """ Holds a chronic recording, based on hdf5 file and a matlab rec
        structure
    """

    def __init__(self, recording_dir, suppress_output=True):
        """ Initializes chronic recording, each item being associated with
            a hdf5 file, represented by the rec class
            recording_dir:  Path to recording files
        """

        # Initialize data holding dictionary
        self.recs = {   'Baseline Out-of-Task': [],
                        'Baseline Task': [],
                        'Not-learned Prototype Task': [],
                        'Learned Prototype Task': [],
                        'Not-learned Category Task': [],
                        'Learned Category Task': [],
                        'Not-learned Out-of-Task': [],
                        'Learned Out-of-Task': [] }

        # Find recording files in directory
        rec_files = glob.glob( os.path.join( recording_dir, '*.mat' ) )

        # Loop files and load recordings into 'rec' classes
        for nr,fn in enumerate(rec_files):
            this_rec = rec(filename=fn,readonly=True,suppress_output=True)
            if not suppress_output:
                print("{}) Timepoint: {} (mouse={}, date={})".format( \
                    nr, this_rec.timepoint, this_rec.mouse, this_rec.date ))
            self.recs[this_rec.timepoint].append( this_rec )

    @property
    def prototype_stimuli(self):
        """ Returns list of tuples (spatial frequency, orientation) with
            prototype stimuli
        """

        # Get an in-task category time point
        if len(self.recs["Learned Prototype Task"]) != 0:
            rec_list = self.recs["Learned Prototype Task"]
        else:
            rec_list = self.recs["Not-learned Prototype Task"]
        rec = rec_list[-1]

        # make dictionary with stimuli
        prot_stimuli = {"left": rec.left_category, "right": rec.right_category}

        # Add distance index
        prot_stimuli['left']['distix'] = [1.0,]
        prot_stimuli['right']['distix'] = [1.0,]

        # Return dictionary
        return prot_stimuli

    @property
    def shuffled_category_stimuli(self):
        """ Returns list of tuples (spatial frequency, orientation) with
            category stimuli, but shuffled between categories
        """
        cat_stims = self.category_stimuli
        all_dirs = np.concatenate( ( cat_stims["left"]["dir"], cat_stims["right"]["dir"] ) )
        all_spfs = np.concatenate( ( cat_stims["left"]["spf"], cat_stims["right"]["spf"] ) )

        leftright = np.zeros(10)
        leftright[:5] = 1
        one_to_five = np.concatenate((np.arange(5),np.arange(5)))
        shuffle_ix = np.random.choice(10,size=10,replace=False)
        leftright = leftright[shuffle_ix]
        one_to_five = one_to_five[shuffle_ix]
        for ix,(dir_,spf) in enumerate(zip(all_dirs,all_spfs)):
            c_ix = one_to_five[ix]
            if leftright[ix] == 0:
                cat_stims["left"]["dir"][c_ix] = dir_
                cat_stims["left"]["spf"][c_ix] = spf
            if leftright[ix] == 1:
                cat_stims["right"]["dir"][c_ix] = dir_
                cat_stims["right"]["spf"][c_ix] = spf
        return cat_stims

    @property
    def swapped_category_stimuli(self):
        """ Returns list of tuples (spatial frequency, orientation) with
            category stimuli, but shuffled between categories
        """
        cat_stims = self.category_stimuli
        swap3_leave2 = np.array([0,0,1,1,1])
        swap3_leave2 = swap3_leave2[np.random.choice(5,size=5,replace=False)]
        for ix in range(len(cat_stims["left"]["dir"])):
            if swap3_leave2[ix] == 1:
                left_dir = cat_stims["left"]["dir"][ix]
                left_spf = cat_stims["left"]["spf"][ix]
                cat_stims["left"]["dir"][ix] = cat_stims["right"]["dir"][ix]
                cat_stims["left"]["spf"][ix] = cat_stims["right"]["spf"][ix]
                cat_stims["right"]["dir"][ix] = left_dir
                cat_stims["right"]["spf"][ix] = left_spf
        return cat_stims

    @property
    def infointegr_category_stimuli(self):
        """ Returns list of tuples (spatial frequency, orientation) with
            category stimuli for the information integration boundary
        """
        ii_cat = {"right": {"dir": np.zeros(5), "spf": np.zeros(5), "distix": np.zeros(5)}, "left": {"dir": np.zeros(5), "spf": np.zeros(5), "distix": np.zeros(5)}}
        cat_stims = self.category_stimuli
        cat_dirs = np.append(cat_stims["left"]["dir"],cat_stims["right"]["dir"])
        cat_spfs = np.append(cat_stims["left"]["spf"],cat_stims["right"]["spf"])
        cat_dists = np.append(cat_stims["left"]["distix"],cat_stims["right"]["distix"])

        # Get stimuli from an in-task baseline category time point
        rec = self.recs["Baseline Task"][0]
        bs_stims = {"left": rec.left_category, "right": rec.right_category}

        # Loop category stimuli and sort according to baseline exp
        cnt = 0
        for d,s in zip(bs_stims["left"]["dir"],bs_stims["left"]["spf"]):
            for dd,ss,dst in zip(cat_dirs,cat_spfs,cat_dists):
                if dd == d and ss == s:
                    ii_cat["left"]["dir"][cnt] = dd
                    ii_cat["left"]["spf"][cnt] = ss
                    ii_cat["left"]["distix"][cnt] = dst
                    cnt += 1
                    break
        cnt = 0
        for d,s in zip(bs_stims["right"]["dir"],bs_stims["right"]["spf"]):
            for dd,ss,dst in zip(cat_dirs,cat_spfs,cat_dists):
                if dd == d and ss == s:
                    ii_cat["right"]["dir"][cnt] = dd
                    ii_cat["right"]["spf"][cnt] = ss
                    ii_cat["right"]["distix"][cnt] = dst
                    cnt += 1
                    break
        return ii_cat

    @property
    def category_stimuli(self):
        """ Returns list of tuples (spatial frequency, orientation) with
            category stimuli
        """

        # Get an in-task category time point
        if len(self.recs["Learned Category Task"]) != 0:
            rec_list = self.recs["Learned Category Task"]
        else:
            rec_list = self.recs["Not-learned Category Task"]
        rec = rec_list[-1]

        # Make dict with category stimuli
        cat_stimuli = {"left": rec.left_category, "right": rec.right_category}

        # Recode categories
        recoded_cat_stimuli = recode_category_stimuli(cat_stimuli)

        # Get boundary distance in matrix
        linear_value_mat = matdata.convert_recoded_cat_to_1Dvalue_matrix( recoded_cat_stimuli, mat_size=(5,6) )

        # Assign boundary distance to categories
        cat_stimuli['left']['distix'] = []
        for dx,sx in zip( recoded_cat_stimuli['left']['dir'], recoded_cat_stimuli['left']['spf'] ):
            cat_stimuli['left']['distix'].append(np.abs(linear_value_mat[int(sx),int(dx)] - 4.0 ))

        cat_stimuli['right']['distix'] = []
        for dx,sx in zip( recoded_cat_stimuli['right']['dir'], recoded_cat_stimuli['right']['spf'] ):
            cat_stimuli['right']['distix'].append(np.abs(linear_value_mat[int(sx),int(dx)] - 5.0 ))

        # Return dictionary with category stimuli
        return cat_stimuli

    @property
    def inferred_category_stimuli(self):
        """ Returns list of tuples (spatial frequency, orientation) with
            category stimuli as inferred by the mouse
        """

        # Get mouse name and category stimuli
        mouse = self.recs["Learned Category Task"][0].mouse
        cat_stimuli = self.category_stimuli
        recoded_cat_stimuli = recode_category_stimuli(cat_stimuli)

        # Load behavioral data of the period when mouse learned categories
        data_dict = np.load( BEHAVIOR_CATEGORY_CHRONIC_INFOINT ).item()
        performance = data_dict["performance"]
        mouse_names = data_dict["mouse_names"]

        # Calculate the boundary using the plane-fitting method
        b_a, (x_line,y_line), inferred_category_grid = matdata.calculate_boundary( performance[mouse], normalize=True, mat_size=(5,6), return_grid=True )
        inferred_category_grid[np.isnan(performance[mouse])] = np.NaN

        # Assign the spf & dir to the new categories
        inferred_left = {'dir': [], 'spf': []}
        inferred_right = {'dir': [], 'spf': []}
        for d,s,dx,sx in zip( cat_stimuli['left']['dir'], cat_stimuli['left']['spf'], recoded_cat_stimuli['left']['dir'], recoded_cat_stimuli['left']['spf'] ):
            if inferred_category_grid[int(sx),int(dx)] == 1.0:
                inferred_left['dir'].append(d)
                inferred_left['spf'].append(s)
            else:
                inferred_right['dir'].append(d)
                inferred_right['spf'].append(s)

        for d,s,dx,sx in zip( cat_stimuli['right']['dir'], cat_stimuli['right']['spf'], recoded_cat_stimuli['right']['dir'], recoded_cat_stimuli['right']['spf'] ):
            if inferred_category_grid[int(sx),int(dx)] == 1.0:
                inferred_left['dir'].append(d)
                inferred_left['spf'].append(s)
            else:
                inferred_right['dir'].append(d)
                inferred_right['spf'].append(s)

        inferred_cat_stimuli = {"left": inferred_left, "right": inferred_right}

        # Recode inferred categories
        recoded_inferred_cat_stimuli = recode_category_stimuli(inferred_cat_stimuli)

        # Get boundary distance in matrix
        linear_value_mat = matdata.convert_recoded_cat_to_1Dvalue_matrix( recoded_inferred_cat_stimuli, mat_size=(5,6) )

        # Assign boundary distance to categories
        inferred_cat_stimuli['left']['distix'] = []
        max_left_ix = np.max(linear_value_mat[inferred_category_grid==1.0])
        for dx,sx in zip( recoded_inferred_cat_stimuli['left']['dir'], recoded_inferred_cat_stimuli['left']['spf'] ):
            inferred_cat_stimuli['left']['distix'].append(np.abs(linear_value_mat[int(sx),int(dx)]-max_left_ix))

        inferred_cat_stimuli['right']['distix'] = []
        min_right_ix = np.min(linear_value_mat[inferred_category_grid==0.0])
        for dx,sx in zip( recoded_inferred_cat_stimuli['right']['dir'], recoded_inferred_cat_stimuli['right']['spf'] ):
            inferred_cat_stimuli['right']['distix'].append(np.abs(linear_value_mat[int(sx),int(dx)]-min_right_ix))

        return inferred_cat_stimuli


########################################################################
### Class that holds a single recording
########################################################################

class rec(object):
    """ Holds a single recording, based on hdf5 file and a matlab rec
        structure
    """

    def __init__(self, filename, readonly=True, suppress_output=False):
        """ Initializes a recording associated with a file
            filename:  Name of the file associated with this recording
        """

        # Read or create file
        if os.path.isfile(filename) and readonly:
            self.file = h5py.File(filename,'r')
            if not suppress_output:
                print("  Opened HDF5 file for reading:\n  -> {}".format(
                    filename))
        else:
            self.file = h5py.File(filename)
            if not suppress_output:
                print("  Opened HDF5 file for reading and writing:\n" +\
                    "  -> {}".format(filename))

        # Calculate properties
        n_frames_fr_on = self.file['rec/imag/FrameOnsets'].shape[0]
        n_frames_dror = self.file[self.file['rec']['L'][0,0]]['dRoR'].shape[0]
        if n_frames_fr_on > n_frames_dror:
            print("Frame-matrix has fewer frames ({}) than triggers actually recorded ({}).".format(n_frames_dror,n_frames_fr_on))
        self._n_frames = n_frames_dror

        # Set defaults
        self.neurons = np.arange(self.n_rois).astype(int)
        self._shuffle = False

    ####################################################################
    # Descriptive parameter handling
    @property
    def mouse(self):
        """ Returns the name of the mouse """
        mouse_ascii = self.file['rec/mouse']
        return ''.join(chr(c) for c in mouse_ascii)

    @property
    def mouse_no(self):
        """ Returns the number of the mouse """
        return mouse_no[self.mouse]

    @property
    def date(self):
        """ Returns the date as a string """
        date_ascii = self.file['rec/date']
        return ''.join(chr(c) for c in date_ascii)

    @property
    def sf(self):
        """ Returns the sampling frequency """
        return self.file['rec/imag/SamplingFreq'][0,0]

    @property
    def timepoint(self):
        """ Finds out what imaging timepoint the recording is """
        Mice   = [ \
                   'F02','F03','F04','21a','21b','K01','K02','K03','K04','K06','K07','K08' ]
        BSout  = [  209,  209,  209,  408,  408,  331,  401,  407,  414,  414,  414,  401  ]
        BSin   = [  220,  220,  220,  413,  413,  410,  411,  412,  424,  421,  426,  410  ]
        PROTnl = [  315,  314,  304,  523,  516,  520,  608,  602,  607,  602,  610,  625  ]
        CATnl  = [  421,  419,  423,  605,  528,  606,  704,  628,  615,  629,  705,  728  ]

        # Get index for the current mouse
        mouse_ix = Mice.index(self.mouse)

        # Determine whether 'In-Task' or 'Out-of-Task'
        if 'RightCat' in self.file['rec/stim/S']:
            n_cat_stim = self.file['rec/stim/S/RightCat/Angles'].shape[0]
            if n_cat_stim == 1:
                if int(self.date[-3:]) <= PROTnl[mouse_ix]:
                    return "Not-learned Prototype Task"
                else:
                    return "Learned Prototype Task"
            if n_cat_stim == 5:
                if int(self.date[-3:]) <= CATnl[mouse_ix]:
                    return "Not-learned Category Task"
                else:
                    return "Learned Category Task"
            if n_cat_stim == 6:
                return "Baseline Task"
        else:
            if int(self.date[-3:]) <= BSout[mouse_ix]:
                return "Baseline Out-of-Task"
            elif int(self.date[-3:]) <= CATnl[mouse_ix]:
                return "Not-learned Out-of-Task"
            else:
                return "Learned Out-of-Task"

    @property
    def n_trials(self):
        """ Returns the total number of visual stimulus trials """
        return self.file['rec/X/onFrames'].shape[0]

    @property
    def n_frames(self):
        """ Returns the total number of imaging frames """
        return self._n_frames

    @property
    def n_rois(self):
        """ Returns the total number of rois across layers """
        n_rois = 0
        for layer_obj in self.file['rec']['L'][:,0]:
            n_rois += self.file[layer_obj]['nROI'][0,0]
        return n_rois

    @property
    def groups(self):
        """ Returns the group numbers of all neurons """
        group_list = []
        layer_increment = 0
        for layer_obj in self.file['rec']['L'][:,0]:
            for roi_obj in self.file[layer_obj]['ROI/group'][:,0]:
                if self.file[roi_obj][0,0] > 0:
                    group_list.append( int( \
                        layer_increment+self.file[roi_obj][0,0] ) )
                else:
                    group_list.append( int( 0 ) )
            layer_increment += 1000
        return group_list


    ####################################################################
    ### Stimulus identity per trial
    @property
    def stimuli(self):
        """ Returns the list of unique stimulus id's """
        return self.file['rec/X/uStimId'][:,0]

    @property
    def direction(self):
        """ Returns a list with directions of the stimuli """
        if 'Angles' in self.file['rec/stim/S']:
            direction_list = self.file['rec/stim/S/Angles']
            return np.array(
                [direction_list[int(s)-1,0] for s in self.stimuli])
        else:
            direction_list = np.concatenate( ( \
                self.file['rec/stim/S/LeftCat/Angles'][:,0],
                self.file['rec/stim/S/RightCat/Angles'][:,0]), axis=0 )
            return np.array(
                [direction_list[int(s)-1] for s in self.stimuli])

    @property
    def direction_id(self):
        """ Returns a list with 'id' of the direction of the stimuli """
        (_,direction_id) = np.unique(self.direction,return_inverse=True)
        return direction_id

    @property
    def spatialf(self):
        """ Returns a list with spatial frequencies of the stimuli """
        if 'spatialF' in self.file['rec/stim/S']:
            spatialf_list = self.file['rec/stim/S/spatialF']
            return np.array(
                [spatialf_list[int(s)-1,0] for s in self.stimuli])
        else:
            spatialf_list = np.concatenate( ( \
                self.file['rec/stim/S/LeftCat/spatialF'][:,0],
                self.file['rec/stim/S/RightCat/spatialF'][:,0]), axis=0 )
            return np.array(
                [spatialf_list[int(s)-1] for s in self.stimuli])

    @property
    def spatialf_id(self):
        """ Returns a list with 'id' of the spatial frequency of the stimuli """
        (_,spatialf_id) = np.unique(self.spatialf,return_inverse=True)
        return spatialf_id

    def get_1d_stimulus_ix_in_2d_grid(self):
        """ Returns 2d grids with stimulus, direction, orientation ids """
        stim_2d = np.full((n_spatialfs,n_directions,),np.NaN)
        rec_spatialf = self.spatialf
        rec_directions = self.direction
        rec_stimuli = self.stimuli
        n_unique_stimuli = int(rec_stimuli.max())
        for s in range(1,n_unique_stimuli+1):
            s_ix = np.where(rec_stimuli==s)[0][0]
            dir = rec_directions[s_ix]
            sf = rec_spatialf[s_ix]
            dir_ix = np.where(directions==dir)[0][0]
            sf_ix = np.where(spatialfs==sf)[0][0]
            # print("{}: {}, dir={}, sf={}, dir_ix={}, sf_ix={}".format( \
            #     s, s_ix, dir, sf, dir_ix, sf_ix ))
            stim_2d[sf_ix,dir_ix] = s
        return stim_2d

    def select_stimuli(self, stimuli):
        """ Returns a list (#trials,) with nrs of included stimuli """
        stimulus_included = np.zeros(self.n_trials)
        dir_id = self.direction
        spf_id = self.spatialf
        stim_id = self.stimuli
        for t in range(self.n_trials):
            for d,s in zip( stimuli['dir'], stimuli['spf'] ):
                if dir_id[t] == d and spf_id[t] == s:
                    stimulus_included[t] = stim_id[t]
                    break
        return stimulus_included


    ####################################################################
    ### Category identity

    @property
    def left_category(self):
        """ Returns a list with (dir,spf) tuples of the left category stimuli
        """
        if 'LeftCat' in self.file['rec/stim/S']:
            direction = self.file['rec/stim/S/LeftCat/Angles'][:,0]
            spatialf = self.file['rec/stim/S/LeftCat/spatialF'][:,0]
            return {'dir': direction, 'spf': spatialf}
        else:
            return None

    @property
    def right_category(self):
        """ Returns a list with (dir,spf) tuples of the right category stimuli
        """
        if 'RightCat' in self.file['rec/stim/S']:
            direction = self.file['rec/stim/S/RightCat/Angles'][:,0]
            spatialf = self.file['rec/stim/S/RightCat/spatialF'][:,0]
            return {'dir': direction, 'spf': spatialf}
        else:
            return None

    def calculate_inferred_category_stimuli(self, normalize=False):
        """ Returns list of tuples (spatial frequency, orientation) with
            category stimuli as inferred from the mouses current performance
        """

        # Get trial identifiers and categories
        cat_stims = {"left": self.left_category, "right": self.right_category}
        outcomes = rec.outcome
        dirs = rec.direction
        spfs = rec.spatialf

        # Get average fraction of left choices for left category stimuli
        l_left_choices = np.zeros_like(cat_stims['left']['dir'])
        for nr,(d,s) in enumerate( zip( cat_stims['left']['dir'], cat_stims['left']['spf'] ) ):
            l_left_choices[nr] = np.nanmean(outcomes[np.logical_and(dirs==d,spfs==s)])

        # Get average fraction of left choices for right category stimuli
        r_left_choices = np.zeros_like(cat_stims['right']['dir'])
        for nr,(d,s) in enumerate( zip( cat_stims['right']['dir'], cat_stims['right']['spf'] ) ):
            r_left_choices[nr] = 1 -  np.nanmean(outcomes[np.logical_and(dirs==d,spfs==s)])

        # Rescale both left and right left-choice vectors
        all_choices = np.concatenate([l_left_choices,r_left_choices])
        min_,max_ = np.min(all_choices), np.max(all_choices)
        if normalize:
            l_left_choices = (l_left_choices-min_) / (max_-min_)
            r_left_choices = (r_left_choices-min_) / (max_-min_)

        # Shift values -exactly- in the middle to the minority group
        if 0.5 in l_left_choices:
            if sum(all_choices>0.5) > sum(all_choices<0.5):
                l_left_choices[l_left_choices==0.5] = 0.49
            else:
                l_left_choices[l_left_choices==0.5] = 0.51
        if 0.5 in r_left_choices:
            if sum(all_choices>0.5) > sum(all_choices<0.5):
                r_left_choices[r_left_choices==0.5] = 0.49
            else:
                r_left_choices[r_left_choices==0.5] = 0.51

        # Now add all stimuli that are above 0.5 to the left category and stimuli that are below 0.5 to the right category
        inferred_left = {'dir': [], 'spf': []}
        inferred_right = {'dir': [], 'spf': []}
        for nr,(d,s) in enumerate( zip( cat_stims['left']['dir'], cat_stims['left']['spf'] ) ):
            if l_left_choices[nr] > 0.5:
                inferred_left['dir'].append(d)
                inferred_left['spf'].append(s)
            else:
                inferred_right['dir'].append(d)
                inferred_right['spf'].append(s)
        for nr,(d,s) in enumerate( zip( cat_stims['right']['dir'], cat_stims['right']['spf'] ) ):
            if r_left_choices[nr] > 0.5:
                inferred_left['dir'].append(d)
                inferred_left['spf'].append(s)
            else:
                inferred_right['dir'].append(d)
                inferred_right['spf'].append(s)

        # Return (category dictionary, left-choice fraction dictionary)
        return { 'left': inferred_left, 'right': inferred_right }, { 'left': l_left_choices, 'right': r_left_choices }


    ####################################################################
    ### Category identity per trial

    def get_trial_category_id(self, cat_stimuli):
        """ Returns a list with the category id of each trial """
        category_id = np.zeros(self.n_trials)
        dir_id = self.direction
        spf_id = self.spatialf
        for t in range(self.n_trials):
            for d,s in zip( cat_stimuli['left']['dir'],
                            cat_stimuli['left']['spf'] ):
                if dir_id[t] == d and spf_id[t] == s:
                    category_id[t] = 1
                    break
            for d,s in zip( cat_stimuli['right']['dir'],
                            cat_stimuli['right']['spf'] ):
                if dir_id[t] == d and spf_id[t] == s:
                    category_id[t] = 2
                    break
        return category_id

    def get_trial_boundary_distance(self, cat_stimuli):
        """ Returns a list with the boundary distance of the stimulus at each trial """

        boundary_distance = np.zeros(self.n_trials)
        dir_id = self.direction
        spf_id = self.spatialf
        for t in range(self.n_trials):
            for d,s,b in zip( cat_stimuli['left']['dir'], cat_stimuli['left']['spf'], cat_stimuli['left']['distix'] ):
                if dir_id[t] == d and spf_id[t] == s:
                    boundary_distance[t] = b
                    break
            for d,s,b in zip( cat_stimuli['right']['dir'], cat_stimuli['right']['spf'], cat_stimuli['right']['distix'] ):
                if dir_id[t] == d and spf_id[t] == s:
                    boundary_distance[t] = b
                    break
        return boundary_distance


    ####################################################################
    ### Behavioral parameters per trial

    @property
    def outcome(self):
        """ Returns a list with outcome of each trial """
        if 'Outcome' in self.file['rec/stim']:
            # Check if baseline task
            if self.file['rec/stim/S/LeftCat/spatialF'].shape[0] == 6:
                outcome = self.file['rec/behav/FirstLickOutcomeIx'][:,0]
            else:
                outcome = self.file['rec/stim/Outcome'][:,0]
            return np.array(outcome)
        else:
            return np.full_like(self.stimuli,np.NaN)

    @property
    def lickside(self):
        """ Returns a list with lickside of each trial """
        if 'ResponseSide' in self.file['rec/stim']:
            lickside = self.file['rec/stim/ResponseSide'][:,0]
            return np.array(lickside)
        else:
            return np.full_like(self.stimuli,0)


    ####################################################################
    ### Indices of behavioral events

    @property
    def spontaneous_before(self):
        """ Returns the frames of spontaneous activity before task/stim """
        data_onset_frame = \
            int(self.file['rec/darkFrames/DataOnsetFrame'][:,0])-1
        task_start_frame = \
            int(np.argmax(np.array(self.file['rec/behav/TaskCh'])>0.7))
        return (data_onset_frame+100, task_start_frame-100)

    @property
    def spontaneous_after(self):
        """ Returns the frames of spontaneous activity after task/stim """
        task_channel = np.array(self.file['rec/behav/TaskCh'])
        task_end_frame = \
            int(len(task_channel)-np.argmax(task_channel[::-1]>0.4))
        data_end_frame = \
            int(len(task_channel))-1
        return (task_end_frame+100, data_end_frame-100)

    @property
    def vis_on_frames(self):
        """ Returns the frames at which the visual stimulus turned on """
        return self.file['rec/X/onFrames'][:,0].astype(int)-1

    @property
    def vis_off_frames(self):
        """ Returns the frames at which the visual stimulus turned off """
        return self.file['rec/X/offFrames'][:,0].astype(int)-1

    @property
    def left_lick_frames(self):
        """ Returns the frames at which left licks were detected """
        return self.file['rec/behav/LeftLickIx'][:,0].astype(int)-1

    @property
    def right_lick_frames(self):
        """ Returns the frames at which right licks were detected """
        return self.file['rec/behav/RightLickIx'][:,0].astype(int)-1

    @property
    def left_drop_frames(self):
        """ Returns the frames at which left drops were given """
        return self.file['rec/behav/LeftDropIx'][:,0].astype(int)-1

    @property
    def right_drop_frames(self):
        """ Returns the frames at which right drops were given """
        return self.file['rec/behav/RightDropIx'][:,0].astype(int)-1

    @property
    def response_win_on_frames(self):
        """ Returns the frames at which the response window started """
        return self.file['rec/behav/RespWinStartIx'][:,0].astype(int)-1

    @property
    def first_lick_frames(self):
        """ Returns the frame at which the first lick after stimulus presenation was detected """
        vis_stim_on = self.file['rec/X/onFrames'][:,0]-1
        resp_win_off = self.file['rec/behav/RespWinStopIx'][:,0]-1
        licks = self.file['rec/behav/LickIx'][:,0]-1
        first_licks = np.zeros_like(vis_stim_on)
        for tr,(v_on,r_off) in enumerate(zip(vis_stim_on,resp_win_off)):
            f_licks = licks[ np.logical_and(licks>=v_on,licks<=r_off) ]
            if f_licks.shape[0] > 0:
                first_licks[tr] = f_licks[0]
            else:
                first_licks[tr] = np.NaN
        return first_licks

    @property
    def first_lick_chosen_side_frames(self):
        """ Returns the frame at which the first lick on the chosen side, after stimulus presenation, was detected """
        vis_stim_on = self.file['rec/X/onFrames'][:,0]-1
        resp_win_off = self.file['rec/behav/RespWinStopIx'][:,0]-1
        left_licks = self.file['rec/behav/LeftLickIx'][:,0]-1
        right_licks = self.file['rec/behav/RightLickIx'][:,0]-1
        lickside = self.file['rec/stim/ResponseSide'][:,0]

        first_licks = np.zeros_like(vis_stim_on)
        for tr,(v_on,r_off,side) in enumerate(zip(vis_stim_on,resp_win_off,lickside)):

            if side == 1:
                f_licks = left_licks[ np.logical_and(left_licks>=v_on,left_licks<=r_off) ]
            elif side == 2:
                f_licks = right_licks[ np.logical_and(right_licks>=v_on,right_licks<=r_off) ]
            else:
                f_licks = np.array([])

            if f_licks.shape[0] > 0:
                first_licks[tr] = f_licks[0]
            else:
                first_licks[tr] = np.NaN
        return first_licks

    @property
    def first_lick_sequence_frames(self):
        """ Returns the frame at which the first lick of a sequence of minimally three licks on the chosen side, after stimulus presenation, was detected """
        vis_stim_on = self.file['rec/X/onFrames'][:,0]-1
        resp_win_on = self.file['rec/behav/RespWinStartIx'][:,0]-1
        resp_win_off = self.file['rec/behav/RespWinStopIx'][:,0]-1
        left_licks = self.file['rec/behav/LeftLickIx'][:,0]-1
        right_licks = self.file['rec/behav/RightLickIx'][:,0]-1
        lickside = self.file['rec/stim/ResponseSide'][:,0]

        first_licks = np.zeros_like(vis_stim_on)
        for tr,(v_on,r_on,r_off,side) in enumerate(zip(vis_stim_on,resp_win_on,resp_win_off,lickside)):

            # Get all licks on chosen side and all licks on rejected side
            if side == 1:
                choice_licks = left_licks[ np.logical_and(left_licks>=v_on,left_licks<=r_off) ]
                reject_licks = right_licks[ np.logical_and(right_licks>=v_on,right_licks<=r_off) ]
            elif side == 2:
                choice_licks = right_licks[ np.logical_and(right_licks>=v_on,right_licks<=r_off) ]
                reject_licks = left_licks[ np.logical_and(left_licks>=v_on,left_licks<=r_off) ]
            else:
                choice_licks = np.array([])
                reject_licks = np.array([])

            # If no licks on chosen side, add NaN regardless of reject licks
            if choice_licks.shape[0] == 0:
                first_licks[tr] = np.NaN

            # If only choice licks, take first choice lick
            elif choice_licks.shape[0] > 0 and reject_licks.shape[0] == 0:
                first_licks[tr] = choice_licks[0]

            # If less than 3 choice licks (but >0 reject licks), take last choice lick as choice
            elif choice_licks.shape[0] < 3:
                first_licks[tr] = choice_licks[-1]

            # Else, find the first lick of a sequence of three licks on the chosen side, that is not interupted by reject licks
            else:
                for l_nr in range(choice_licks.shape[0]-2):
                    ch1 = choice_licks[l_nr]
                    ch2 = choice_licks[l_nr+2]
                    n_in_between_rej_licks =  np.sum( np.logical_and(reject_licks>=ch1,reject_licks<=ch2) )
                    if n_in_between_rej_licks == 0:
                        break

                # Also get the deciding lick in the response window
                deciding_lick = choice_licks[choice_licks>=r_on]

                # Return the earliest lick of the two
                if deciding_lick.shape[0] == 0:
                    first_licks[tr] = choice_licks[l_nr]
                else:
                    first_licks[tr] = np.min((choice_licks[l_nr],deciding_lick[0]))

        return first_licks

    @property
    def trial_running_onset_frames(self):
        """ Returns the frame at which the running speed after stimulus increased over the 1.0 cm/s (WaitForNoLickSpeedHigh) threshold """
        vis_stim_on = self.file['rec/X/onFrames'][:,0]-1
        resp_win_off = self.file['rec/behav/RespWinStopIx'][:,0]-1
        speed = self.file['rec/behav/Speed'][:,0]
        run_onsets = np.zeros_like(vis_stim_on)
        for tr,(v_on,r_off) in enumerate(zip(vis_stim_on,resp_win_off)):
            speed_snap = speed[int(v_on):int(r_off)]
            if np.max(speed_snap) > 1.0:
                run_onsets[tr] = np.argmax(speed_snap > 1.0) + v_on
            else:
                run_onsets[tr] = np.NaN
        return run_onsets

    @property
    def choice_lick_frames(self):
        """ Returns the frame at which the first response-lick was detected """
        resp_win_on = self.file['rec/behav/RespWinStartIx'][:,0]-1
        resp_win_off = self.file['rec/behav/RespWinStopIx'][:,0]-1
        licks = self.file['rec/behav/LickIx'][:,0]-1
        choice_licks = np.zeros_like(resp_win_on)
        for tr,(r_on,r_off) in enumerate(zip(resp_win_on,resp_win_off)):
            r_licks = licks[ np.logical_and(licks>=r_on,licks<=r_off) ]
            if r_licks.shape[0] > 0:
                choice_licks[tr] = r_licks[0]
            else:
                choice_licks[tr] = np.NaN
        return choice_licks

    @property
    def trial_drop_frames(self):
        """ Returns per trial the frame at which the drop was detected """
        vis_stim_on = self.file['rec/X/onFrames'][:,0]-1
        resp_win_off = (self.file['rec/behav/RespWinStopIx'][:,0]-1)+10
        drops = self.file['rec/behav/DropIx'][:,0]-1
        trial_drops = np.zeros_like(vis_stim_on)
        for tr,(v_on,r_off) in enumerate(zip(vis_stim_on,resp_win_off)):
            t_drops = drops[ np.logical_and(drops>=v_on,drops<=r_off) ]
            if t_drops.shape[0] > 0:
                trial_drops[tr] = t_drops[0]
            else:
                trial_drops[tr] = np.NaN
        return trial_drops


    ####################################################################
    ### Average offsets between behavioral events in seconds

    def offset(self, event1, event2):
        """ Returns the average offset of event2 relative to event1 """
        event_dict = { "stimulus onset": self.vis_on_frames, "stimulus offset": self.vis_off_frames, "first lick": self.first_lick_frames, "first chosen lick": self.first_lick_chosen_side_frames, "first chosen lick sequence": self.first_lick_sequence_frames, "response window": self.response_win_on_frames, "choice lick": self.choice_lick_frames, "drop": self.trial_drop_frames, "trial run onset": self.trial_running_onset_frames }
        offset = np.nanmean(event_dict[event2.lower()]-event_dict[event1.lower()]) / self.sf
        if np.isnan(offset):
            offset = 0
        return offset


    ####################################################################
    ### Behavioral parameters and events per frame

    @property
    def speed(self):
        """ Returns the ball rotation speed per frame """
        return self.file['rec/behav/Speed'][:self.n_frames,0]

    def boxcar_predictor(self, name, learned_stimuli=None):
        """ Returns a vector with indices labeled 0 or 1, and a vector with the frame-stamps """

        # Stimulus onset
        if name.lower() == "stimulus onset":
            ixs = self.vis_on_frames
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "left category stimulus onset":
            learned_stim_ids = self.get_trial_category_id(learned_stimuli)
            ixs = self.vis_on_frames[learned_stim_ids==1]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "right category stimulus onset":
            learned_stim_ids = self.get_trial_category_id(learned_stimuli)
            ixs = self.vis_on_frames[learned_stim_ids==2]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "orientation stimuli onsets":
            directions = self.direction
            vis_on_frames = self.vis_on_frames
            regressors = []
            for d in np.unique(directions):
                ixs = vis_on_frames[directions==d]
                regressors.append( (create_boxcar_vector( indices=ixs, n_frames=self.n_frames ),ixs) )
            return regressors

        if name.lower() == "spatial frequency stimuli onsets":
            spatialf = self.spatialf
            vis_on_frames = self.vis_on_frames
            regressors = []
            for sf in np.unique(spatialf):
                ixs = vis_on_frames[spatialf==sf]
                regressors.append( (create_boxcar_vector( indices=ixs, n_frames=self.n_frames ),ixs) )
            return regressors

        if name.lower() == "individual stimuli onsets":
            cat_ids = self.get_trial_category_id(learned_stimuli)
            bound_dists = self.get_trial_boundary_distance(learned_stimuli)
            vis_on_frames = self.vis_on_frames
            # Make left negative, right positive
            bound_dists = ((bound_dists+1) * (-2*(cat_ids-1.5))).astype(int)
            regressors = []
            for bd in np.unique(bound_dists):
                ixs = vis_on_frames[bound_dists==bd]
                regressors.append( (create_boxcar_vector( indices=ixs, n_frames=self.n_frames ),ixs) )
            return regressors

        if name.lower() == "stimulus onset, go":
            ixs = self.vis_on_frames[ ~np.isnan(self.choice_lick_frames) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "stimulus onset, miss":
            ixs = self.vis_on_frames[ np.isnan(self.choice_lick_frames) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        # First lick
        if name.lower() == "first lick":
            choice_made_ix = self.first_lick_frames
            ixs = choice_made_ix[ ~np.isnan(choice_made_ix) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "left first lick":
            choice_made_ix = self.first_lick_frames
            ixs = choice_made_ix[ np.logical_and(~np.isnan(choice_made_ix),self.lickside==1) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "right first lick":
            choice_made_ix = self.first_lick_frames
            ixs = choice_made_ix[ np.logical_and(~np.isnan(choice_made_ix),self.lickside==2) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        # First lick sequence
        if name.lower() == "left first lick sequence":
            choice_made_ix = self.first_lick_sequence_frames
            ixs = choice_made_ix[ np.logical_and(~np.isnan(choice_made_ix),self.lickside==1) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "right first lick sequence":
            choice_made_ix = self.first_lick_sequence_frames
            ixs = choice_made_ix[ np.logical_and(~np.isnan(choice_made_ix),self.lickside==2) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        # First lick on chosen side
        if name.lower() == "first lick, chosen side":
            choice_made_ix = self.first_lick_chosen_side_frames
            ixs = choice_made_ix[ ~np.isnan(choice_made_ix) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "left first lick, chosen side":
            choice_made_ix = self.first_lick_chosen_side_frames
            ixs = choice_made_ix[ np.logical_and(~np.isnan(choice_made_ix),self.lickside==1) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "right first lick, chosen side":
            choice_made_ix = self.first_lick_chosen_side_frames
            ixs = choice_made_ix[ np.logical_and(~np.isnan(choice_made_ix),self.lickside==2) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        # Response window
        if name.lower() == "response window":
            ixs = self.response_win_on_frames
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "left category response window":
            learned_stim_ids = self.get_trial_category_id(learned_stimuli)
            ixs = self.response_win_on_frames[learned_stim_ids==1]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "right category response window":
            learned_stim_ids = self.get_trial_category_id(learned_stimuli)
            ixs = self.response_win_on_frames[learned_stim_ids==2]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "go, response window":
            ixs = self.response_win_on_frames[ ~np.isnan(self.choice_lick_frames) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "no-go, response window":
            ixs = self.response_win_on_frames[ np.isnan(self.choice_lick_frames) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "reward, response window":
            ixs = self.response_win_on_frames[~np.isnan(self.trial_drop_frames)]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "no reward, response window":
            ixs = self.response_win_on_frames[np.isnan(self.trial_drop_frames)]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        # Choice lick
        if name.lower() == "choice lick":
            choice_made_ix = self.choice_lick_frames
            ixs = choice_made_ix[ ~np.isnan(choice_made_ix) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "left choice lick":
            choice_made_ix = self.choice_lick_frames
            ixs = choice_made_ix[ np.logical_and(~np.isnan(choice_made_ix),self.lickside==1) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "right choice lick":
            choice_made_ix = self.choice_lick_frames
            ixs = choice_made_ix[ np.logical_and(~np.isnan(choice_made_ix),self.lickside==2) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "reward, choice lick":
            ixs = self.choice_lick_frames[~np.isnan(self.trial_drop_frames)]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "no reward, choice lick":
            ixs = self.choice_lick_frames[np.isnan(self.trial_drop_frames)]
            ixs = ixs[~np.isnan(ixs)]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        # Reward
        if name.lower() == "reward":
            ixs = self.trial_drop_frames[~np.isnan(self.trial_drop_frames)]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        # Lick (throughout whole experiment)
        if name.lower() == "lick":
            ixs = np.unique( np.concatenate( (self.left_lick_frames,self.right_lick_frames) ) )
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "left lick":
            ixs = self.left_lick_frames
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "right lick":
            ixs = self.right_lick_frames
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        # Drops/rewr throughout whole experiment
        if name.lower() == "drop":
            ixs = np.unique( np.concatenate( (self.left_drop_frames,self.right_drop_frames) ) )
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "left drop":
            ixs = self.left_drop_frames
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "right drop":
            ixs = self.right_drop_frames
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        # Speed / running
        if name.lower() == "speed":
            return self.speed,np.array([])

        if name.lower() == "trial running onset":
            run_onset_ix = self.trial_running_onset_frames
            ixs = run_onset_ix[ ~np.isnan(run_onset_ix) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "trial running onset, left choice":
            run_onset_ix = self.trial_running_onset_frames
            ixs = run_onset_ix[ np.logical_and(~np.isnan(run_onset_ix),self.lickside==1) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs

        if name.lower() == "trial running onset, right choice":
            run_onset_ix = self.trial_running_onset_frames
            ixs = run_onset_ix[ np.logical_and(~np.isnan(run_onset_ix),self.lickside==2) ]
            return create_boxcar_vector(indices=ixs, n_frames=self.n_frames),ixs


    ####################################################################
    ### neuron selector handling

    @property
    def neurons(self):
        """ Returns the list of selected neurons """
        return self._neurons

    @neurons.setter
    def neurons(self, selection):
        """ Sets the list of selected neurons """
        self._neurons = selection

    @property
    def neuron_groups(self):
        """ Returns the list of group numbers of selected neurons """
        neuron_groups_list = []
        groups_list = self.groups
        for nr in self._neurons:
            neuron_groups_list.append(groups_list[nr])
        return neuron_groups_list

    @neuron_groups.setter
    def neuron_groups(self, selected_groups):
        """ Sets the list of selected neurons by group number
        selected_groups: list of ints
        """
        if isinstance(selected_groups,int):
            selected_groups = [selected_groups,]
        selection = []
        groups = np.array(self.groups)
        for grp_nr in selected_groups:
            selection.append( int(np.argwhere( groups==grp_nr )) )
        self.neurons = selection


    ####################################################################
    # Data handling
    @property
    def shuffle(self):
        """ If set to true, the activity trace of each cell will be individually temporally shuffled """
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle_setting):
        """ Sets value of shuffle variable (string) """
        self._shuffle = shuffle_setting

    @property
    def spikes(self):
        """ Returns the spike matrix """
        spikes_list = []
        for layer_obj in self.file['rec']['L'][:,0]:
            spikes_list.append( self.file[layer_obj]['spike'][:,:] )
        spikes_list = np.concatenate( spikes_list, axis=1 )
        if self._shuffle is not False:
            if self._shuffle == "shuffle":
                print("Warning: Spike data is shuffled (per neuron) in time domain")
                for ix in range(spikes_list.shape[1]):
                    np.random.shuffle(spikes_list[:,ix])
            elif self._shuffle == "resample_from_baseline":
                print("Warning: Spike data is resampled (per neuron) in time domain from baseline values only (-22:-1 pre-visual-stimulus frames)")
                bs_frames = CAgeneral.psth( spikes_list, frame_ixs=self.vis_on_frames, frame_range=[-22,-1] )
                bs_frames = np.swapaxes(bs_frames,0,1)
                bs_frames = np.reshape( bs_frames, [bs_frames.shape[0], bs_frames.shape[1]*bs_frames.shape[2] ] )
                for ix in range(spikes_list.shape[1]):
                    spikes_list[:,ix] = np.random.choice( bs_frames[ix,:], size=(spikes_list.shape[0],) )
            elif "normal" in self._shuffle:
                trimmed_data = stats.trimboth(spikes_list, proportiontocut=0.01, axis=None)
                mu = trimmed_data.mean()
                sigma = trimmed_data.std()
                print("Warning: Spike data is randomly sampled from a {} distribution (databased: mu={}, sigma={})".format( self._shuffle, mu, sigma ))
                if self._shuffle == "normal":
                    spikes_list = np.random.normal(loc=mu, scale=sigma, size=spikes_list.shape)
                if self._shuffle == "lognormal":
                    spikes_list = np.random.lognormal(mean=mu, sigma=sigma, size=spikes_list.shape)
        return spikes_list[:,self.neurons]

    @property
    def dror(self):
        """ Returns the dr/r matrix """
        dror_list = []
        for layer_obj in self.file['rec']['L'][:,0]:
            dror_list.append( self.file[layer_obj]['dRoR'][:,:] )
        dror_list = np.concatenate( dror_list, axis=1 )
        if self._shuffle is not False:
            if self._shuffle == "shuffle":
                print("Warning: dR/R data is shuffled (per neuron) in time domain")
                for ix in range(dror_list.shape[1]):
                    np.random.shuffle(dror_list[:,ix])
            elif self._shuffle == "resample_from_baseline":
                print("Warning: dR/R data is resampled (per neuron) in time domain from baseline values only (-22:-1 pre-visual-stimulus frames)")
                bs_frames = CAgeneral.psth( dror_list, frame_ixs=self.vis_on_frames, frame_range=[-22,-1] )
                bs_frames = np.swapaxes(bs_frames,0,1)
                bs_frames = np.reshape( bs_frames, [bs_frames.shape[0], bs_frames.shape[1]*bs_frames.shape[2] ] )
                for ix in range(dror_list.shape[1]):
                    dror_list[:,ix] = np.random.choice( bs_frames[ix,:], size=(dror_list.shape[0],) )
            elif "normal" in self._shuffle:
                trimmed_data = stats.trimboth(dror_list, proportiontocut=0.01, axis=None)
                mu = trimmed_data.mean()
                sigma = trimmed_data.std()
                print("Warning: dR/R data is randomly sampled from a {} distribution (databased: mu={}, sigma={})".format( self._shuffle, mu, sigma ))
                if self._shuffle == "normal":
                    dror_list = np.random.normal(loc=mu, scale=sigma, size=dror_list.shape)
                if self._shuffle == "lognormal":
                    dror_list = np.random.lognormal(mean=mu, sigma=sigma, size=dror_list.shape)
        return dror_list[:,self.neurons]
