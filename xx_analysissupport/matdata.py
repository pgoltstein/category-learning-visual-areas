#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 2, 2017

Contains functions for loading matlab behavioral data of head-fixed category learning experiments

To get all info from hdf5 file
h5dump -n ./cat_behav_full_infointegr.mat

@author: pgoltstein
"""

import os
import h5py
import numpy as np
import scipy.optimize
import scipy.linalg
import glob, datetime


def get_category_stimuli( s ):
    left_cat = {'dir': s["leftcat-angles"], 'spf': s["leftcat-spatialf"]}
    right_cat = {'dir': s["rightcat-angles"], 'spf': s["rightcat-spatialf"]}
    return {"left": left_cat, "right": right_cat}


def convert_recoded_cat_to_1Dvalue_matrix( cat_recoded, mat_size=(5,6) ):
    linear_value_mat = np.full(mat_size,np.NaN)

    leftup_cat = False
    if 0 in cat_recoded["left"]["spf"] and 0 in cat_recoded["left"]["dir"]:
        leftup_cat = True
    if 0 in cat_recoded["right"]["spf"] and 0 in cat_recoded["right"]["dir"]:
        leftup_cat = True

    left_larger_right_cat = False
    if np.max(cat_recoded["left"]["dir"]) > np.max(cat_recoded["right"]["dir"]):
        left_larger_right_cat = True

    rule_cat = True
    if len(np.unique(cat_recoded["left"]["dir"])) == 4:
        rule_cat = False

    for s,d in zip( cat_recoded["left"]["spf"], cat_recoded["left"]["dir"] ):
        if leftup_cat:
            linear_value_mat[int(s),int(d)] = s+d
        else:
            linear_value_mat[int(s),int(d)] = (mat_size[0]-1-s)+d
    for s,d in zip( cat_recoded["right"]["spf"], cat_recoded["right"]["dir"] ):
        if leftup_cat:
            linear_value_mat[int(s),int(d)] = s+d
        else:
            linear_value_mat[int(s),int(d)] = (mat_size[0]-1-s)+d

    if left_larger_right_cat:
        linear_value_mat = np.abs( 9-linear_value_mat )

    if not rule_cat:
        middle_row = int((linear_value_mat.shape[0]-1)/2)
        linear_value_mat[middle_row,:] = linear_value_mat[middle_row,-1::-1]

    return linear_value_mat


def convert_bound_dist_to_matrix( cat_recoded, bound_dist, mat_size=(5,6) ):
    dist_mat = np.full(mat_size,np.NaN)
    for s,d,b in zip( cat_recoded["left"]["spf"], cat_recoded["left"]["dir"], bound_dist["left"] ):
        dist_mat[int(s),int(d)] = b
    for s,d,b in zip( cat_recoded["right"]["spf"], cat_recoded["right"]["dir"], bound_dist["right"] ):
        dist_mat[int(s),int(d)] = b
    return dist_mat


def calculate_boundary( performance_matrix, normalize=False, mat_size=(6,7), return_grid=False ):
    """ Calulates the boundary equation and angle using the 6x7 performance matrix as input, matrix should include np.NaN on not-used entries """

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    if normalize:
        performance_matrix = (performance_matrix - np.nanmin(performance_matrix)) / (np.nanmax(performance_matrix)-np.nanmin(performance_matrix))

    # Prepare matrix with x,y,z values
    n_datapts = mat_size[0]*mat_size[1]
    X,Y = np.meshgrid(np.arange(mat_size[1]),np.arange(mat_size[0]))
    X = np.reshape(X,(n_datapts,))
    Y = np.reshape(Y,(n_datapts,))
    P = performance_matrix.ravel()
    notNanIx = ~np.isnan(P)
    data = np.stack( (X[notNanIx], Y[notNanIx], P[notNanIx]) ).T
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]

    # Get coefficients of fitted plane
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

    # Calculate line that divides fitted plane at Z=0.5
    x_line = np.arange(-1,mat_size[1]+1)
    y_line = (((-1*C[0]*x_line) -C[2])+0.5)/C[1]

    # Calculate angle of line
    boundary_angle = np.rad2deg(np.arctan2(y_line[-1] - y_line[0], x_line[-1] - x_line[0]))

    if return_grid:
        # Calculate plane-grid with applied boundary
        X,Y = np.meshgrid(np.arange(mat_size[1]),np.arange(mat_size[0]))
        grid = np.zeros(mat_size)
        grid[ C[0]*X + C[1]*Y + C[2] > 0.5 ] = 1.0
        return boundary_angle, (x_line,y_line), grid
    else:
        return boundary_angle, (x_line,y_line)


def bounded_sigmoid(p,x):
    # x0: time point of maximum steepness
    # k: steepness
    x0,k=p
    y = 1 / (1.0 + np.exp(-k*(x-x0)))
    return y


def get_bounded_sigmoid_fit( x, y ):
    # returns parameters (x0,k), see bounded sigmoid for explanation
    def residuals(p,x,y):
        return y - bounded_sigmoid(p,x)
    p_guess=(np.median(x),1.0)
    p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
                    residuals,p_guess,args=(x,y),full_output=1)
    return p


def get_chronic_infointegr_data(basepath="../.."):
    """ Get filenames and open files
        returns datafile reference
    """
    mouse_names = [ "F02","F03","F04", "21a","21b", "K01","K02","K03","K06","K07" ]
    datafiles = []
    for mouse in mouse_names:
        datafiles.append(h5py.File(basepath+"/data/p3a_chronicimagingbehavior/cat_behav_chronic_{}.mat".format(mouse),'r'))
    return datafiles


def quick_load_chronic_infointegr_data(basepath="../.."):
    load_filename = basepath+"/data/p3a_chronicimagingbehavior/data_chronic_infointegr.npy"
    save_dict = np.load( load_filename, allow_pickle=True ).item()
    data_dict = save_dict["data_dict"]
    mouse_names = save_dict["mouse_names"]
    print("Loaded data from: {}".format(load_filename))
    return mouse_names,data_dict


def get_muscimol_infointegr_data(basepath="../.."):
    """ Get filename and open file
        returns datafile reference
    """
    data_file_name = basepath+"/data/p3b_corticalinactivation/cat_behav_muscimol_chronic.mat"
    datafile = h5py.File(data_file_name,'r')
    return datafile

def quick_load_muscimol_infointegr_data(basepath="../.."):
    load_filename = basepath+"/data/p3b_corticalinactivation/data_muscimol_infointegr.npy"
    save_dict = np.load( load_filename, allow_pickle=True ).item()
    data_dict_acsf = save_dict["data_dict_acsf"]
    data_dict_musc = save_dict["data_dict_musc"]
    mouse_names = save_dict["mouse_names"]
    print("Loaded data from: {}".format(load_filename))
    return mouse_names,data_dict_acsf,data_dict_musc


def get_full_infointegr_data(basepath="../.."):
    """ Get filename and open file
        -- to get overview of main contents, use these lines --
        for k in datafile.keys():
            print(k)
        returns datafile reference
    """
    data_file_name = basepath+"/data/p2a_headfixbehavior/cat_behav_full_infointegr.mat"
    datafile = h5py.File(data_file_name,'r')
    return datafile

def quick_load_full_infointegr_data(basepath="../.."):
    load_filename = basepath+"/data/p2a_headfixbehavior/data_full_infointegr.npy"
    save_dict = np.load( load_filename, allow_pickle=True ).item()
    data_dict = save_dict["data_dict"]
    mouse_names = save_dict["mouse_names"]
    print("Loaded data from: {}".format(load_filename))
    return mouse_names,data_dict


def get_second_infointegr_data(basepath="../.."):
    """ Get filename and open file
        -- to get overview of main contents, use these lines --
        for k in datafile.keys():
            print(k)
        returns datafile reference
    """
    mouse_names = ["W01","W02","W03","W04","W05", "W06","W07","W08","W09","W10","W11","W12"]
    datafiles = []
    for mouse in mouse_names:
        datafiles.append(h5py.File(basepath+"/data/p2a_headfixbehavior/cat_behav_second_{}.mat".format(mouse),'r'))
    return datafiles

def quick_load_second_infointegr_data(basepath="../.."):
    load_filename = basepath+"/data/p2a_headfixbehavior/data_second_infointegr.npy"
    save_dict = np.load( load_filename, allow_pickle=True ).item()
    data_dict = save_dict["data_dict"]
    mouse_names = save_dict["mouse_names"]
    print("Loaded data from: {}".format(load_filename))
    return mouse_names,data_dict


def get_retinotopy_infointegr_data(basepath="../.."):
    """ Get filename and open file
        -- to get overview of main contents, use these lines --
        for k in datafile.keys():
            print(k)
        returns datafile reference
    """
    data_file_name = basepath+"/data/p2b_retinotopybehavior/cat_behav_shifted_infointegr.mat"
    datafile = h5py.File(data_file_name,'r')
    return datafile

def quick_load_retinotopy_infointegr_data(basepath="../.."):
    load_filename = basepath+"/data/p2b_retinotopybehavior/data_retinotopy_infointegr.npy"
    save_dict = np.load( load_filename, allow_pickle=True ).item()
    data_dict = save_dict["data_dict"]
    mouse_names = save_dict["mouse_names"]
    print("Loaded data from: {}".format(load_filename))
    return mouse_names,data_dict


def extract_chronic_infointegr_data( datafiles, basepath="../.." ):
    """ Extracts data from multiple hdf5 files (one per mouse)
        returns mouse names and a data dictionary
    """

    mouse_names = []
    data_dict = {}
    for datafile in datafiles:

        # Mouse name
        mouse_ascii = datafile["MouseName"][:,0]
        mouse = ''.join(chr(c) for c in mouse_ascii)
        mouse_names.append( mouse )
        print("Loading data of mouse {}".format(mouse))

        # Get session type
        date_list = []
        name_list = []
        type_list = []
        typenr_list = []
        session_iterable = []
        for dt,nm,tp,tp_nr,s_ref in zip( datafile["BehavData/date"][:,0], datafile["BehavData/name"][:,0], datafile["BehavData/type"][:,0], datafile["BehavData/typenr"][:,0], datafile["BehavData/data"][:,0] ):
            date_list.append(dt)
            typenr_list.append(tp_nr)
            name_list.append(''.join(chr(c) for c in datafile[nm]))
            type_list.append(''.join(chr(c) for c in datafile[tp]))
            session_iterable.append( datafile[s_ref] )

        # Get data dict
        data_dict[mouse] = extract_session_data( session_iterable, mouse, n_angle_sets=1, name_list=name_list, date_list=date_list, type_list=type_list, typenr_list=typenr_list, reduced_cats=True )

    save_dict = { "data_dict": data_dict, "mouse_names": mouse_names }
    save_filename = basepath+"/data/p3a_chronicimagingbehavior/data_chronic_infointegr.npy"
    np.save( save_filename, save_dict )
    print("Saved data in: {}".format(save_filename))

    return mouse_names,data_dict


def extract_muscimol_infointegr_data( datafile, basepath="../.." ):
    """ Extracts data from hdf5 file
        returns mouse names and a data dictionary
    """
    mdata_acsf = datafile["acsfData"][:,0]
    mdata_musc = datafile["muscData"][:,0]
    mice = datafile["mouseNames"][:,0]
    mouse_names = []
    data_dict_acsf = {}
    data_dict_musc = {}
    for m,da,dm in zip(mice,mdata_acsf,mdata_musc):

        # Mouse name
        mouse_ascii = datafile[m][:,0]
        mouse = ''.join(chr(c) for c in mouse_ascii)
        mouse_names.append( mouse )
        print("Loading data of mouse {}".format(mouse))

        # Extract session data for muscimol experiment
        session_iterable = []
        for s_ref in datafile[da][:,0]:
            session_iterable.append( datafile[s_ref] )
        typenr_list = np.full((len(session_iterable)),1.0)
        data_dict_acsf[mouse] = extract_session_data( session_iterable, mouse, n_angle_sets=1, typenr_list=typenr_list, reduced_cats=True, aux_samp_freq=5000.0 )

        # Extract session data for muscimol experiment
        session_iterable = []
        for s_ref in datafile[dm][:,0]:
            session_iterable.append( datafile[s_ref] )
        data_dict_musc[mouse] = extract_session_data( session_iterable, mouse, n_angle_sets=1, typenr_list=typenr_list, reduced_cats=True, aux_samp_freq=5000.0 )

    save_dict = { "data_dict_acsf": data_dict_acsf, "data_dict_musc": data_dict_musc, "mouse_names": mouse_names }
    save_filename = basepath+"/data/p3b_corticalinactivation/data_muscimol_infointegr.npy"
    np.save( save_filename, save_dict )
    print("Saved data in: {}".format(save_filename))

    return mouse_names,data_dict_acsf,data_dict_musc


def extract_full_infointegr_data( datafile, basepath="../.." ):
    """ Extracts data from hdf5 file

        -- Get overview of contents in python using thise lines --
        session_ref = datafile[d]["data"][0,0]
        print(datafile[session_ref])
        for k in datafile[session_ref].keys():
            print(k)

        returns mouse names and a data dictionary
    """
    mdata = datafile["BehavData"][:,0]
    mice = datafile["Mice"][:,0]
    mouse_names = []
    data_dict = {}
    for m,d in zip(mice,mdata):

        # Mouse name
        mouse_ascii = datafile[m][:,0]
        mouse = ''.join(chr(c) for c in mouse_ascii)
        mouse_names.append( mouse )
        print("Loading data of mouse {}".format(mouse))

        # Extract session data
        session_iterable = []
        for s_ref in datafile[d]["data"][:,0]:
            session_iterable.append( datafile[s_ref] )
        data_dict[mouse] = extract_session_data( session_iterable, mouse, n_angle_sets=2 )

    save_dict = { "data_dict": data_dict, "mouse_names": mouse_names }
    save_filename = basepath+"/data/p2a_headfixbehavior/data_full_infointegr.npy"
    np.save( save_filename, save_dict )
    print("Saved data in: {}".format(save_filename))

    return mouse_names,data_dict


def extract_second_infointegr_data( datafiles, basepath="../.." ):
    """ Extracts data from multiple hdf5 files (one per mouse)
        for k in datafile["BehavData"].keys():
            print(k)
        print(datafile["BehavData/data"][0,0]) -> objref
        print(datafile["BehavData/date"][0,0]) -> int
        print(datafile["BehavData/name"][0,0]) -> objref
        print(datafile["BehavData/type"][0,0]) -> objref
        print(datafile["BehavData/typenr"][0,0]) -> int
        returns mouse names and a data dictionary
    """

    mouse_names = []
    data_dict = {}
    for datafile in datafiles:

        # Mouse name
        mouse_ascii = datafile["MouseName"][:,0]
        mouse = ''.join(chr(c) for c in mouse_ascii)
        mouse_names.append( mouse )
        print("Loading data of mouse {}".format(mouse))

        # Get session type
        date_list = []
        name_list = []
        type_list = []
        typenr_list = []
        session_iterable = []
        for dt,nm,tp,tp_nr,s_ref in zip( datafile["BehavData/date"][:,0], datafile["BehavData/name"][:,0], datafile["BehavData/type"][:,0], datafile["BehavData/typenr"][:,0], datafile["BehavData/data"][:,0] ):
            if tp_nr > 2 and dt not in date_list:
                date_list.append(dt)
                typenr_list.append(tp_nr)
                name_list.append(''.join(chr(c) for c in datafile[nm]))
                type_list.append(''.join(chr(c) for c in datafile[tp]))
                session_iterable.append( datafile[s_ref] )

        # Get data dict
        data_dict[mouse] = extract_session_data( session_iterable, mouse, n_angle_sets=1, name_list=name_list, date_list=date_list, type_list=type_list, typenr_list=typenr_list )

    save_dict = { "data_dict": data_dict, "mouse_names": mouse_names }
    save_filename = basepath+"/data/p2a_headfixbehavior/data_second_infointegr.npy"
    np.save( save_filename, save_dict )
    print("Saved data in: {}".format(save_filename))

    return mouse_names,data_dict


def extract_retinotopy_infointegr_data( datafile, basepath="../.." ):
    """ Extracts data from hdf5 file
        returns mouse names and a data dictionary
    """
    mdata = datafile["BehavData"][:,0]
    mice = datafile["Mice"][:,0]
    mouse_names = []
    data_dict = {}
    for m,d in zip(mice,mdata):

        # Mouse name
        mouse_ascii = datafile[m][:,0]
        mouse = ''.join(chr(c) for c in mouse_ascii)
        mouse_names.append( mouse )
        print("Loading data of mouse {}".format(mouse))

        # Get session type
        date_list = []
        name_list = []
        shift_list = []
        eye1_list = []
        eye2_list = []
        session_iterable = []
        for dt,nm,sh,eye1,eye2,s_ref in zip( datafile[d]["date"][:,0], datafile[d]["name"][:,0], datafile[d]["shift"][:,0], datafile[d]["eye1"][:,0], datafile[d]["eye2"][:,0], datafile[d]["data"][:,0] ):

            date_list.append(dt)
            name_list.append(''.join(chr(c) for c in datafile[nm]))
            shift_list.append(sh)
            eye1_list.append(eye1)
            eye2_list.append(eye2)
            session_iterable.append( datafile[s_ref] )

        # Extract session data
        data_dict[mouse] = extract_session_data( session_iterable, mouse, n_angle_sets=1, name_list=name_list, date_list=date_list, shift_list=shift_list )

    save_dict = { "data_dict": data_dict, "mouse_names": mouse_names }
    save_filename = basepath+"/data/p2b_retinotopybehavior/data_retinotopy_infointegr.npy"
    np.save( save_filename, save_dict )
    print("Saved data in: {}".format(save_filename))

    return mouse_names,data_dict


def extract_session_data( session_iterable, mouse, n_angle_sets, name_list=None, date_list=None, type_list=None, typenr_list=None, shift_list=None, eye1_list=None, eye2_list=None, reduced_cats=False, aux_samp_freq=None ):

    # Prepare lists with all angles and spatialfs and sessions
    all_angles = [[],[]]
    all_spatialfs = [[],[]]
    mouse_sessions = []
    data_dict = {}
    for s_nr,ses_data in enumerate(session_iterable):

        # Get task name, skip is still phase 4
        if "TaskName" not in ses_data and type_list is None:
            print(" ** No taskName .. session with missing stim data, skipping")
            continue
        elif "TaskName" not in ses_data:
            taskname = type_list[s_nr]
        taskname_ascii = ses_data["TaskName"][:,0]
        taskname = ''.join(chr(c) for c in taskname_ascii)
        if "4" in taskname:
            print(" ** Training phase 4 data, skipping")
            continue
        mouse_sessions.append({})
        mouse_sessions[-1]["taskname"] = taskname
        mouse_sessions[-1]["exclude"] = False
        # for k in datafile[s_ref].keys():
        #     print(k)

        # Get date
        if "date" in ses_data:
            date = ses_data["date"][0,0]
            mouse_sessions[-1]["date"] = date
        elif date_list is not None:
            mouse_sessions[-1]["date"] = date_list[s_nr]
        elif "MouseDateId" in ses_data:
            date_ascii = ses_data["MouseDateId"][4:10,0]
            date_str = ''.join(chr(c) for c in date_ascii)
            mouse_sessions[-1]["date"] = int("20"+date_str)
        else:
            mouse_sessions[-1]["date"] = "no date listed"

        # Get original file name
        if "filename" in ses_data:
            origfilename_ascii = ses_data["filename"][:,0]
            origfilename = ''.join(chr(c) for c in origfilename_ascii)
            mouse_sessions[-1]["filename"] = origfilename
        elif name_list is not None:
            mouse_sessions[-1]["filename"] = name_list[s_nr]
        elif "MouseDateId" in ses_data:
            origfilename_ascii = ses_data["MouseDateId"][:,0]
            origfilename = ''.join(chr(c) for c in origfilename_ascii)
            mouse_sessions[-1]["filename"] = origfilename
        else:
            mouse_sessions[-1]["filename"] = "no filename listed"

        # print( "{:03.0f}) {:08.0f}, {}".format( len(mouse_sessions), mouse_sessions[-1]["date"], mouse_sessions[-1]["filename"] ) )

        # Outcome, CategoryId, StimulusId
        mouse_sessions[-1]["outcome"] = np.array(ses_data["Outcome"][:,0])
        mouse_sessions[-1]["categoryid"] = np.array(ses_data["CategoryId"][:,0])
        mouse_sessions[-1]["stimulusid"] = np.array(ses_data["StimulusId"][:,0])

        # Aux sampling freq
        if reduced_cats:
            if aux_samp_freq is None:
                mouse_sessions[-1]["aux-sf"] = ses_data["AuxSampFreq"][0,0]
            else:
                mouse_sessions[-1]["aux-sf"] = aux_samp_freq
        else:
            mouse_sessions[-1]["aux-sf"] = 500

        # # Reaction times
        # rt = np.array(ses_data["TrRtRespLickIx"][:,0])
        # rt[rt<0] = np.NaN
        # mouse_sessions[-1]["TrRtRespLickIx"] = rt
        # rt = np.array(ses_data["TrRespRtFirstLickIx"][:,0])
        # rt[rt<0] = np.NaN
        # mouse_sessions[-1]["TrRespRtFirstLickIx"] = rt

        # Check if unfortunate mismatch auxdata, if so, exclude session
        if np.array(ses_data["RespLickOutcomeIx"][:,0]).shape[0] != mouse_sessions[-1]["categoryid"].shape[0]:
            print(" ** Session {} Mismatching #trials, categoryid={}, aux-outcome={}".format( mouse_sessions[-1]["date"], mouse_sessions[-1]["categoryid"].shape[0], np.array(ses_data["RespLickOutcomeIx"][:,0]).shape[0] ) )
            mouse_sessions[-1]["exclude"] = True
            continue

        # Raw trigger indices
        mouse_sessions[-1]["WaitForNoLickIx"] = np.array(ses_data["WaitForNoLickIx"][:,0])
        mouse_sessions[-1]["StimOnsetIx"] = np.array(ses_data["StimOnsetIx"][:,0])
        mouse_sessions[-1]["StimOffsetIx"] = np.array(ses_data["StimOffsetIx"][:,0])
        mouse_sessions[-1]["RespWinStartIx"] = np.array(ses_data["RespWinStartIx"][:,0])
        mouse_sessions[-1]["RespWinStopIx"] = np.array(ses_data["RespWinStopIx"][:,0])

        # Raw data indices
        mouse_sessions[-1]["DropIx"] = np.array(ses_data["DropIx"][:,0])
        mouse_sessions[-1]["LickIx"] = np.array(ses_data["LickIx"][:,0])
        mouse_sessions[-1]["LickDirIx"] = np.array(ses_data["LickDirIx"][:,0])

        # Stimulus specs ordered by id
        mouse_sessions[-1]["leftcat-angles"] = np.array(ses_data["S/LeftCat/Angles"][:,0])
        mouse_sessions[-1]["rightcat-angles"] = np.array(ses_data["S/RightCat/Angles"][:,0])

        mouse_sessions[-1]["leftcat-spatialf"] = np.array(ses_data["S/LeftCat/spatialF"][:,0])
        mouse_sessions[-1]["rightcat-spatialf"] = np.array(ses_data["S/RightCat/spatialF"][:,0])

        if "CatLevel" in ses_data["S/LeftCat"]:
            mouse_sessions[-1]["leftcat-level"] = np.array(ses_data["S/LeftCat/CatLevel"][:,0])
            mouse_sessions[-1]["rightcat-level"] = np.array(ses_data["S/RightCat/CatLevel"][:,0])
        else:
            if mouse_sessions[-1]["leftcat-angles"].shape[0] == 1:
                mouse_sessions[-1]["leftcat-level"] = np.array([1.0,])
                mouse_sessions[-1]["rightcat-level"] = np.array([1.0,])
            elif mouse_sessions[-1]["leftcat-angles"].shape[0] == 3:
                mouse_sessions[-1]["leftcat-level"] = np.array([1.0,2.0,2.0])
                mouse_sessions[-1]["rightcat-level"] = np.array([1.0,2.0,2.0])

        # Get typenr from list if supplied
        if typenr_list is not None:
            mouse_sessions[-1]["typenr"] = typenr_list[s_nr]

        # Get CT level
        if reduced_cats:
            if mouse_sessions[-1]["typenr"] == 1:
                mouse_sessions[-1]["ct"] = int(0)
            elif mouse_sessions[-1]["typenr"] == 2:
                mouse_sessions[-1]["ct"] = int(1)
            elif mouse_sessions[-1]["typenr"] == 3:
                mouse_sessions[-1]["ct"] = int(1)
            elif mouse_sessions[-1]["typenr"] == 4:
                mouse_sessions[-1]["ct"] = int(2)
            elif mouse_sessions[-1]["typenr"] == 5:
                mouse_sessions[-1]["ct"] = int(2)
            elif mouse_sessions[-1]["typenr"] == 6:
                mouse_sessions[-1]["ct"] = int(3)
            elif mouse_sessions[-1]["typenr"] == 7:
                mouse_sessions[-1]["ct"] = int(3)
        else:
            if mouse_sessions[-1]["leftcat-angles"].shape[0] == 1:
                mouse_sessions[-1]["ct"] = 0
            elif mouse_sessions[-1]["leftcat-angles"].shape[0] == 3:
                mouse_sessions[-1]["ct"] = 1
            elif mouse_sessions[-1]["leftcat-angles"].shape[0] == 6:
                mouse_sessions[-1]["ct"] = 2
            elif mouse_sessions[-1]["leftcat-angles"].shape[0] == 10:
                mouse_sessions[-1]["ct"] = 3
            elif mouse_sessions[-1]["leftcat-angles"].shape[0] == 15:
                mouse_sessions[-1]["ct"] = 4
            elif mouse_sessions[-1]["leftcat-angles"].shape[0] == 21:
                mouse_sessions[-1]["ct"] = 5

        # Get type nr from ses if not supplied in list
        if typenr_list is  None:
            if mouse_sessions[-1]["ct"] == 0:
                mouse_sessions[-1]["typenr"] = 3
            else:
                mouse_sessions[-1]["typenr"] = 4

        # Correct specific baseline imaging sessions
        if reduced_cats and mouse_sessions[-1]["typenr"] == 1.0:
            mouse_sessions[-1]["outcome"] = np.array(ses_data["RespLickOutcomeIx"][:,0])

        # print( "{:03.0f}) {:08.0f}, {} (type={}, ct={}, sf={})".format( len(mouse_sessions), mouse_sessions[-1]["date"], mouse_sessions[-1]["filename"], mouse_sessions[-1]["typenr"], mouse_sessions[-1]["ct"], mouse_sessions[-1]["aux-sf"] ) )

        # Get shift nr
        if shift_list is not None:
            mouse_sessions[-1]["shift"] = shift_list[s_nr]
        else:
            mouse_sessions[-1]["shift"] = 0

        # print( "{:03.0f}) {:08.0f}, ct={}, shift={}".format( len(mouse_sessions), date, mouse_sessions[-1]["ct"], mouse_sessions[-1]["shift"] ) )

        # Get angle set by calculating spacing
        all_session_angles = np.sort( np.unique( np.concatenate( (mouse_sessions[-1]["leftcat-angles"], mouse_sessions[-1]["rightcat-angles"]) ) ) )
        all_session_spatialfs = np.sort( np.unique( np.concatenate( (mouse_sessions[-1]["leftcat-spatialf"], mouse_sessions[-1]["rightcat-spatialf"]) ) ) )
        angle_spacing = all_session_angles[1]-all_session_angles[0]
        angle_set = 0
        if angle_spacing == 20 and n_angle_sets > 1:
            angle_set = 1
        all_angles[angle_set].append(all_session_angles)
        all_spatialfs[angle_set].append(all_session_spatialfs)
        mouse_sessions[-1]["angleset"] = angle_set

    # Add number of sessions
    data_dict["n_sessions"] = len(mouse_sessions)

    # Get list of used angles and spatialf's
    for aset in range(2):
        if len(all_angles[aset]) == 0:
            continue
        all_angles[aset] = np.unique(np.concatenate(all_angles[aset]))
        all_spatialfs[aset] = np.unique(np.concatenate(all_spatialfs[aset]))
    data_dict["angles"] = all_angles
    data_dict["spatialfs"] = all_spatialfs

    # Add stimulus specs and ix per trial
    for s_nr,s in enumerate(mouse_sessions):
        if s["exclude"]:
            continue
        mouse_sessions[s_nr]["angle"] = np.zeros(s["outcome"].shape)
        mouse_sessions[s_nr]["spatialf"] = np.zeros(s["outcome"].shape)
        mouse_sessions[s_nr]["level"] = np.zeros(s["outcome"].shape)
        mouse_sessions[s_nr]["angleix"] = np.zeros(s["outcome"].shape)
        mouse_sessions[s_nr]["spatialfix"] = np.zeros(s["outcome"].shape)
        ses_angleset = s["angleset"]
        ses_angles = data_dict["angles"][ses_angleset]
        ses_spatialfs = data_dict["spatialfs"][ses_angleset]
        for t,(c,st) in enumerate(zip(s["categoryid"],s["stimulusid"])):
            if c == 1:
                angle = s["leftcat-angles"][int(st)-1]
                spatialf = s["leftcat-spatialf"][int(st)-1]
                level = s["leftcat-level"][int(st)-1] - 6
            elif c == 2:
                angle = s["rightcat-angles"][int(st)-1]
                spatialf = s["rightcat-spatialf"][int(st)-1]
                level = np.abs( s["rightcat-level"][int(st)-1] - 6 )+1
            mouse_sessions[s_nr]["angle"][t] = angle
            mouse_sessions[s_nr]["spatialf"][t] = spatialf
            mouse_sessions[s_nr]["level"][t] = level
            angleix = int(np.argwhere(ses_angles==angle))
            spatialfix = int(np.argwhere(ses_spatialfs==spatialf))
            mouse_sessions[s_nr]["angleix"][t] = angleix
            mouse_sessions[s_nr]["spatialfix"][t] = spatialfix

    # Get number of cts
    if reduced_cats:
        n_cts = 4
    else:
        n_cts = 6

    # Add sessions to corresponding ct group
    data_dict["sessions"] = [[] for _ in range(n_cts)]
    date_list = [[] for _ in range(n_cts)]
    for s_nr,s in enumerate(mouse_sessions):
        if s["exclude"]:
            continue
        ct = s["ct"]
        data_dict["sessions"][ct].append(s)
        date_list[ct].append(s["date"])

    # Sort ct-groups by date
    for ct in range(n_cts):
        data_dict["sessions"][ct] = [x for _,x in sorted( zip(date_list[int(ct)],data_dict["sessions"][int(ct)]) )]

    # Display included sessions
    for ctdata in data_dict["sessions"]:
        for s_nr,s in enumerate(ctdata):
            print( "{:3.0f}) {:08.0f}, {} (type={}, ct={})".format( s_nr, s["date"], s["filename"], s["typenr"], s["ct"] ) )

    return data_dict
