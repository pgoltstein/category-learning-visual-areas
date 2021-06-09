#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 2, 2017

Contains functions for analyzing klimbic / med-associates behavioral
chamber data

@author: pgoltstein
"""

import collections
import os
import csv
import numpy as np
import scipy.linalg
import glob, datetime

GRATING = {}
GRATING["spatialf"] = np.array(7*[[0.03,0.035,0.04,0.05,0.07,0.09,0.13]]).T.ravel()
GRATING["orientation"] = np.array([7*[d] for d in [0.0,15.0,30.0,45.0,60.0,75.0,90.0]]).T.ravel()

spatialf_ids = np.array(7*[[1.0,2.0,3.0,4.0,5.0,6.0,7.0]]).T
orientation_ids = np.array([7*[d] for d in [1.0,2.0,3.0,4.0,5.0,6.0,7.0]]).T
GRATING["spatialf_id"] = spatialf_ids.ravel()
GRATING["orientation_id"] = orientation_ids.ravel()

CATEGORY = {}
for m_nr,m in enumerate(["C01","C02","C03","C04","C05","C06","C07","C08"]):
    if np.mod(m_nr,2) == 0:
        CATEGORY[m] = {}
        CATEGORY[m]["target"] = ((spatialf_ids + np.flip(orientation_ids,axis=1)) > 8).ravel()
        CATEGORY[m]["dummy"] = ((spatialf_ids + np.flip(orientation_ids,axis=1)) < 8).ravel()
        CATEGORY[m]["level"] = np.abs((spatialf_ids + np.flip(orientation_ids,axis=1)) - 8).ravel()
    else:
        CATEGORY[m] = {}
        CATEGORY[m]["target"] = ((spatialf_ids + orientation_ids) < 8).ravel()
        CATEGORY[m]["dummy"] = ((spatialf_ids + orientation_ids) > 8).ravel()
        CATEGORY[m]["level"] = np.abs((spatialf_ids + orientation_ids) - 8).ravel()
    CATEGORY[m]["level"][CATEGORY[m]["level"]==7] = 0
    CATEGORY[m]["level"][CATEGORY[m]["dummy"]==1] *= -1

def calculate_boundary( performance_matrix ):
    """ Calulates the boundary equation and angle using the 7x7 performance matrix as input, matrix should include np.NaN on not-used entries """

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

    # Prepare matrix with x,y,z values
    X = GRATING["orientation_id"]
    Y = GRATING["spatialf_id"]
    P = performance_matrix.ravel()
    notNanIx = ~np.isnan(P)
    data = np.stack( (X[notNanIx], Y[notNanIx], P[notNanIx]) ).T
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]

    # Get coefficients of fitted plane
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

    # Calculate line that divides fitted plane at Z=0.5
    x_line = np.arange(1,8)
    y_line = (((-1*C[0]*x_line) -C[2])+0.5)/C[1]

    # Calculate angle of line
    boundary_angle = np.rad2deg(np.arctan2(y_line[-1] - y_line[0], x_line[-1] - x_line[0]))

    return boundary_angle, (x_line,y_line)


def mouse_summary( mouse_data ):
    print('\n-----------{}--------------'.format(mouse_data['name']))
    print('name : {}'.format( mouse_data["name"] ))
    print('date : {}'.format( mouse_data["date"] ))
    print('time : {}'.format( mouse_data["time"] ))
    print('box : {}'.format( mouse_data["box"] ))
    print('session type : {}'.format( mouse_data["session type"] ))
    print('protocol type : {}'.format( mouse_data["protocol type"] ))
    print('stimulus set : {}'.format( mouse_data["stimulus set"] ))
    # for k, v in mouse_data.items():
    #     print('{} : {}'.format(k, v))


def quick_load_behavioral_box_data_run1_run2(basepath="../.."):
    filepath=basepath+"/data/p1_behavioralchambers"
    load_filename = os.path.join( filepath, "data_behav_chamb.npy" )
    data_dict = np.load( load_filename, allow_pickle=True ).item()
    mdata = data_dict["mdata"]
    run_nr = data_dict["run_nr"]
    all_mice = data_dict["all_mice"]
    print("Loaded data from: {}".format(load_filename))

    return mdata,run_nr,all_mice

def load_behavioral_box_data_run1_run2(basepath="../.."):
    mice_run1 = ["C01","C03","C05","C07"]
    new_mice_run2 = ["C02","C04","C06","C08"]
    all_mice = ["C01","C02","C03","C04","C05","C06","C07","C08"]

    # Get .xls files
    filenames_run1 = glob.glob(basepath+"/data/p1_behavioralchambers/C01-C08-run1/*")
    filenames_run2 = glob.glob(basepath+"/data/p1_behavioralchambers/C01-C08-run2/*")

    # Sort filelist run 1 by date
    file_date_run1 = []
    for fn in filenames_run1:
        date_string = fn[-19:-8]
        date_tm = datetime.datetime.strptime(date_string,"%d-%b-%Y")
        file_date_run1.append((fn,date_tm))
    sorted_file_date_run1 = sorted(file_date_run1,key=lambda d: d[1])

    # Sort filelist run 2 by date
    file_date_run2 = []
    for fn in filenames_run2:
        date_string = fn[-19:-8]
        date_tm = datetime.datetime.strptime(date_string,"%d-%b-%Y")
        file_date_run2.append((fn,date_tm))
    sorted_file_date_run2 = sorted(file_date_run2,key=lambda d: d[1])

    # Load all data of run 1
    data_run1 = []
    for fn,dt in sorted_file_date_run1:
        print("Loading {}".format(fn))
        data_run1.append( read_klimbic( fn ) )

    # Load all data of run 2
    data_run2 = []
    for fn,dt in sorted_file_date_run2:
        print("Loading {}".format(fn))
        data_run2.append( read_klimbic( fn ) )

    # Get data of run 1 per mouse ID
    mdata = {}
    run_nr = {}
    for m in mice_run1:
        mdata[m] = [[],[],[],[],[],[]]
        run_nr[m] = [[],[],[],[],[],[]]
        for x_dt in data_run1:
            for x in x_dt:
                if x["name"] == m:
                    if "Prototypes" in x["stimulus set"]:
                        ct_level = 0
                    elif "CT>=5" in x["stimulus set"]:
                        ct_level = 1
                    elif "CT>=4" in x["stimulus set"]:
                        ct_level = 2
                    elif "CT>=3" in x["stimulus set"]:
                        ct_level = 3
                    elif "CT>=2" in x["stimulus set"]:
                        ct_level = 4
                    elif "CT>=1" in x["stimulus set"]:
                        ct_level = 5
                    if len(x["trial no"]) > 4:
                        mdata[m][ct_level].append(x)
                        run_nr[m][ct_level].append(1)

    # Get data of mice that were 'new' in run 2 per mouse ID
    for m in new_mice_run2:
        mdata[m] = [[],[],[],[],[],[]]
        run_nr[m] = [[],[],[],[],[],[]]
        for x_dt in data_run2:
            for x in x_dt:
                if x["name"] == m:
                    if "Prototypes" in x["stimulus set"]:
                        ct_level = 0
                    elif "CT>=5" in x["stimulus set"]:
                        ct_level = 1
                    elif "CT>=4" in x["stimulus set"]:
                        ct_level = 2
                    elif "CT>=3" in x["stimulus set"]:
                        ct_level = 3
                    elif "CT>=2" in x["stimulus set"]:
                        ct_level = 4
                    elif "CT>=1" in x["stimulus set"]:
                        ct_level = 5
                    if len(x["trial no"]) > 4:
                        mdata[m][ct_level].append(x)
                        run_nr[m][ct_level].append(2)

    # Add data of run1 mice in run 2 per mouse ID, but only include CT>=1 sessions
    for m in mice_run1:
        for x_dt in data_run2:
            for x in x_dt:
                if x["name"] == m and "CT>=1" in x["stimulus set"]:
                    if len(x["trial no"]) > 4:
                        mdata[m][5].append(x)
                        run_nr[m][ct_level].append(2)

    data_dict = { "mdata": mdata, "run_nr": run_nr, "all_mice": all_mice }
    save_filename = os.path.join( basepath+"/data/p1_behavioralchambers", 'data_behav_chamb' )
    np.save( save_filename, data_dict )
    print("Saved data in: {}".format(save_filename))

    return mdata,run_nr,all_mice


def read_klimbic( filename ):
    """Reads the .csv file from klimbic software
    """
    print("Reading: {}".format(filename))

    data = []
    temp_data = []
    no = -1
    read_stim_list = False
    read_trial_header = False
    read_trial_list = False

    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if len(row) == 0:
                continue

            if row[0] == "STARTDATA":
                data.append(collections.OrderedDict({'name': ''}))
                temp_data.append(collections.OrderedDict( { \
                    'stimulus groups': list(), \
                    'stimulus target index': list(), \
                    'stimulus target window': list(), \
                    'stimulus dummy index': list(), \
                    'stimulus dummy window': list() }))
                no += 1

            # Main info
            if row[0] == "Date":
                data[no]['date'] = row[1]
            if row[0] == "Time":
                data[no]['time'] = row[1]
            if row[0] == "Box Index":
                data[no]['box'] = row[1]
            if row[0] == "Session Title":
                data[no]['session type'] = ",".join(row[1:])
            if row[0] == "Protocol Title":
                data[no]['protocol type'] = ",".join(row[1:])
            if row[0] == "Protocol Description":
                data[no]['protocol type'] += "; " + ",".join(row[1:])
            if row[0] == "Subject Id":
                data[no]['name'] = row[1]
            if row[0] == "Duration":
                data[no]['duration'] = row[1]
            if row[0] == "Pellet Count":
                data[no]['pellets'] = row[1]

            # Stimulus definitions
            if read_stim_list == True:
                data[no]['stimulus names'] = row
                read_stim_list = False
            if row[0] == "AC Comment":
                data[no]['stimulus set'] = ",".join(row[2:])
                read_stim_list = True
            if row[0] == "Group":
                temp_data[no]['stimulus groups'].extend(row[1:])
            if row[0] == "TargetImageIndex":
                temp_data[no]['stimulus target index'].extend(row[1:])
            if row[0] == "TargetImageWindow":
                temp_data[no]['stimulus target window'].extend(row[1:])
            if row[0] == "DummyImageIndex":
                temp_data[no]['stimulus dummy index'].extend(row[1:])
            if row[0] == "DummyImageWindow":
                temp_data[no]['stimulus dummy window'].extend(row[1:])

            # Trial data
            if row[0] == "ENDDATA":
                read_trial_list = False
            if read_trial_list == True:
                data[no]['trial no'].append(int(row[0]))
                data[no]['trial outcome'].append(int(row[1]))
                data[no]['trial stimulus'].append(int(row[2]))
                if "No rep" in data[no]['protocol type']:
                    trial_lever_RT = (int(row[4])-int(row[3])) / 100
                    data[no]['trial lever RT'].append(trial_lever_RT)
                    trial_screen_RT = (int(row[9])-int(row[8])) / 100
                    data[no]['trial screen RT'].append(trial_screen_RT)
                else:
                    data[no]['trial lever RT'].append(0)
                    data[no]['trial screen RT'].append(0)
            if read_trial_header == True:
                if row[0] == 'Ref':
                    read_trial_list = True
                    read_trial_header = False
            if row[0] == "Trials":
                data[no]['trial no'] = []
                data[no]['trial outcome'] = []
                data[no]['trial stimulus'] = []
                data[no]['trial lever RT'] = []
                data[no]['trial screen RT'] = []
                read_trial_header = True

    # Group and order stimulus lists
    for no in range(len(data)):
        data[no]['stimulus names'].remove("")
        stim_groups = set(temp_data[no]['stimulus groups'])
        stim_groups.remove("")
        stim_groups = [ int(x) for x in stim_groups ]
        stim_groups = sorted(list(stim_groups))
        data[no]['stimulus groups'] = stim_groups
        data[no]['stimulus target index'] = []
        data[no]['stimulus dummy index'] = []
        data[no]['stimulus target window'] = []
        for grp_id in stim_groups:
            grp_list_index = temp_data[no]['stimulus groups'].index(str(grp_id))
            data[no]['stimulus target index'].append( int(temp_data[no]['stimulus target index'][grp_list_index]) )
            data[no]['stimulus dummy index'].append( int(temp_data[no]['stimulus dummy index'][grp_list_index]) )
            data[no]['stimulus target window'].append(  int(temp_data[no]['stimulus target window'][grp_list_index]) )

        # And get -useful- stimulus id's per trial (see GRATING constant above)
        data[no]['trial target name'] = []
        data[no]['trial target id'] = []
        data[no]['trial dummy name'] = []
        data[no]['trial dummy id'] = []
        for t in range(len(data[no]["trial stimulus"])):
            t_stim_group = data[no]["trial stimulus"][t]

            t_target_ix = data[no]["stimulus target index"][t_stim_group]
            t_target_name = data[no]["stimulus names"][t_target_ix]
            data[no]['trial target name'].append( t_target_name )

            t_dummy_ix = data[no]["stimulus dummy index"][t_stim_group]
            t_dummy_name = data[no]["stimulus names"][t_dummy_ix]
            data[no]['trial dummy name'].append( t_dummy_name )

            if "INT-CAT" in data[no]['stimulus set']:
                if t_target_name == "Rew-ct6":
                    t_target_id = 0
                else:
                    t_target_id = int(t_target_name.partition(" ")[0])-1
                data[no]['trial target id'].append( t_target_id )
                if t_dummy_name == "nRew-ct6":
                    t_dummy_id = 0
                else:
                    t_dummy_id = int(t_dummy_name.partition(" ")[0])-1
                data[no]['trial dummy id'].append( t_dummy_id )

    return data

# filename = basepath+"/data/p1_behavioralchambers/C01-C08-run2/01-Feb-2015_001.csv"
# # filename = basepath+"/data/p1_behavioralchambers/C01-C08-run1/30-Nov-2014_001.csv"
#
# data = read_klimbic( filename )
# for mouse_data in data:
#     mouse_summary(mouse_data)

#      ,       ,     ,Stage (3),Stage (3),Stage (4),Stage (4),Stage (4),Stage (8),Stage (9),Stage (13),
# Trial,       , Item,         , Count :-,         ,         ,Target   ,Reward   ,         ,  Count :-,
# Ref  ,Outcome,Index,    Entry,     S(1),    Entry,     Exit,       Id, Count :-,     Exit,      S(1),
# 0    ,      1,    1,        0,        0,      476,      766,     W(2),        0,        0,         0,
# 1    ,      0,    8,     2301,        0,     3724,     4158,     W(1),        1,     4530,         0,
# 2    ,      2,    6,     5975,        1,     7275,     8775,     W(1),        0,        0,         1,
# 3    ,      1,    0,    10276,        0,    10332,    11102,     W(1),        0,        0,         0,
