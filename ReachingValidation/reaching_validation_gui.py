from cmath import nan
from tkinter import N
from ExtractKinematicData import ExtractKinematicData
import matplotlib.pyplot as plt
from SegmentMovement import segment
from DataLoader import DataLoader
import variableDeclaration as var
import os
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
os.system('cls' if os.name == 'nt' else 'clear') # clears terminal
plt.rcParams.update({'font.size': 15})

'''
    Copyright 2022 Jan Breuer & Shivakeshavan Ratnadurai-Giridharan

    Validation script for the results of the PEACK system to the 
    goldstandard values of VICON. 
'''

def plot_timeseries(vicon_org,vicon,peack_org,peack, t1, t2, ax):
    '''
        Defining a function to plot both the PEACK and VICON data 
    '''
    ax.plot(t1,vicon_org,'--', color='tab:cyan')
    ax.plot(t1,vicon, color='tab:blue')
    # ax.plot(t2,peack_org,'--', color='tab:pink')
    ax.plot(t2,peack, color='tab:red')
    ax.set(xlabel='Time (s)', ylabel='Velocity')
    ax.grid()
    plt.tight_layout()
    # fig1.savefig("Figures//Limbs_angles_LElbow.png", dpi=600, bbox_inches='tight')

def receive_data(participant, all_joints, smoothing_alpha, cutoff, order, median_filter, cutoff_vel, order_vel, median_filter_vel, trunc, enable_late_trunc = True):
    assert(DataLoader(var.ViconFiles).NsubDirs == DataLoader(var.PeackFiles).NsubDirs) # make sure that the (sub)directories are the same

    # loop through the number of participants you want to compare
    for i in participant:#ViconFiles.NsubDirs):
        print("Participant ",i)
        assert(DataLoader(var.ViconFiles).SubDirs[i].Nfiles==DataLoader(var.PeackFiles).SubDirs[i].Nfiles)
        
        # loop through each trial of the i-th participant
        Vicon_original_positional=[]; Peack_original_positional=[]; VICON_speed_original = []; VICON_speed_filtered = []; PEACK_speed_original = []; PEACK_speed_filtered = []; PEACK_time = []; VICON_time = []; VICON_speed_raw = []; PEACK_speed_raw = []; VICON_speed_late_trunc = []; PEACK_speed_late_trunc = []; VICON_stamps = []; PEACK_stamps = []
        for j in range(0,DataLoader(var.ViconFiles).SubDirs[i].Nfiles):

            Vicon_original = ExtractKinematicData(DataLoader(var.ViconFiles).getFile(i,j), all_joints, var.fs_vicon, len(all_joints[0]), smoothing_alpha, cutoff, order, median_filter, trunc, unit_rescale=1000.0, type='VICON', filtered = False)
            Peack_original = ExtractKinematicData(DataLoader(var.PeackFiles).getFile(i,j), all_joints, var.fs_peack, len(all_joints[0]), smoothing_alpha, cutoff, order, median_filter, trunc, unit_rescale=1.0, type='PEACK', filtered = False)

            Vicon = ExtractKinematicData(DataLoader(var.ViconFiles).getFile(i,j), all_joints, var.fs_vicon, len(all_joints[0]), smoothing_alpha, cutoff, order, median_filter, trunc, unit_rescale=1000.0, type='VICON', filtered = True)
            Peack = ExtractKinematicData(DataLoader(var.PeackFiles).getFile(i,j), all_joints, var.fs_peack, len(all_joints[0]), smoothing_alpha, cutoff, order, median_filter, trunc, unit_rescale=1.0, type='PEACK', filtered = True)
            Vicon_original_positional.append(Vicon.data)
            Peack_original_positional.append(Peack.data)

            if enable_late_trunc:
                Vicon_late_trunc = ExtractKinematicData(DataLoader(var.ViconFiles).getFile(i,j), all_joints, var.fs_vicon, len(all_joints[0]), smoothing_alpha, cutoff, order, median_filter, trunc, unit_rescale=1000.0, type='VICON', filtered = True, prior_truncating = False)
                Peack_late_trunc = ExtractKinematicData(DataLoader(var.PeackFiles).getFile(i,j), all_joints, var.fs_peack, len(all_joints[0]), smoothing_alpha, cutoff, order, median_filter, trunc, unit_rescale=1.0, type='PEACK', filtered = True, prior_truncating = False)
                VICON_speed_late_trunc.append(segment(Vicon_late_trunc).get_speed_filtered(cutoff_vel, order_vel, median_filter_vel))
                PEACK_speed_late_trunc.append(segment(Peack_late_trunc).get_speed_filtered(cutoff_vel, order_vel, median_filter_vel))  
            
            VICON_speed_original.append(segment(Vicon_original).get_speed(cutoff_vel, order_vel, median_filter_vel))
            VICON_speed_raw.append(segment(Vicon_original).get_speed(cutoff_vel, order_vel, median_filter_vel, False))
            VICON_speed_filtered.append(segment(Vicon).get_speed_filtered(cutoff_vel, order_vel, median_filter_vel))
            VICON_time.append(Vicon.time[1:])

            joint_stamps = []
            for joint_data in segment(Vicon).get_speed_filtered(cutoff_vel, order_vel, median_filter_vel):
                joint_stamps.append(segment.peaks_and_troughs(joint_data,Vicon.time[1:]))
            VICON_stamps.append(joint_stamps)

            
            PEACK_speed_original.append(segment(Peack_original).get_speed(cutoff_vel, order_vel, median_filter_vel))
            PEACK_speed_raw.append(segment(Peack_original).get_speed(cutoff_vel, order_vel, median_filter_vel, False))
            PEACK_speed_filtered.append(segment(Peack).get_speed_filtered(cutoff_vel, order_vel, median_filter_vel))
            PEACK_time.append(Peack.time[1:])

            joint_stamps = []
            for joint_data in segment(Peack).get_speed_filtered(cutoff_vel, order_vel, median_filter_vel):
                joint_stamps.append(segment.peaks_and_troughs(joint_data,Peack.time[1:]))
            PEACK_stamps.append(joint_stamps)


    data_files = {
        "VICON_positional": Vicon_original_positional,
        "PEACK_positional": Peack_original_positional,
        "VICON_speed_original": VICON_speed_original,
        "VICON_speed_filtered": VICON_speed_filtered,
        "VICON_speed_raw": VICON_speed_raw,
        "VICON_speed_late_trunc" : VICON_speed_late_trunc,
        "PEACK_speed_original": PEACK_speed_original,
        "PEACK_speed_filtered": PEACK_speed_filtered,
        "PEACK_speed_raw": PEACK_speed_raw,
        "PEACK_speed_late_trunc" : PEACK_speed_late_trunc,
        "VICON_time": VICON_time,
        "PEACK_time": PEACK_time,
        "VICON_stamps": VICON_stamps,
        "PEACK_stamps": PEACK_stamps
    }
    return data_files
    


def calc_correlation(VICON_speed_filtered, PEACK_speed_filtered, VICON_time, PEACK_time):

    # takes several list as inputs. Every index in the list is equal to one trial
    corr_values = [ [] for _ in range(VICON_speed_filtered[0].shape[0]) ]
    corr_lag_values = [ [] for _ in range(VICON_speed_filtered[0].shape[0]) ]
    rmse_values = [ [] for _ in range(VICON_speed_filtered[0].shape[0]) ]
    for i in range(0,len(VICON_speed_filtered)):
        for j in range(0,VICON_speed_filtered[0].shape[0]):
            temp = crosscorr(VICON_speed_filtered[i][j],PEACK_speed_filtered[i][j],VICON_time[i],PEACK_time[i],plot=False)
            corr_values[j].append(temp[0]); rmse_values[j].append(temp[1]); corr_lag_values[j].append(temp[2])
    return [corr_values, rmse_values, corr_lag_values]

def crosscorr(VICON,PEACK,VICON_time,PEACK_time,plot=False):
    pearson = 0
    for a in np.linspace(0, VICON_time[0]-PEACK_time[0], num=5):
        temp = np.interp(PEACK_time,VICON_time-a,VICON) # express the VICON data in the time domain of PEACK

        corr_scipy= scipy.signal.correlate(temp, PEACK, mode='full')
        corr_scipy_lag=scipy.signal.correlation_lags(temp.size, PEACK.size, mode='full')
        lag = corr_scipy_lag[np.argmax(corr_scipy)]
        lag_sec = PEACK_time[lag]-PEACK_time[1]

        try:
            if lag > 0: 
                tmpPearson = scipy.stats.pearsonr(temp[lag:], PEACK[:-lag])[0]
            else:
                tmpPearson = scipy.stats.pearsonr(temp, PEACK)[0]
        except: tmpPearson = 0

        if tmpPearson > pearson:
            pearson=tmpPearson

        try: rmse = mean_squared_error(temp[lag:], PEACK[:-lag], squared=False)
        except: rmse = nan
    
    return [round(pearson,3), rmse, lag_sec] # returns corr_score and lag in seconds


    