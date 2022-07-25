import numpy as np
import os
import pandas as pd
from PreProcess import PreProcess
from PEACK import get_UL_prop
import PEACK_Filters as PF
from scipy import stats
import variableDeclaration as var
import warnings
'''
    Copyright 2022 Jan Breuer & Shivakeshavan Ratnadurai-Giridharan

    A class for loading PEACK / VICON kinematic data.
    This code runs cubic spline interpolation 
'''

class ExtractKinematicData:

    def __init__(self, fn, all_joints, fs, N_ul_joints, smoothing_alpha=0.5, cutoff=1, order=3, median_filter=1, trunc= [200,0], unit_rescale=1.0, type='PEACK', filtered = False, prior_truncating = True):
        self.filename = fn
        self.all_joints = all_joints
        self.type = type
        self.fs = fs
        self.unit_rescale = unit_rescale
        self.smoothing_alpha = smoothing_alpha
        self.cutoff = cutoff
        self.order = order
        self.median_filter = median_filter
        self.N_ul_joints = N_ul_joints
        self.skiprows = 4 if self.type == 'VICON' else  0
        self.filtered = filtered
        self.trunc = trunc
        self.prior_truncating = prior_truncating
        self.getData()

    def getData(self):
        # creates a start and end time for a time vector - removes time to avoid edge artifacts
        temp = pd.read_csv(self.filename,skiprows=self.skiprows).values
        start = int((self.trunc[0]/1000)*self.fs)
        stop = int((self.trunc[1]/1000)*self.fs) if int((self.trunc[1]/1000)*self.fs) != 0 else 1

        self.time = temp[start:-stop, 0]/self.fs if self.type=='VICON' else temp[start:-stop, 0]
        if self.prior_truncating:
            data = temp[start:-stop, 2:]/self.unit_rescale if self.type=='VICON' else temp[start:-stop, 1:]/self.unit_rescale
        else:
            data = temp[:, 2:]/self.unit_rescale if self.type=='VICON' else temp[:, 1:]/self.unit_rescale
        
        data = PreProcess(data); del temp
        
        # Extract all samples in timeseries
        self.data = get_UL_prop(data, self.all_joints, 1, -1, self.N_ul_joints, self.type)  

        if(self.type=='PEACK'):
            dt = np.diff(self.time)
            sdt = (dt==0) + 0.
            t_ratio = np.sum(sdt) / len(dt)
            # only reconstructs it if t_ratio is bigger than 10%
            if var.debug_time:
                print('dt')
                print(dt)
                print('')
                print('self.time')
                print(self.time)
                print('')
                print('t_ratio')
                print(t_ratio)
                # print('')
                # print('true_dt')
                # print(true_dt)
                print('')
                print('start/stop')
                print(start)
                print(stop)
                breakpoint()
            true_dt = stats.mode(dt[dt!=0])[0][0]#np.mean(dt[dt!=0])
            if(t_ratio>0.1):    #If the timing information is wrong (>10% of times are not updated) then reconstruct time
                self.time = start/self.fs + np.arange(0,len(self.time)*true_dt,true_dt)
        
        if self.filtered:
            self.data_filtered = np.zeros_like(self.data)
            warnings.filterwarnings("ignore")
            #self.raw_data = np.copy(self.data)
            for j in range(0,self.N_ul_joints):
                joint = np.squeeze(self.data[j, :, :])
                mf_len = int(self.fs*1.5)
                joint = PF.position_median_filter(joint, self.fs/self.median_filter)
                joint = PF.exp_smoothing(joint,self.fs, self.smoothing_alpha)
                joint = PF.butter_lowpass_filter(joint, self.cutoff, self.fs, self.order)   # Filtering with a 5th order butterworth lowpass filter @ 6Hz.
                self.data_filtered[j, :, :] = joint  
        else:
            self.data_filtered=self.data

        if self.prior_truncating == False:
            # this needs to be altered!
            self.data_filtered = self.data_filtered[:,start:-stop,:]
        
    def reFilter(self,smoothing_alpha, cutoff, order, median_filter, trunc):
        self.smoothing_alpha = smoothing_alpha
        self.cutoff = cutoff
        self.order = order
        self.median_filter = median_filter
        self.trunc = trunc
        self.data_filtered = np.zeros_like(self.data)

        for j in range(0,self.N_ul_joints):
            joint = np.squeeze(self.data[j, :, :])
            mf_len = int(self.fs*1.5)
            joint = PF.position_median_filter(joint, self.fs/self.median_filter)
            joint = PF.exp_smoothing(joint,self.fs, self.smoothing_alpha)
            joint = PF.butter_lowpass_filter(joint, self.cutoff, self.fs, self.order)   # Filtering with a 5th order butterworth lowpass filter @ 6Hz.
            self.data_filtered[j, :, :] = joint
