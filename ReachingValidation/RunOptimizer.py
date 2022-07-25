from cmath import nan
from tkinter import N
from ExtractKinematicData import ExtractKinematicData
import matplotlib.pyplot as plt
from SegmentMovement import segment
import pandas as pd
from DataLoader import DataLoader
import variableDeclaration as var
from scipy import optimize
import os
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from GUI import MyApp as MA
import warnings
warnings.filterwarnings("ignore")
os.system('cls' if os.name == 'nt' else 'clear') # clears terminal

#Load raw data
ParticipantData = [];

def LoadRaw():

    all_joints = MA.initialize_joints(MA,["RShoulder", "RWrist", "RElbow"])
    assert(DataLoader(var.ViconFiles).NsubDirs == DataLoader(var.PeackFiles).NsubDirs) # make sure that the (sub)directories are the same
    # loop through the number of participants you want to compare
    for i in var.participants:#ViconFiles.NsubDirs):
        assert(DataLoader(var.ViconFiles).SubDirs[i].Nfiles==DataLoader(var.PeackFiles).SubDirs[i].Nfiles)
        print('Subject: ',i+1,'   Trial: ', end=" ", flush=True) # shows a progress bar through the trials
        # loop through each trial of the i-th participant
        Participant = []
        for j in range(0,DataLoader(var.ViconFiles).SubDirs[i].Nfiles):
            print(j+1, end=" ", flush=True) # updates the progress bar through the trials
            Temp = ExtractKinematicData(DataLoader(var.ViconFiles).getFile(i,j), all_joints, var.fs_vicon, len(all_joints[0]), unit_rescale=1000.0, type='VICON')
            Peack = ExtractKinematicData(DataLoader(var.PeackFiles).getFile(i,j), all_joints, var.fs_peack, len(all_joints[0]), unit_rescale=1.0, type='PEACK')
            #Vicon = np.interp(Peack.time,Temp.time,Temp.data)
            f = scipy.interpolate.interp1d(Temp.time, Temp.data, axis=1, kind='cubic')
            Vicon = Temp
            Vicon.data = f(Peack.time)
            Vicon.time = Peack.time
            Vicon.fs = Peack.fs

            Participant.append([Vicon, Peack])

        ParticipantData.append(Participant)
        # print(ParticipantData[0][0][0].data.shape)
        # breakpoint()

        # Structure ParticipantData [Participant - list][trials - list][type - list][attributes - class][joints]

def crosscorr(VICON,PEACK,VICON_time,PEACK_time):
    # express the VICON data in the time domain of PEACK
    #temp = np.interp(PEACK_time,VICON_time,VICON)
    temp = VICON
    # calculate correlation and lag
    temp = np.squeeze(temp)
    PEACK = np.squeeze(PEACK)
    corr_scipy= scipy.signal.correlate(temp - np.mean(temp), PEACK - np.mean(PEACK), mode='full')
    corr_scipy_lag=scipy.signal.correlation_lags(temp.size, PEACK.size, mode='full')
    lag = corr_scipy_lag[np.argmax(corr_scipy)]

    if(lag==0):
        try: pearson = scipy.stats.pearsonr(temp, PEACK)[0]
        except: pearson = 0

        try: rmse = mean_squared_error(temp, PEACK, squared=False)
        except: rmse = nan;
    else:
        try: pearson = scipy.stats.pearsonr(temp[lag:], PEACK[:-lag])[0]
        except: pearson = 0

        try: rmse = mean_squared_error(temp[lag:], PEACK[:-lag], squared=False)
        except: rmse = nan;


    lag_sec = PEACK_time[lag]-PEACK_time[1]
    return [pearson, rmse, lag_sec] # returns corr_score and lag in seconds

bounds = [[0.1, 0.6],[0.4, 6],[1, 6],[1, 5],[0.4, 6],[1, 6],[1, 5]]
def Optimize(x, draw_option=False):

    x2 = x[:]
    for cb in range(len(x2)):
        x2[cb] = bounds[cb][0] + x[cb]*(bounds[cb][1] - bounds[cb][0])
        if np.isnan(x2[cb]):
            x2[cb] = bounds[cb][0]


    smoothing_alpha = x2[0]
    cutoff=x2[1]
    order=round(x2[2])
    median_filter=x2[3]
    cutoff_vel=x2[4]
    order_vel=round(x2[5])
    median_filter_vel=x2[6]
    trunc = [200,0]

    CorrMatrix=[]

    if draw_option==True:
        fig, axs = plt.subplots(3,4)


    for i in range(len(ParticipantData)): # Number of Participant
        VICON_speed_filtered = []; PEACK_speed_filtered = []; PEACK_time = []; VICON_time = []; VICON_stamps = []; PEACK_stamps = []
        for j in range(len(ParticipantData[i])): # Number of Trials

            ParticipantData[i][j][0].reFilter(smoothing_alpha, cutoff, order, median_filter, trunc)    #VICON
            ParticipantData[i][j][1].reFilter(smoothing_alpha, cutoff, order, median_filter, trunc)    #PEACK

            VICON_speed_filtered = segment(ParticipantData[i][j][0]).get_speed_filtered(cutoff_vel, order_vel, median_filter_vel)
            VICON_speed = segment(ParticipantData[i][j][0]).get_speed(cutoff_vel, order_vel, median_filter_vel,filt_vel=False)
            VICON_time = ParticipantData[i][j][0].time[1:]

            PEACK_speed_filtered = segment(ParticipantData[i][j][1]).get_speed_filtered(cutoff_vel, order_vel, median_filter_vel)
            PEACK_time = ParticipantData[i][j][1].time[1:]

            

            if draw_option==True:
                plot_timeseries(
                    np.squeeze(VICON_speed_filtered),
                    np.squeeze(PEACK_speed_filtered),
                    np.squeeze(VICON_time),
                    np.squeeze(PEACK_time),
                    axs[int(j/4)][j%4])


            for k in range(0, len(VICON_speed_filtered)):
                corr1 = crosscorr(VICON_speed_filtered[k], PEACK_speed_filtered[k], VICON_time, PEACK_time)    # Cost fn for VICON vs PEACK matching
                corr2 = crosscorr(VICON_speed_filtered[k], VICON_speed[k], VICON_time, VICON_time)    # Cost fn for VICON filtered vs Unfiltered
                CorrMatrix.append(-(corr1[0] + corr2[0])/2)    #Reverse correlation score for loss fn minimization
                #CorrMatrix.append(corr[1])
        
    if draw_option==True:
        plt.show()
    loss = sum(CorrMatrix)/len(CorrMatrix)
    print(x)
    print(loss)
    return loss


def plot_timeseries(vicon,peack, t1, t2, ax):
    '''
        Defining a function to plot both the PEACK and VICON data
    '''
    ax.plot(t1,vicon, color='tab:blue')
    # ax.plot(t2,peack_org,'--', color='tab:pink')
    ax.plot(t2,peack, color='tab:red')
    ax.set(xlabel='Time (s)', ylabel='Velocity')
    ax.grid()
    plt.tight_layout()
    # fig1.savefig("Figures//Limbs_angles_LElbow.png", dpi=600, bbox_inches='tight')

LoadRaw()

smoothing_alpha=0.1         # Declare exponential smoothing filter
cutoff=3                  # Declaring lowPass Cutoff frequency in Hz
order=5                     # Declare order of filtering
median_filter = 2           # Declare what the window size of the median filter should be.. value of x = 1/x sec

# VELOCITY FILTER PARAMETERS
cutoff_vel=3              # Declaring lowPass Cutoff frequency in Hz
order_vel=5                 # Declare order of filtering
median_filter_vel = 2       # Declare what the window size of the median filter should be.. value of x = 1/x sec

#x0 = [smoothing_alpha, cutoff, order, median_filter, cutoff_vel, order_vel, median_filter_vel]
x0 = np.random.random_sample(size = 7)
#print(x0)
#x = [0.1, 5.9756744 , 1.95686406, 2.85101124, 4.96125351, 1.85919413, 1.]
#x = [0.1, 4.55930863, 2.12608292, 1.10952607, 0.60029232, 3.06894431, 1.]
#x = [0.1, 5.19829084, 2.30806267, 3.07973078, 4.75285561, 1.78954535, 1.27529314]
#x = [0.03520499, 1.        , 0.96276599, 0.61251912, 1.        ,1.        , 0.        ] #0.73 Corr z-scored
minimizer_kwargs = {"method":"L-BFGS-B", "jac":False, "bounds":([0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]), "options":{'gtol':1e-6, 'eps': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 'maxiter': 100}}
optimal = optimize.basinhopping(Optimize, x0, minimizer_kwargs=minimizer_kwargs, niter=20) # change iterations back to 20

#Optimize(x, True)
print("Optimal params", optimal)
Optimize(optimal.x,True)

'''
Optimal values
'''
#[0.03520499, 1.        , 0.96276599, 0.61251912, 1.        ,1.        , 0.        ] #0.73 Corr z-scored
#array([0.25743905, 4.04267077, 1.52984152, 2.43236397, 1.63464633, 3.77491474, 1.01796172]) 0.28 Corr
#array([0.16938533, 3.09557495, 1.80669756, 2.03098639, 1.55759179, 3.51167098, 2.0207994 ]) 0.27 Corr
