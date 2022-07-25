import numpy as np
'''
    Copyright 2022 Jan Breuer & Shivakeshavan Ratnadurai-Giridharan

    This is the sheet where different variables can be altered that would otherwise be hardcoded! 
'''

ViconFiles = 'Scratch/Reaching'
PeackFiles = 'scratch_peack/Reaching'

# participants for validation - if value = 3, only subject four will be used
participants = [0]
partNum = 3 # 17 = max for current directory
movNum = 4
maxTrials = 11

# select joints based on kinematic allocation in csv
jointsPEACK = [3,4,5,6,7,8,2,9]; jointsVICON = [1,3,6,9,11,14,18,19]
### tweak that so that no all data columns will be evaluated and computed

#turn individual joints on/off
RShoulder = True
RElbow = True
RWrist = True
LShoulder = True
LElbow = True
LWrist = True
Neck = True
Midpoint = True #MidSternum for VICON and MidHip for PEACK

t = [RShoulder,RElbow, RWrist, LShoulder, LElbow, LWrist, Neck, Midpoint]
jointsPEACK = [jointsPEACK[i] for i in np.where(t)[0]]
jointsVICON = [jointsVICON[i] for i in np.where(t)[0]]
# assert(len(jointsVICON)==len(jointsPEACK))

# fs, N_ul_joints
fs_vicon = 100
fs_peack = 60
N_ul_joints = len(jointsPEACK)
joint_selection = 2
joint_selector = 'RWrist'

# time edge artifact removal
truncating = True
if truncating:
    trunc = [500,500]
else: 
    trunc = [0,0]

# filter settings POSITION DATA
filter_position_data = True
smoothing_alpha = 0.1
cutoff = 3
order = 5

# filter settings VELOCITY
LowPassEnable = True
LowPassCutOff = 3
LowPassOrder = 5
MedianFilterEnable = True
MedianFilterWindowMultiplier = 0.5
max_suppression_threshold = 100
peack_detect_threshold_multiplier = 0.2

# turn on debugging mode
debug = False

#truncating debug
trunc_debug = False
debug_time = False




