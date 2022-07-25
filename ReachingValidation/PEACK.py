import numpy as np
import variableDeclaration as var
'''
    Copyright 2022 Jan Breuer & Shivakeshavan Ratnadurai-Giridharan

    A function for selecting Upperlimb timeseries from PEACK kinematic data.

'''

def get_joint(pdata,limb_idx,ref_idx,start,samples):
    Data = np.zeros((pdata.shape[0], 3))

    # assigns the adjacent zero array with positonal data of the indv. joint
    if ref_idx <=0:
        if samples <=0:
            Data = pdata[start:, 3*(limb_idx-1):3*(limb_idx-1)+3]
        else:
            Data = pdata[start:start+samples, 3*(limb_idx-1):3*(limb_idx-1)+3]
    else:
        if samples <=0:
            Ref = pdata[start:,3*(ref_idx-1):3*(ref_idx-1)+3]
            Data = pdata[start:,3*(limb_idx-1):3*(limb_idx-1)+3] - Ref
        else:
            Ref = pdata[start:, 3 * (ref_idx - 1):3 * (ref_idx - 1) + 3]
            Data = pdata[start:start+samples, 3 * (limb_idx - 1):3 * (limb_idx - 1) + 3] - Ref

    return Data

def get_UL_prop(pdata, all_joints, start, samples, n_joints, type):
    # creates an zero-array with number-of-joints subarrays for positional data later
    if samples <= 0:
        data = np.zeros((n_joints, pdata.shape[0]-start+1, 3))
    else:
        data = np.zeros((n_joints, samples, 3))

    start = start - 1; ref_idx = -1

    # loops through all kinematic data to find adjacent joints
    if type == 'PEACK':
        for i in range(0,n_joints):
            data[i, :, :] = get_joint(pdata, all_joints[1][i], ref_idx, start, samples)
    elif type == 'VICON':
        for i in range(0,n_joints): ### apparently range(0,0) is not entered at all but range(0,1) twice?
            # a special check for elbow data is necessary since VICON has two points
            if all_joints[0][i] == 3 or all_joints[0][i] == 11:
                t1 = get_joint(pdata, all_joints[0][i], ref_idx, start, samples)
                t2 = get_joint(pdata, all_joints[0][i]+1, ref_idx, start, samples)
                data[i, :, :] = (t1 + t2)/2.0
            else: 
                data[i, :, :] = get_joint(pdata, all_joints[0][i], ref_idx, start, samples)
    else: 
        raise Exception("Invalid file type. Please choose PEACK or VICON only.")
    
    return data
