from cmath import nan
from multiprocessing import set_forkserver_preload
import sys
import pickle
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, Qt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import *
from PyQt5 import QtCore 
from PyQt5 import QtWidgets 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from reaching_validation_gui import receive_data
from reaching_validation_gui import calc_correlation
from SegmentMovement import segment
import variableDeclaration as var
from visualization_window import visualization_win, segment_correlation
from select_joints_window import select_joints_win
from parameter_modules import calc_parameter_matrix

'''
    Copyright 2022 Jan Breuer & Shivakeshavan Ratnadurai-Giridharan

    This is file contains functions for the main GUI, for the calculation of parameter tensors and the initialization of parameters 
'''

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initialize_values()
        self.initialize_layout()

    def initialize_values(self):
        print("Starting GUI..."); print("Starting initialization phase...")
        print(); print("Loading exemplary participant...")
        self.late_trunc_state = True; self.joints = ["RWrist"]; self.joint_selection = 0
        self.all_joints = self.initialize_joints(self.joints)
        part_init = [4] # first starting participant
        # receiving values from starting values
        self.single_data_dict = receive_data( part_init, self.all_joints, smoothing_alpha=0.1, cutoff=1.55, order=2, median_filter = 2.42, cutoff_vel=5.96, order_vel=4, median_filter_vel = 2.68, trunc = [200,0])
        self.corr = calc_correlation(self.single_data_dict["VICON_speed_filtered"], self.single_data_dict["PEACK_speed_filtered"], self.single_data_dict["VICON_time"], self.single_data_dict["PEACK_time"])

        print(); print("follow GUI for further steps...")
        self.all_data_dict = { # create starting dictionary
            "VICON_original" : [self.single_data_dict["VICON_speed_original"]],
            "VICON_filtered" : [self.single_data_dict["VICON_speed_filtered"]],
            "VICON_raw" : [self.single_data_dict["VICON_speed_raw"]],
            "VICON_late" : [self.single_data_dict["VICON_speed_late_trunc"]],
            "PEACK_original" : [self.single_data_dict["PEACK_speed_original"]],
            "PEACK_filtered" : [self.single_data_dict["PEACK_speed_filtered"]],
            "PEACK_raw" : [self.single_data_dict["PEACK_speed_raw"]],
            "PEACK_late" : [self.single_data_dict["PEACK_speed_late_trunc"]],
            "VICON_time" : [self.single_data_dict["VICON_time"]],
            "PEACK_time" : [self.single_data_dict["PEACK_time"]],
            "VICON_stamps": [self.single_data_dict["VICON_stamps"]],
            "PEACK_stamps": [self.single_data_dict["PEACK_stamps"]],
            "VICON_positional": [self.single_data_dict["VICON_positional"]],
            "PEACK_positional": [self.single_data_dict["PEACK_positional"]],
            "corr" : [self.corr[0]],
            "corr_lag" : [self.corr[1]]}
        # stamps organization: [participants:list][trials:list][joints:list][timestamps:nparray]
        # stored in the way all_data_dict[part][trials][n_joints][actual velocity data]
    
    def initialize_joints(self, joints):
        jointsPEACK = [3,4,5,6,7,8,2,9]; jointsVICON = [1,3,6,9,11,14,18,19]
        RShoulder = True if 'RShoulder' in joints else False
        RElbow = True if 'RElbow' in joints else False
        RWrist = True if 'RWrist' in joints else False
        LShoulder = True if 'LShoulder' in joints else False
        LElbow = True if 'LElbow' in joints else False
        LWrist = True if 'LWrist' in joints else False
        Neck = True if 'Neck' in joints else False
        Midpoint = True if 'Midpoint' in joints else False
        t = [RShoulder,RElbow, RWrist, LShoulder, LElbow, LWrist, Neck, Midpoint]
        joint_names = ["RShoulder","RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "Neck", "Midpoint"]
        jointsPEACK = [jointsPEACK[i] for i in np.where(t)[0]]
        jointsVICON = [jointsVICON[i] for i in np.where(t)[0]]
        joint_names = [joint_names[i] for i in np.where(t)[0]]
        alljoints = [jointsVICON, jointsPEACK, joint_names]
        return alljoints
        
    def initialize_layout(self):
        self.setWindowTitle('Validation GUI')
        self.window_width, self.window_height = 600, 400
        self.setMinimumSize(self.window_width, self.window_height)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.title = QLabel()
        self.title.setFont(QtGui.QFont('Arial',14))
        self.title.setText('Select Parameters')
        self.fontsize = 27
        self.parameter_layout = QGridLayout()

        self.filter_label_pos = QLabel('Positional filter parameters')
        self.filter_label_pos.setFont(QtGui.QFont('Arial',self.fontsize))
        self.parameter_layout.addWidget(self.filter_label_pos,1,1)

        self.filter_label_vel = QLabel('Velocity filter parameters')
        self.parameter_layout.addWidget(self.filter_label_vel,1,2)
        self.filter_label_vel.setFont(QtGui.QFont('Arial',self.fontsize))
    
        self.smoothing_alpha_selection = QLineEdit()
        self.smoothing_alpha_selection.setText('0.1')
        self.smoothing_alpha_label = QLabel('Select alpha')
        self.smoothing_alpha_label.setFont(QtGui.QFont('Arial',self.fontsize))
        self.parameter_layout.addWidget(self.smoothing_alpha_label,2,0)
        self.parameter_layout.addWidget(self.smoothing_alpha_selection,2,1)

        self.cutoff_selection = QLineEdit()
        self.cutoff_selection.setText('1.55')
        self.cutoff_selection_vel = QLineEdit()
        self.cutoff_selection_vel.setText('5.96')
        self.cutoff_label = QLabel('Select cutoff')
        self.cutoff_label.setFont(QtGui.QFont('Arial',self.fontsize))
        self.parameter_layout.addWidget(self.cutoff_label,3,0)
        self.parameter_layout.addWidget(self.cutoff_selection,3,1)
        self.parameter_layout.addWidget(self.cutoff_selection_vel,3,2)

        self.order_selection = QLineEdit()
        self.order_selection.setText('2')
        self.order_selection_vel = QLineEdit()
        self.order_selection_vel.setText('4')
        self.order_label = QLabel('Select order')
        self.order_label.setFont(QtGui.QFont('Arial',self.fontsize))
        self.parameter_layout.addWidget(self.order_label,4,0)
        self.parameter_layout.addWidget(self.order_selection,4,1)
        self.parameter_layout.addWidget(self.order_selection_vel,4,2)

        self.median_filter_selection = QLineEdit()
        self.median_filter_selection.setText('2.42')
        self.median_filter_selection_vel = QLineEdit()
        self.median_filter_selection_vel.setText('2.68')
        self.median_filter_label = QLabel('Select median filter')
        self.median_filter_label.setFont(QtGui.QFont('Arial',self.fontsize))
        self.parameter_layout.addWidget(self.median_filter_label,5,0)
        self.parameter_layout.addWidget(self.median_filter_selection,5,1)
        self.parameter_layout.addWidget(self.median_filter_selection_vel,5,2)


        self.enable_late_trunc = QCheckBox("Enable late trunc")
        self.trunc_start_label = QLabel('Start')
        self.trunc_start_label.setFont(QtGui.QFont('Arial',self.fontsize))
        self.trunc_end_label = QLabel('End')
        self.trunc_end_label.setFont(QtGui.QFont('Arial',self.fontsize))
        self.parameter_layout.addWidget(self.enable_late_trunc,7,0)
        self.parameter_layout.addWidget(self.trunc_start_label,7,1)
        self.parameter_layout.addWidget(self.trunc_end_label,7,2)

        self.trunc_start_selection = QLineEdit()
        self.trunc_start_selection.setText('200')
        self.trunc_end_selection = QLineEdit()
        self.trunc_end_selection.setText('0')
        self.trunc_label = QLabel('Truncating')
        self.trunc_label.setFont(QtGui.QFont('Arial',self.fontsize))
        self.parameter_layout.addWidget(self.trunc_label,8,0)
        self.parameter_layout.addWidget(self.trunc_start_selection,8,1)
        self.parameter_layout.addWidget(self.trunc_end_selection,8,2)

        self.select_joints_button = QPushButton('Select Joints (MUST: RWrist)')
        self.select_joints_button.clicked.connect(self.open_select_joints)

        self.visualize_button = QPushButton('Show Graphs')
        self.visualize_button.clicked.connect(self.open_visualization)

        self.create_matrix_button = QPushButton('Update Parameters / Create Matrix')
        self.create_matrix_button.clicked.connect(self.create_matrix)
        self.create_matrix_button.setEnabled(False)

        self.calc_parameters_button = QPushButton('Print Parameter')
        self.calc_parameters_button.clicked.connect(self.calc_parameter)
        self.calc_parameters_button.setEnabled(False)

        # add a row for a slider to select the trial
        self.data_choice = 0;
        self.data_choice_type = ["Time to peak", "Peak Velocity", "Mean Velocity", "Trajectory Variability", "Total Path", "Acceleration Mean / Peak", "Number of Submovements"]

        self.data_choice_label = QLabel(self.data_choice_type[self.data_choice])
        self.data_choice_label.setFont(QtGui.QFont('Arial',10))

        self.data_choice_selection = QSlider(Qt.Horizontal, self)
        self.data_choice_selection.setGeometry(30, 40, 200, 30)
        self.data_choice_selection.setRange(0, len(self.data_choice_type)-1)
        self.data_choice_selection.valueChanged[int].connect(self.update_data_sel)

        self.layout.addWidget(self.title,1,0, QtCore.Qt.AlignHCenter)
        self.layout.addLayout(self.parameter_layout,2,0)
        self.layout.addWidget(self.select_joints_button,3,0)
        self.layout.addWidget(self.visualize_button,4,0)
        self.layout.addWidget(self.create_matrix_button,5,0)
        self.layout.addWidget(self.calc_parameters_button,6,0)
        self.layout.addWidget(self.data_choice_label,7,0)
        self.layout.addWidget(self.data_choice_selection,8,0)

        self.layout.setContentsMargins(100,10,100,100)

    def update_data_sel(self):
        self.data_choice = self.data_choice_selection.value()
        self.data_choice_label.setText(self.data_choice_type[self.data_choice])

    def calc_parameter(self):
        calc_parameter_matrix(self.all_data_dict, self.data_choice, self.joint_selection)


    def create_matrix(self):
        self.calc_parameters_button.setEnabled(True)

        smoothing_alpha = self.smoothing_alpha_selection.text()
        cutoff = self.cutoff_selection.text()
        order = self.order_selection.text()
        median_filter = self.median_filter_selection.text()
        cutoff_vel = self.cutoff_selection_vel.text()
        order_vel = self.order_selection_vel.text()
        median_filter_vel = self.median_filter_selection_vel.text()
        trunc_start = self.trunc_start_selection.text()
        trunc_end = self.trunc_end_selection.text()

        try: smoothing_alpha = float(smoothing_alpha)
        except ValueError: smoothing_alpha = 0

        try:
            cutoff = float(cutoff)
            if cutoff < 0.1:
                cutoff=0.1
        except ValueError: cutoff = 1

        try:
            order = int(order)
            if order < 1:
                order=1
        except ValueError: order = 1

        try:
            median_filter = float(median_filter)
            if median_filter < 0.1:
                median_filter=0.1
        except ValueError: median_filter = 1

        try:
            cutoff_vel = float(cutoff_vel)
            if cutoff_vel < 0.1:
                cutoff_vel=0.1
        except ValueError: cutoff_vel = 1

        try:
            order_vel = int(order_vel)
            if order_vel < 1:
                order_vel=1
        except ValueError: order_vel = 1

        try:
            median_filter_vel = float(median_filter_vel)
            if median_filter_vel < 0.1:
                median_filter_vel=0.1
        except ValueError: median_filter_vel = 1

        try: trunc_start = float(trunc_start)
        except ValueError: trunc_start = 0

        try: trunc_end = float(trunc_end)
        except ValueError: trunc_end = 0
        
        trunc = [trunc_start, trunc_end]
        os.system('cls' if os.name == 'nt' else 'clear') # clears terminal
        self.all_data_dict = {
            "VICON_original" : [],
            "VICON_filtered" : [],
            "VICON_raw" : [],
            "VICON_late" : [],
            "VICON_velocity_segmented" : [],
            "PEACK_original" : [],
            "PEACK_filtered" : [],
            "PEACK_raw" : [],
            "PEACK_late" : [],
            "VICON_time" : [],
            "PEACK_time" : [],
            "PEACK_velocity_segmented" : [],
            "VICON_stamps" : [],
            "PEACK_stamps" : [],
            "VICON_positional" : [],
            "PEACK_positional" : [],
            "corr" : [],
            "corr_lag" : [],
            "rmse" : [],
            "corr_late" : []
        }

        for part in range(0,var.partNum):
            participant = [part]
            self.single_data_dict = receive_data(participant, self.all_joints, smoothing_alpha, cutoff, order, median_filter, cutoff_vel, order_vel, median_filter_vel, trunc, self.enable_late_trunc.isChecked())
            
            self.all_data_dict["VICON_original"].append(self.single_data_dict["VICON_speed_original"])
            self.all_data_dict["VICON_filtered"].append(self.single_data_dict["VICON_speed_filtered"])
            self.all_data_dict["VICON_raw"].append(self.single_data_dict["VICON_speed_raw"])
            self.all_data_dict["VICON_late"].append(self.single_data_dict["VICON_speed_late_trunc"])
            self.all_data_dict["PEACK_original"].append(self.single_data_dict["PEACK_speed_original"])
            self.all_data_dict["PEACK_filtered"].append(self.single_data_dict["PEACK_speed_filtered"])
            self.all_data_dict["PEACK_late"].append(self.single_data_dict["PEACK_speed_late_trunc"])
            self.all_data_dict["PEACK_raw"].append(self.single_data_dict["PEACK_speed_raw"])
            self.all_data_dict["VICON_time"].append(self.single_data_dict["VICON_time"])
            self.all_data_dict["PEACK_time"].append(self.single_data_dict["PEACK_time"])
            self.all_data_dict["VICON_stamps"].append(self.single_data_dict["VICON_stamps"])
            self.all_data_dict["PEACK_stamps"].append(self.single_data_dict["PEACK_stamps"])
            self.all_data_dict["VICON_positional"].append(self.single_data_dict["VICON_positional"])
            self.all_data_dict["PEACK_positional"].append(self.single_data_dict["PEACK_positional"])
            self.corr = calc_correlation(self.single_data_dict["VICON_speed_filtered"], self.single_data_dict["PEACK_speed_filtered"], self.single_data_dict["VICON_time"], self.single_data_dict["PEACK_time"])            
            self.all_data_dict["corr"].append(self.corr[0])
            self.all_data_dict["rmse"].append(self.corr[1])
            self.all_data_dict["corr_lag"].append(self.corr[2])

        os.system('cls' if os.name == 'nt' else 'clear') # clears terminal
        segmental_results_matrix = []

        for i in range(var.movNum):
            segmental_results_matrix.append(self.calculate_segment_matrix(i,False))
        # print(segmental_results_matrix)
        with open('velocity_correlation.pkl', 'wb') as f:
            pickle.dump(segmental_results_matrix, f) # save velocity parameter if wanted!
        # seg_res_df = pd.DataFrame(segmental_results_matrix)
        # seg_res_df.columns = self.all_joints[2] # this 2 here is not the wrist joint - just the third array
        # print(seg_res_df)

        matrix_calc(self.all_data_dict["corr"], 'pearson', self.all_joints)
        if self.enable_late_trunc.isChecked():
            self.corr_late = calc_correlation(self.single_data_dict["VICON_speed_late_trunc"], self.single_data_dict["PEACK_speed_late_trunc"], self.single_data_dict["VICON_time"], self.single_data_dict["PEACK_time"])
            self.all_data_dict["corr_late"].append(self.corr_late[0])
            matrix_calc(self.all_data_dict["corr_late"], 'pearson', self.all_joints)
        
        self.late_trunc_state = self.enable_late_trunc.isChecked()

    def calculate_segment_matrix(self,mov_segment, printing = False):
        joint_segmented = [ [] for _ in range(len(self.all_joints[0])) ]
        for joint in range(len(joint_segmented)):
            joint_segmented[joint] = [ [] for _ in range(0,var.partNum) ]
            for part in range(len(joint_segmented[joint])):
                joint_segmented[joint][part] = [ [] for _ in range(0,var.maxTrials) ]
        # create new entries for segmented data
        self.all_data_dict["VICON_velocity_segmented"]= self.create_dict_entry()
        self.all_data_dict["VICON_time_segmented"]= self.create_dict_entry()
        self.all_data_dict["PEACK_velocity_segmented"]= self.create_dict_entry()
        self.all_data_dict["PEACK_time_segmented"]= self.create_dict_entry()

        for participant in range(0,var.partNum):
            for trial in range(0,len(self.all_data_dict["VICON_filtered"][participant])):
                for joint in range(0,len(self.all_data_dict["VICON_filtered"][participant][trial])):
                    x = []
                    if len(segment.peaks_and_troughs(self.all_data_dict["VICON_filtered"][participant][trial][self.joint_selection],self.all_data_dict["VICON_time"][participant][trial])) == var.movNum and len(segment.peaks_and_troughs(self.all_data_dict["PEACK_filtered"][participant][trial][self.joint_selection],self.all_data_dict["PEACK_time"][participant][trial])) == var.movNum and len(self.all_data_dict["VICON_stamps"][participant][trial][self.joint_selection]) == var.movNum and len(self.all_data_dict["PEACK_stamps"][participant][trial][self.joint_selection]) == var.movNum:
                        for mov_segment_cnt in range(0,len(segment.peaks_and_troughs(self.all_data_dict["VICON_filtered"][participant][trial][self.joint_selection],self.all_data_dict["VICON_time"][participant][trial]))):
                            # calculate segemnted data, correlation and use all_joints[2] (hopefully wrist) for that!
                            temp=segment_correlation(self.all_data_dict["VICON_filtered"][participant][trial][joint],self.all_data_dict["PEACK_filtered"][participant][trial][joint],self.all_data_dict["VICON_time"][participant][trial],self.all_data_dict["PEACK_time"][participant][trial], mov_segment_cnt, self.all_data_dict["VICON_stamps"][participant][trial][self.joint_selection],self.all_data_dict["PEACK_stamps"][participant][trial][self.joint_selection]) # the number two of these two inputs indicates that the joint used to calculate the segmentation of the others is the third one in the all_joints array. This is not totally correct in our case, only if the third one is the RWrist 
                            x.append(temp[0][0])
                            self.all_data_dict["VICON_velocity_segmented"][participant][trial][joint].append(temp[1])
                            self.all_data_dict["VICON_time_segmented"][participant][trial][joint].append(temp[2])
                            self.all_data_dict["PEACK_velocity_segmented"][participant][trial][joint].append(temp[3])
                            self.all_data_dict["PEACK_time_segmented"][participant][trial][joint].append(temp[4])
                    else: 
                        x.append(0)
                    if len(x) == var.movNum:
                        joint_segmented[joint][participant][trial]=np.round(x[mov_segment], 3) 
                    else:
                        joint_segmented[joint][participant][trial]=0

        empty_cells = 0
        for joint in range(len(self.all_joints[2])):
            for part in range(var.partNum):
                for trial in range(var.maxTrials):
                    if joint_segmented[joint][part][trial] == []:
                        joint_segmented[joint][part][trial]=0
                        empty_cells += 1
        empty_cells = empty_cells/len(self.all_joints[2])

        correlation_results = []
        for i in range(len(self.all_joints[2])):
            joint_df = pd.DataFrame(joint_segmented[i])
            results=joint_df[joint_df > 0].count().sum()
            correlation_results.append([np.round(joint_df[joint_df != 0].mean().mean(),3),np.round((results/(joint_df.size-empty_cells))*100,1)])
            if printing:
                print("")
                print(self.all_joints[2][i]) 
                print(joint_df)
                print("")
                print("Average:",joint_df[joint_df != 0].mean().mean())
                print("Scored ", results,"/",joint_df.size-empty_cells,". ",np.round((results/(joint_df.size-empty_cells))*100,1),"%")
        
        # with open('saved_dictionary.pkl', 'wb') as f:
        #     pickle.dump(self.all_data_dict, f)
        
        return joint_segmented

    def open_visualization(self): #this opens a visualization window
        self.data_window = visualization_win(self.all_data_dict, self.late_trunc_state, self.all_joints, self.joint_selection)
        self.data_window.show()
        self.is_data_window_open = True  
    
    def open_select_joints(self): #this opens a joint selection window
        self.joints_window = select_joints_win()
        self.joints_window.submitClicked.connect(self.on_sub_window_confirm)
        self.joints_window.show()
        self.is_joints_window_open = True  
        self.create_matrix_button.setEnabled(True)
    
    def on_sub_window_confirm(self, url):  # <-- This is the main window's slot
        self.joints = url
        self.all_joints = self.initialize_joints(self.joints)
        for a in range(len(self.all_joints[2])):
            if self.all_joints[2][a] == var.joint_selector:
                self.joint_selection = a

    def create_dict_entry(self):
        segmented_data = [ [] for _ in range(var.partNum) ]
        for participant in range(len(segmented_data)):
            segmented_data[participant] = [ [] for _ in range(var.maxTrials) ]
            for trial in range(len(segmented_data[participant])):
                segmented_data[participant][trial] = [ [] for _ in range(len(self.all_joints[0])) ]
        return segmented_data


def matrix_calc(array, type, all_joints):
    print("")
    for joint in range(0, len(all_joints[2])):
        print("Correlation Matrix for ",all_joints[2][joint])
        a = 0; temp = 0
        for i in array:  
            if len(i[joint]) > a: a = len(i[joint]) 

        for i in array: 
            if len(i[joint]) < a:
                for j in range(0,a-len(i[joint])):
                    array[temp][joint].append(0)
            temp += 1

        joint_array = []
        for i in array:
            joint_array.append(i[joint])

        df = pd.DataFrame(list(joint_array))
        if len(array[0][0]) == 11:
            df.columns = ['Trial_1', 'Trial_2', 'Trial_3','Trial_4', 'Trial_5', 'Trial_6','Trial_7', 'Trial_8', 'Trial_9','Trial_10', 'Trial_11']
        elif len(array[0][0]) == 10:
            df.columns = ['Trial_1', 'Trial_2', 'Trial_3','Trial_4', 'Trial_5', 'Trial_6','Trial_7', 'Trial_8', 'Trial_9','Trial_10']

        print(df)
        print("")
        score = df[df != 0].mean().mean()
        score_no_outliers = df[df > 0.5].mean().mean()
        if type == "pearson":
            print("Mean Pearson Coefficient: ", score)
            print("Mean Pearson Coefficient w/o outliers: ", score_no_outliers)
        elif type == "rmse":
            print("Mean RMSE Score: ", score)
        print("")


if __name__ == '__main__':
    # don't auto scale when drag app to a different monitor.
    # QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 20px;
        }
    ''')
    
    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')


