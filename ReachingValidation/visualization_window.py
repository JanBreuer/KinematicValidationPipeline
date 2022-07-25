'''
This is visualization pop up window to display given parameters
'''

from cProfile import label
import sys
import os
import time
import csv
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QPainter, QBrush, QPen, QPolygon
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, Qt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import *
# from PyQt5 import QWidget
import numpy as np
from reaching_validation_gui import crosscorr
from SegmentMovement import segment


class visualization_win(QWidget):
    def __init__(self, all_data_dict, enable_late_trunc, all_joints, joint_selection):
        super().__init__()

        # initialize values
        self.all_data_dict = all_data_dict
        self.enable_late_trunc = enable_late_trunc
        self.all_joints = all_joints
        self.joint_select = joint_selection
        print(self.all_joints[2])

        self.initialize_layout()
        self.insert_ax()

    def initialize_layout(self):
        self.setMinimumSize(600,600)
        self.setWindowIcon(QtGui.QIcon('utils/logo_icon.jpg'))
        self.setWindowTitle('Plotting Validation Graphs')

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.title_layout = QHBoxLayout()
        self.parameter_layout = QGridLayout()

        # title - shows at top of gui
        self.title = QLabel()
        self.title.setFont(QtGui.QFont('Arial',14))
        self.title.setText('Plot Velocity Validation')
        self.title_layout.addWidget(self.title)

        # add a row for slider to select the participant
        self.participant_label = QLabel('Select Participant')
        self.participant_label.setFont(QtGui.QFont('Arial',10))

        self.participant_selection = QSlider(Qt.Horizontal, self)
        self.participant_selection.setGeometry(30, 40, 200, 30)
        self.participant_selection.setRange(0, len(self.all_data_dict["VICON_original"])-1)
        self.participant_selection.valueChanged[int].connect(self.update_chart)
        
        self.participantname = "Participant 1"
        self.participant_choice = QLabel(self.participantname)
        self.participant_choice.setFont(QtGui.QFont('Arial',10))

        self.parameter_layout.addWidget(self.participant_label,1,0)
        self.parameter_layout.addWidget(self.participant_selection,1,1)
        self.parameter_layout.addWidget(self.participant_choice,1,2)

        # add a row for a slider to select the trial
        self.trial_label = QLabel('Select trial')
        self.trial_label.setFont(QtGui.QFont('Arial',10))

        self.trial_selection = QSlider(Qt.Horizontal, self)
        self.trial_selection.setGeometry(30, 40, 200, 30)
        self.trial_selection.setRange(0, len(self.all_data_dict["VICON_original"][0])-1)
        self.trial_selection.valueChanged[int].connect(self.update_chart)
        
        self.trialname = "Trial 1"
        self.trial_choice = QLabel(self.trialname)
        self.trial_choice.setFont(QtGui.QFont('Arial',10))

        self.parameter_layout.addWidget(self.trial_label,2,0)
        self.parameter_layout.addWidget(self.trial_selection,2,1)
        self.parameter_layout.addWidget(self.trial_choice,2,2)

        # add a row for a slider to select the joint

        ### a little more tricky since the slider has to go through names and not count up numbers
        self.joint_label = QLabel('Select joint')
        self.joint_label.setFont(QtGui.QFont('Arial',10))

        self.joint_selection = QSlider(Qt.Horizontal, self)
        self.joint_selection.setGeometry(30, 40, 200, 30)
        self.joint_selection.setRange(0, len(self.all_data_dict["VICON_original"][0][0])-1)
        self.joint_selection.valueChanged[int].connect(self.update_chart)
        
        self.jointname = self.all_joints[2][0]
        self.joint_choice = QLabel(self.jointname)
        self.joint_choice.setFont(QtGui.QFont('Arial',10))

        self.parameter_layout.addWidget(self.joint_label,3,0)
        self.parameter_layout.addWidget(self.joint_selection,3,1)
        self.parameter_layout.addWidget(self.joint_choice,3,2)

        # add a tick box for displaying the original values
        self.show_original_selection = QCheckBox("Overlay raw VICON")
        self.show_original_selection.stateChanged.connect(self.update_chart)
        self.parameter_layout.addWidget(self.show_original_selection,4,0)

        self.show_legend = QCheckBox("Show legend")
        self.show_legend.stateChanged.connect(self.update_chart)
        self.parameter_layout.addWidget(self.show_legend,4,1)

        self.show_unfilt_vel = QCheckBox("Show unfiltered")
        self.show_unfilt_vel.stateChanged.connect(self.update_chart)
        self.parameter_layout.addWidget(self.show_unfilt_vel,4,2)

        self.show_late_trunc = QCheckBox("Show late trunc")
        self.show_late_trunc.stateChanged.connect(self.update_chart)
        self.parameter_layout.addWidget(self.show_late_trunc,5,0)
        self.show_late_trunc.setEnabled(self.enable_late_trunc)

        # initialize a space for a figure
        self.canvas = FigureCanvas(plt.Figure(figsize=(20, 8)))
        
        # specifiy properties of the layout
        self.layout.setContentsMargins(100,10,100,100)
        self.layout.addLayout(self.title_layout,0,0,1,-1, QtCore.Qt.AlignHCenter)
        self.layout.addLayout(self.parameter_layout,1,0)
        self.layout.addWidget(self.canvas,2,0)

    def insert_ax(self):
        font = {
            'weight': 'normal',
            'size': 16
        }
        matplotlib.rc('font', **font)

        # over here it apparently goes [participant][joint][trial]
        lag = self.all_data_dict["corr_lag"][0][0][0]
        
        print("lag ", lag)

        self.ax = self.canvas.figure.subplots(1,2)
        self.plot = self.ax[0].plot(self.all_data_dict["VICON_time"][0][0], self.all_data_dict["VICON_filtered"][0][0][0], color='tab:blue')
        self.plot = self.ax[0].plot(self.all_data_dict["PEACK_time"][0][0]+lag, self.all_data_dict["PEACK_filtered"][0][0][0], color='tab:red')
        self.ax[0].set(xlabel='Time (s)', ylabel='Velocity')
        self.ax[0].grid()
        self.ax[0].set_title('Filtered data')

        for i in self.all_data_dict["VICON_stamps"][0][0][0]: 
            self.plot = self.ax[0].plot(self.all_data_dict["VICON_time"][0][0][i], self.all_data_dict["VICON_filtered"][0][0][0][i], 'bx') 
        for i in self.all_data_dict["PEACK_stamps"][0][0][0]: 
                self.plot = self.ax[0].plot(self.all_data_dict["PEACK_time"][0][0][i]+lag, self.all_data_dict["PEACK_filtered"][0][0][0][i], 'rx') 


        self.plot = self.ax[1].plot(self.all_data_dict["VICON_time"][0][0], self.all_data_dict["VICON_original"][0][0][0], '--', color='tab:cyan')
        self.plot = self.ax[1].plot(self.all_data_dict["PEACK_time"][0][0]+lag, self.all_data_dict["PEACK_original"][0][0][0], '--', color='tab:pink')
        self.ax[1].set(xlabel='Time (s)', ylabel='Velocity')
        self.ax[1].grid()
        self.ax[1].set_title('Raw data')

    def update_chart(self):
        participant = self.participant_selection.value()
        self.participantname = "Participant " + str(participant+1)
        self.participant_choice.setText(self.participantname)

        trial = self.trial_selection.value()
        self.trialname = "Trial " + str(trial+1)
        self.trial_choice.setText(self.trialname)

        joint = self.joint_selection.value()
        self.jointname = self.all_joints[2][joint]
        self.joint_choice.setText(self.jointname)

        try:
            trial = int(trial)
            if trial>=len(self.all_data_dict["VICON_time"][participant]) or trial < 0:
                trial=0
        except ValueError:
            trial = 0

        lag = self.all_data_dict["corr_lag"][participant][joint][trial]

        if self.plot:
            self.ax = self.canvas.figure.clf()
            self.ax = self.canvas.figure.subplots(1,2)
            self.ax[0].set(xlabel='Time (s)', ylabel='Velocity')
            self.ax[1].set(xlabel='Time (s)', ylabel='Velocity')
            self.ax[0].grid(); self.ax[1].grid()
            self.ax[0].set_title('Filtered data'); self.ax[1].set_title('Raw data')
        
        if self.show_late_trunc.isChecked() == True:
            self.plot = self.ax[0].plot(self.all_data_dict["VICON_time"][participant][trial], self.all_data_dict["VICON_late"][participant][trial][joint], color='tab:blue', label="VICON")
            self.plot = self.ax[0].plot(self.all_data_dict["PEACK_time"][participant][trial]+lag, self.all_data_dict["PEACK_late"][participant][trial][joint], color='tab:red', label="PEACK")
        else:
            self.plot = self.ax[0].plot(self.all_data_dict["VICON_time"][participant][trial], self.all_data_dict["VICON_filtered"][participant][trial][joint], color='tab:blue', label="VICON")
            self.plot = self.ax[0].plot(self.all_data_dict["PEACK_time"][participant][trial]+lag, self.all_data_dict["PEACK_filtered"][participant][trial][joint], color='tab:red', label="PEACK")
        
        if self.show_original_selection.isChecked() == True:
            self.plot = self.ax[0].plot(self.all_data_dict["VICON_time"][participant][trial], self.all_data_dict["VICON_original"][participant][trial][joint],'--', color='tab:cyan', label="VICON")
            # self.plot = self.ax[0].plot(self.PEACK_time[participant][trial], self.PEACK_speed_original[participant][trial],'--', color='tab:pink', label="PEACK")
        if self.show_unfilt_vel.isChecked() == True:
            self.plot = self.ax[1].plot(self.all_data_dict["VICON_time"][participant][trial], self.all_data_dict["VICON_raw"][participant][trial][joint],'--', color='tab:cyan', label="VICON")
            self.plot = self.ax[1].plot(self.all_data_dict["PEACK_time"][participant][trial], self.all_data_dict["PEACK_raw"][participant][trial][joint],'--', color='tab:pink', label="PEACK")
        else:
            self.plot = self.ax[1].plot(self.all_data_dict["VICON_time"][participant][trial], self.all_data_dict["VICON_original"][participant][trial][joint],'--', color='tab:cyan', label="VICON")
            self.plot = self.ax[1].plot(self.all_data_dict["PEACK_time"][participant][trial]+lag, self.all_data_dict["PEACK_original"][participant][trial][joint],'--', color='tab:pink', label="PEACK")
        
        #plot timestamps
        for i in self.all_data_dict["VICON_stamps"][participant][trial][self.joint_select]: 
            self.plot = self.ax[0].plot(self.all_data_dict["VICON_time"][participant][trial][i], self.all_data_dict["VICON_filtered"][participant][trial][joint][i], 'bx') 
        for i in self.all_data_dict["PEACK_stamps"][participant][trial][self.joint_select]: 
                self.plot = self.ax[0].plot(self.all_data_dict["PEACK_time"][participant][trial][i]+lag, self.all_data_dict["PEACK_filtered"][participant][trial][joint][i], 'rx') 
                # self.plot = self.ax[0].vlines(self.all_data_dict["PEACK_time"][participant][trial][i]+lag, ymin = 0, ymax = 1, colors = 'r', linestyle = 'dashed') 



        if self.show_legend.isChecked() == True:
            self.ax[0].legend()
            self.ax[1].legend()
        
        self.canvas.draw()

        # os.system('cls' if os.name == 'nt' else 'clear') # clears terminal

        # if len(segment.peaks_and_troughs(self.all_data_dict["VICON_filtered"][participant][trial][joint],self.all_data_dict["VICON_time"][participant][trial])) == len(segment.peaks_and_troughs(self.all_data_dict["PEACK_filtered"][participant][trial][joint],self.all_data_dict["PEACK_time"][participant][trial])):
        #     for i in range(0,len(segment.peaks_and_troughs(self.all_data_dict["VICON_filtered"][participant][trial][joint],self.all_data_dict["VICON_time"][participant][trial]))):
        #         print(segment_correlation(self.all_data_dict["VICON_filtered"][participant][trial][joint],self.all_data_dict["PEACK_filtered"][participant][trial][joint],self.all_data_dict["VICON_time"][participant][trial],self.all_data_dict["PEACK_time"][participant][trial], i))
        # else:
        #     print("Uneven detected segments")
        

def segment_correlation(VICON,PEACK,VICON_time,PEACK_time, mov_segment, VICON_stamps,PEACK_stamps):
    # think about using the dict wrist stamps already as an input here instead of calculating the stamps anew! Saves computational power
    # and gets you the wanted results of only using the wrist data as an indicator!
    VICON_segment = VICON[VICON_stamps[mov_segment][0]:VICON_stamps[mov_segment][1]];
    PEACK_segment = PEACK[PEACK_stamps[mov_segment][0]:PEACK_stamps[mov_segment][1]];
    VICON_segment_time = VICON_time[VICON_stamps[mov_segment][0]:VICON_stamps[mov_segment][1]];
    PEACK_segment_time = PEACK_time[PEACK_stamps[mov_segment][0]:PEACK_stamps[mov_segment][1]];

    return [crosscorr(VICON_segment,PEACK_segment,VICON_segment_time,PEACK_segment_time),VICON_segment, VICON_segment_time, PEACK_segment, PEACK_segment_time]




if __name__ == '__main__':    
    app = QApplication(sys.argv)    
    win = visualization_win() 
    win.show() 
    sys.exit(app.exec())