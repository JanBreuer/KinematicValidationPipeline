from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, Qt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import *
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
import variableDeclaration as var



class select_joints_win(QtWidgets.QDialog):
    submitClicked = qtc.pyqtSignal(list)  # <-- This is the sub window's signal

    def __init__(self, parent=None):
        super(select_joints_win, self).__init__(parent)
        self.layout = QtWidgets.QVBoxLayout()
        self.listWidget = QtWidgets.QListWidget()
        self.listWidget.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.listWidget.setGeometry(QtCore.QRect(10, 10, 211, 291))
        t = ["RShoulder","RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "Neck", "Midpoint"]
        for i in t:
            item = QtWidgets.QListWidgetItem(i)
            self.listWidget.addItem(item)
        self.listWidget.itemClicked.connect(self.printItemText)
        self.layout.addWidget(self.listWidget)
        self.setLayout(self.layout)

        self.select_joints_button = QPushButton('Confirm selection')
        self.select_joints_button.clicked.connect(self.confirm)
        self.select_joints_button.setEnabled(False)
        self.layout.addWidget(self.select_joints_button)

    def printItemText(self):
        items = self.listWidget.selectedItems()
        self.x = []
        for i in range(len(items)):
            self.x.append(str(self.listWidget.selectedItems()[i].text()))
        if var.joint_selector in self.x:
            self.select_joints_button.setEnabled(True)
        else:
            self.select_joints_button.setEnabled(False)

            

    def pass_selected_joints(self):
        print("")

    def confirm(self):  # <-- Here, the signal is emitted *along with the data we want*
        self.submitClicked.emit(self.x)
        self.close()


if __name__ == '__main__':    
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = select_joints_win() 
    win.show() 
    sys.exit(app.exec())