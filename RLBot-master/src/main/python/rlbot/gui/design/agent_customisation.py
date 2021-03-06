# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'agent_customisation.ui'
#
# Created by: PyQt5 UI code generator 5.11
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_AgentPresetCustomiser(object):
    def setupUi(self, AgentPresetCustomiser):
        AgentPresetCustomiser.setObjectName("AgentPresetCustomiser")
        AgentPresetCustomiser.resize(484, 313)
        self.centralwidget = QtWidgets.QWidget(AgentPresetCustomiser)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.presets_listwidget = QtWidgets.QListWidget(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.presets_listwidget.sizePolicy().hasHeightForWidth())
        self.presets_listwidget.setSizePolicy(sizePolicy)
        self.presets_listwidget.setMinimumSize(QtCore.QSize(150, 0))
        self.presets_listwidget.setObjectName("presets_listwidget")
        self.verticalLayout.addWidget(self.presets_listwidget)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.preset_new_pushbutton = QtWidgets.QPushButton(self.groupBox_3)
        self.preset_new_pushbutton.setObjectName("preset_new_pushbutton")
        self.horizontalLayout_2.addWidget(self.preset_new_pushbutton)
        self.preset_load_pushbutton = QtWidgets.QPushButton(self.groupBox_3)
        self.preset_load_pushbutton.setObjectName("preset_load_pushbutton")
        self.horizontalLayout_2.addWidget(self.preset_load_pushbutton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addWidget(self.groupBox_3)
        self.right_frame = QtWidgets.QFrame(self.centralwidget)
        self.right_frame.setObjectName("right_frame")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.right_frame)
        self.verticalLayout_4.setContentsMargins(-1, -1, -1, 1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.preset_config_groupbox = QtWidgets.QGroupBox(self.right_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.preset_config_groupbox.sizePolicy().hasHeightForWidth())
        self.preset_config_groupbox.setSizePolicy(sizePolicy)
        self.preset_config_groupbox.setObjectName("preset_config_groupbox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.preset_config_groupbox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_2 = QtWidgets.QLabel(self.preset_config_groupbox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.preset_config_groupbox)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 6, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.preset_config_groupbox)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 3, 0, 1, 1)
        self.preset_save_pushbutton = QtWidgets.QPushButton(self.preset_config_groupbox)
        self.preset_save_pushbutton.setObjectName("preset_save_pushbutton")
        self.gridLayout_2.addWidget(self.preset_save_pushbutton, 8, 2, 1, 1)
        self.preset_autosave_checkbox = QtWidgets.QCheckBox(self.preset_config_groupbox)
        self.preset_autosave_checkbox.setObjectName("preset_autosave_checkbox")
        self.gridLayout_2.addWidget(self.preset_autosave_checkbox, 8, 3, 1, 1, QtCore.Qt.AlignHCenter)
        self.python_file_select_button = QtWidgets.QPushButton(self.preset_config_groupbox)
        self.python_file_select_button.setObjectName("python_file_select_button")
        self.gridLayout_2.addWidget(self.python_file_select_button, 8, 0, 1, 2)
        self.preset_python_file_lineedit = QtWidgets.QLineEdit(self.preset_config_groupbox)
        self.preset_python_file_lineedit.setReadOnly(True)
        self.preset_python_file_lineedit.setClearButtonEnabled(False)
        self.preset_python_file_lineedit.setObjectName("preset_python_file_lineedit")
        self.gridLayout_2.addWidget(self.preset_python_file_lineedit, 6, 1, 1, 3)
        self.preset_path_lineedit = QtWidgets.QLineEdit(self.preset_config_groupbox)
        self.preset_path_lineedit.setReadOnly(True)
        self.preset_path_lineedit.setObjectName("preset_path_lineedit")
        self.gridLayout_2.addWidget(self.preset_path_lineedit, 3, 1, 1, 3)
        self.preset_name_lineedit = QtWidgets.QLineEdit(self.preset_config_groupbox)
        self.preset_name_lineedit.setObjectName("preset_name_lineedit")
        self.gridLayout_2.addWidget(self.preset_name_lineedit, 0, 1, 1, 3)
        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setColumnStretch(2, 2)
        self.gridLayout_2.setColumnStretch(3, 2)
        self.verticalLayout_4.addWidget(self.preset_config_groupbox)
        self.agent_parameters_groupbox = QtWidgets.QGroupBox(self.right_frame)
        self.agent_parameters_groupbox.setObjectName("agent_parameters_groupbox")
        self.verticalLayout_4.addWidget(self.agent_parameters_groupbox)
        self.verticalLayout_4.setStretch(1, 1)
        self.horizontalLayout.addWidget(self.right_frame)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 2)
        AgentPresetCustomiser.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(AgentPresetCustomiser)
        self.statusbar.setObjectName("statusbar")
        AgentPresetCustomiser.setStatusBar(self.statusbar)

        self.retranslateUi(AgentPresetCustomiser)
        QtCore.QMetaObject.connectSlotsByName(AgentPresetCustomiser)

    def retranslateUi(self, AgentPresetCustomiser):
        _translate = QtCore.QCoreApplication.translate
        AgentPresetCustomiser.setWindowTitle(_translate("AgentPresetCustomiser", "Agent Preset Customiser"))
        self.groupBox_3.setTitle(_translate("AgentPresetCustomiser", "Agent Presets"))
        self.preset_new_pushbutton.setText(_translate("AgentPresetCustomiser", "New"))
        self.preset_load_pushbutton.setText(_translate("AgentPresetCustomiser", "Load"))
        self.preset_config_groupbox.setTitle(_translate("AgentPresetCustomiser", "Preset Config"))
        self.label_2.setText(_translate("AgentPresetCustomiser", "Preset Name:"))
        self.label.setText(_translate("AgentPresetCustomiser", "Python File:"))
        self.label_17.setText(_translate("AgentPresetCustomiser", "File Path:"))
        self.preset_save_pushbutton.setText(_translate("AgentPresetCustomiser", "Save Preset"))
        self.preset_autosave_checkbox.setText(_translate("AgentPresetCustomiser", "Autosave"))
        self.python_file_select_button.setText(_translate("AgentPresetCustomiser", "Select Agent File"))
        self.preset_path_lineedit.setPlaceholderText(_translate("AgentPresetCustomiser", "Currently not stored on disk"))
        self.agent_parameters_groupbox.setTitle(_translate("AgentPresetCustomiser", "Agent Parameters"))

