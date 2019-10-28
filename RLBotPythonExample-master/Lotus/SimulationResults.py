  #imports for LQR functions
from __future__ import division, print_function
import csv
import sys
import os


import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import Plotting
import Predictions
from CoordinateSystems import CoordinateSystems
# from BallController import BallController
from pyquaternion import Quaternion
# from TrueProportionalNavigation import TPN
from Controller import Controller
# import State as s
# import DrivingEquations
# import DiscreteDynamicModel as ddm
import Optimization2
from Trajectory import Trajectory
# from Car import Car
from GUI import GUI
from tkinter import Tk, Label, Button, StringVar, Entry, Listbox

# import tensorflow
# tf.merge_all_summaries = tf.summary.merge_all
# tf.train.SummaryWriter = tf.summary.FileWriter
# import tf
print("imported math")

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import CarState
from rlbot.utils.game_state_util import Physics
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.game_state_util import Rotator
from rlbot.utils.game_state_util import BallState

from enum import Enum
import random
#import slycot

#imports for LQR functions
# from __future__ import division, print_function
import numpy as np
import scipy.linalg
import time

import traceback
import copy
import queue

import queue_testing
from EnvironmentManipulator import EnvironmentManipulator

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimulationResults:
    def __init__(self):
        #Trajectory data
        self.ts = []
        self.sx = []
        self.sy = []
        self.vx = []
        self.vy = []
        self.yaw = []
        self.omega = []
        self.curvature = []

        # State data from game (actual state, thats whats the _a stands for)
        self.t_now = []
        self.sx_a = []
        self.sy_a = []
        self.vx_a = []
        self.vy_a = []
        self.yaw_a = []

        # Input data from realtime controller
        self.controller_state = []

    def setTrajectoryData(self):
        None

    def addControlData(self, t_now, state, controller_state):
        self.t_now.append(t_now)
        self.sx_a.append(state.location.x)
        self.sy_a.append(state.location.y)
        self.vx_a.append(state.velocity.x)
        self.vy_a.append(state.velocity.y)
        self.yaw_a.append(state.rotation.yaw)
        self.controller_state.append(controller_state)
        # self.printData()

    def plotData(self):
        None
    def printData(self):
        print("tNOW:", self.t_now)
        # print("car_state: ", self.car_state)
        # print("controller_state: ", self.controller_state)

    def setTrajectoryData(self, ts, sx, sy, vx, vy, yaw, omega, curvature):
        self.ts = ts
        self.sx = sx
        self.sy = sy
        self.vx = vx
        self.vy = vy
        self.yaw = yaw
        self.omega = omega
        self.curvature = curvature

    def clear(self):
        self.__init__() # Re-initialize data
