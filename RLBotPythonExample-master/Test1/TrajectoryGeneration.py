  #imports for LQR functions
from __future__ import division, print_function

import math
from CoordinateSystems import CoordinateSystems
from BallController import BallController
from pyquaternion import Quaternion
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

import random
import control #Control system python library
import numpy as np
#import slycot

#imports for LQR functions
# from __future__ import division, print_function
import numpy as np
import scipy.linalg

class Trajectory:
    def __init__(self):
        self.counter = 0
        self.position = None
        self.trajectoryType = None

    def circularTrajectory(self, radius, omega, time, height):
        w = omega #angular velocity of trajectory
        A = radius
        t = time
        #positions
        z = height
        x = A*math.cos(w*t)
        y = A*math.sin(w*t)
        #velocities
        vx = -1*A*w*math.sin(w*t)
        vy = A*w*math.cos(w*t)
        vz = 0
        return np.array([x,y,z]), np.array([vx,vy,vz])

    def startTrajectory(self, type):
        self.counter = 0
        if(type == 'circular'):
            self.trajectoryType = type
    def resetTrajectory(self):
        self.counter = 0
    def progress(self):
        self.counter = self.counter + 1
