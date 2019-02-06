  #imports for LQR functions
from __future__ import division, print_function


import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import Plotting
import Predictions
from CoordinateSystems import CoordinateSystems
from BallController import BallController
from pyquaternion import Quaternion
from TrajectoryGeneration import Trajectory
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


class TPN:
    def __init__(self):
        None
    def update(self, car, ball):
        self.car = car
        self.ball = ball
        self.LOS = np.array([0,0,0])
        self.LOSunit = np.array([0,0,0])
        self.Vb_parallel = np.array([0,0,0])
        self.Vc_parallel = np.array([0,0,0])
        self.Vb_perpendicular = np.array([0,0,0])
        self.Vc_perpendicular = np.array([0,0,0])
        self.navigationConstant = 2
        a, b = self.getUnitVectors()

    def getUnitVectors(self):
        #Get LOS unit vector
        self.LOS = self.ball.position - self.car.position
        self.LOSunit = self.LOS / np.linalg.norm(self.LOS)
        return self.LOS, self.LOSunit

    def getNavigationValues(self):
        #Get the rate of change of the LOS using Vparallel for car and ball
        self.Vb_parallel_mag = np.dot(self.ball.velocity, self.LOSunit)
        self.Vc_parallel_mag = np.dot(self.car.velocity, self.LOSunit)
        self.Vb_parallel = self.LOSunit * self.Vb_parallel_mag
        self.Vc_parallel = self.LOSunit * self.Vc_parallel_mag

        self.LOSrate = self.Vb_parallel - self.Vc_parallel
        self.LOSrate_mag = np.linalg.norm(self.LOSrate)
        self.closingRate = -1*self.LOSrate_mag
        #Get the LOS rotation rotate
        self.Vb_perpendicular = self.ball.velocity - self.Vb_parallel
        self.Vc_perpendicular = (self.car.velocity - self.Vc_parallel)

        self.LOS_rotation_rate = (self.Vb_perpendicular + self.Vc_perpendicular) / np.linalg.norm(self.LOS)

        self.accelerationVector = (self.navigationConstant * self.LOS_rotation_rate * self.closingRate)
        # self.accelerationVector = np.array([self.accelerationVector.item(0), self.accelerationVector.item(1), -1*self.accelerationVector.item(2)])
        print (self.Vb_parallel_mag, self.Vc_parallel_mag, self.LOSrate)
        return self.Vb_parallel, self.Vc_parallel, self.Vb_perpendicular, self.Vc_perpendicular, self.accelerationVector
