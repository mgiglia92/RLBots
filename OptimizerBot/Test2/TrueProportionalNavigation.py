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
        self.LOS = np.array([0,0,0])
        self.LOS0 = np.array([0,0,0])
        self.LOS1 = np.array([0,0,0])
        self.LOSunit = np.array([0,0,0])
        self.LOS_rotation_rate = np.array([0,0,0])
        self.Vb_parallel = np.array([0,0,0])
        self.Vc_parallel = np.array([0,0,0])
        self.Vb_perpendicular = np.array([0,0,0])
        self.Vc_perpendicular = np.array([0,0,0])
        self.navigationConstant = 5
        self.t0 = 0
        self.t1 = 1

    def update(self, car, ball, time):
        self.car = car
        self.ball = ball
        #get times
        self.t0 = self.t1
        self.t1 = time

        #LOS at times and rate using time
        self.LOS0 = self.LOS1
        self.LOS1 = ball.position - car.position
        self.LOS_rate1 = (self.LOS1 - self.LOS0) / (self.t1 - self.t0) #rate of change of lOS
        self.LOS_rate_mag1 = np.linalg.norm(self.LOS_rate1)

        self.LOSunit = self.LOS1 / np.linalg.norm(self.LOS1)

    def getUnitVectors(self):
        #Get LOS unit vector
        self.LOSunit = self.LOS1 /  np.linalg.norm(self.LOS1)
        return self.LOS1, self.LOSunit

    def getNavigationValues(self):
        #Get the rate of change of the LOS using Vparallel for car and ball
        self.Vb_parallel = np.dot(self.ball.velocity, self.LOSunit)
        self.Vc_parallel = np.dot(self.car.velocity, self.LOSunit)
        self.Vb_parallel_mag = np.linalg.norm(self.Vb_parallel)
        self.Vc_parallel = np.linalg.norm(self.Vc_parallel)

        self.LOSrate2 = self.Vb_parallel - self.Vc_parallel
        self.LOSrate_unit2 = self.LOSrate2 / np.linalg.norm(self.LOSrate2)
        LOS_sign = np.sign(self.LOSrate_unit2)
        self.LOS_rate_mag2 = np.linalg.norm(self.LOSrate2) * LOS_sign
        # self.closingRate = -1*self.LOSrate_mag
        self.Vc =  -1 * self.LOS_rate_mag2
        #Get the LOS rotation rotate
        self.Vb_perpendicular = self.ball.velocity - self.Vb_parallel
        self.Vc_perpendicular = (self.car.velocity - self.Vc_parallel)

        self.LOS_rotation_rate = (self.Vb_perpendicular - self.Vc_perpendicular) / np.linalg.norm(self.LOS1)

        #Nt which is gravity acceleration normal to LOS
        gravity = np.array([0,0,-650])
        self.Nt = gravity - np.dot(gravity, self.LOSunit)


        if(self.LOS1.any() == 0):
            self.omega = np.array([0,0,0])
        else:
            self.omega = np.cross(self.LOS1, (self.ball.velocity - self.car.velocity)) / np.dot(self.LOS1, self.LOS1)
        self.omega_unit = self.omega / np.linalg.norm(self.omega)
        self.omega_mag = np.linalg.norm(self.omega)
        self.Acc_direction = np.cross(self.omega_unit, self.LOSunit)
        self.accelerationVector = (self.navigationConstant * self.omega_mag* self.Vc) * self.Acc_direction

        acc_unit = self.accelerationVector / np.linalg.norm(self.accelerationVector)
        self.finalAccelerationVector = self.accelerationVector #+ acc_unit * ((self.navigationConstant * self.Nt) / 2)

        inner = np.dot(self.LOS1, self.accelerationVector)
        losmag = np.linalg.norm(self.LOS1)
        accmag = np.linalg.norm(self.accelerationVector)
        accAngle = np.arccos(inner / (accmag * losmag)) * 180 / math.pi
        # self.accelerationVector = np.array([self.accelerationVector.item(0), self.accelerationVector.item(1), -1*self.accelerationVector.item(2)])
        # print (accAngle)
        #fixing acceleration vector directions for some reason
        self.accelerationVector = np.array([self.accelerationVector[0],self.accelerationVector[1],self.accelerationVector[2]])
        return self.Vb_parallel, self.Vc_parallel, self.Vb_perpendicular, self.Vc_perpendicular, self.accelerationVector
