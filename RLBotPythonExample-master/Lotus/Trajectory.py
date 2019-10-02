  #imports for LQR functions
from __future__ import division, print_function

import copy
import math
from CoordinateSystems import CoordinateSystems
from BallController import BallController
from pyquaternion import Quaternion
from TrajectoryPlanning import TrajectoryGenerator
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
from Car import Car
        # self.double_jumped = data.double_jumped

class Trajectory:
    def __init__(self):
        self.t = None # Time vector for trajectory
        self.sx = None # Position vector for trajectory
        self.sy = None
        self.vx = None # Velocity vector for trajectory
        self.vy = None
        self.TG = TrajectoryGenerator()


    def get_new_trajectory(self):
        self.sx, self.sy, self.vx, self.vy, self.yaw = self.TG.generateDrivingTrajectory(initial_car_state, final_car_state)

    def get_simple_test_trajectory(self, packet, idx):
        # Set a simple final car state for trajectory generation
        self.TG.final_car_state = copy.deepcopy(self.TG.initial_car_state)
        self.TG.final_car_state.x = 0
        self.TG.final_car_state.y = 0

        # Generate and save simple driving trajectory
        self.sx, self.sy, self.vx, self.vy, self.yaw = self.TG.generateDrivingTrajectory(self.TG.initial_car_state, self.TG.final_car_state)

        # print stuff
        print('sx', self.sx.value)
        print('sy', self.sy.value)
