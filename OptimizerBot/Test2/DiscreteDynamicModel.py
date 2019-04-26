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
from TrueProportionalNavigation import TPN
import controller as con
import State as s
import DrivingEquations
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
import control #Control system python library
import numpy as np
#import slycot

#imports for LQR functions
# from __future__ import division, print_function
import numpy as np
import scipy.linalg

import traceback


# Discrete time car prediction model for MPC optimization function
class ModelState():
    def __init__(self, car, control_variables, coordinate_system, time):
        # Car _packet should be the game_car packet not the entire packet for the game

        # Coordinate system instance at this time
        self.coordinate_system = coordinate_system

        self.velocity_car = np.array([0,0,0])
        self.acceleration_car = np.array([control_variables.boost*991.666, 0, 0])
        self.angular_velocity_car = car.angular_velocity
        self.torques_car = control_variables.torques

        # Values in world reference coordinate system
        self.position_world = car.position
        self.velocity_world = car.velocity

        # Convert local acceleration to world acceleration on car
        self.acceleration_world = coordinate_system.toWorldCoordinates(self.acceleration_car)

        # Other car parameters like wheel contact, super_sonic, jumped, double_jumped etc will be helpful for knowing which parameters affect the car during optimization
        self.is_demolished = car.is_demolished
        self.has_wheel_contact = car.has_wheel_contact
        self.is_super_sonic = car.is_super_sonic
        self.jumped = car.jumped
        self.boost_left = car.boost_left
        self.double_jumped = car.double_jumped

        # Control inputs (essenetially the state of the joystick controller)
        self.control_variables = control_variables

        self.time = time

    def update(self, car, control_variables, coordinate_system, time):
        # Car _packet should be the game_car packet not the entire packet for the game

        # Coordinate system instance at this time
        self.coordinate_system = coordinate_system

        self.velocity_car = np.array([0,0,0])
        self.acceleration_car = np.array([control_variables.boost*991.666, 0, 0])
        self.angular_velocity_car = car.angular_velocity
        self.torques_car = control_variables.torques

        # Values in world reference coordinate system
        self.position_world = car.position
        self.velocity_world = car.velocity

        # Convert local acceleration to world acceleration on car
        self.acceleration_world = coordinate_system.toWorldCoordinates(self.acceleration_car)

        # Other car parameters like wheel contact, super_sonic, jumped, double_jumped etc will be helpful for knowing which parameters affect the car during optimization
        self.is_demolished = car.is_demolished
        self.has_wheel_contact = car.has_wheel_contact
        self.is_super_sonic = car.is_super_sonic
        self.jumped = car.jumped
        self.boost_left = car.boost_left
        self.double_jumped = car.double_jumped

        # Control inputs (essenetially the state of the joystick controller)
        self.control_variables = control_variables

        self.time = time

class ControlVariables():
    def __init__(self, controller_state_packet):
        self.boost = int(controller_state_packet.boost)
        self.torques = np.array([controller_state_packet.roll, controller_state_packet.pitch, controller_state_packet.yaw])
        self.steer = controller_state_packet.steer
        self.throttle = controller_state_packet.throttle #Throttle includes braking, optimization will have to have non-symmetric contraint on this variable
        self.handbrake = controller_state_packet.handbrake
        self.jump = controller_state_packet.jump

    def update(self, controller_state_packet):
        self.boost = int(controller_state_packet.boost)
        self.torques = np.array([controller_state_packet.roll, controller_state_packet.pitch, controller_state_packet.yaw])
        self.steer = controller_state_packet.steer
        self.throttle = controller_state_packet.throttle #Throttle includes braking, optimization will have to have non-symmetric contraint on this variable
        self.handbrake = controller_state_packet.handbrake
        self.jump = controller_state_packet.jump

def get_future_state(state, dt):
    # In the air
    if(state.has_wheel_contact == False):
        g = np.array([0,0,-650])

        a = state.acceleration_world

        v1 = (a + g)*dt + state.velocity_world

        d1 = (a+g)*(dt**2)

        p1 = ((d1) / 2) + (state.velocity_world*dt) + state.position_world

        return p1, v1
