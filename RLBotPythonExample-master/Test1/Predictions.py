  #imports for LQR functions
from __future__ import division, print_function

import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import Plotting
from CoordinateSystems import CoordinateSystems
from BallController import BallController
from pyquaternion import Quaternion
from TrajectoryGeneration import Trajectory


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


def predict(position, velocity, q, omega, a, torques, t0, t1):
    # print("p:", position, "v:", velocity, "q:", q, "omega:", omega, "a:", a, "torques:", torques)#, "t0:", t0, "t1:", t1)
    g = np.array([0,0,-650])
    dt = t1 - t0
    v1 = (a + g)*dt + velocity
    d1 = (a+g)*dt
    p1 = ((d1**2) / 2) + (velocity*dt) + position

    return p1, v1
