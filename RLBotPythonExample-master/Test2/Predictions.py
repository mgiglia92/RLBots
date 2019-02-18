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
    d1 = (a+g)*(dt**2)
    p1 = ((d1) / 2) + (velocity*dt) + position

    return p1, v1

def predictBallTrajectory(ball, tnow):
    s = ball.position
    v = ball.velocity
    g = np.array([0,0,-650]).transpose()
    t = np.linspace(0, 2, num = 10).transpose()
    vplus = (np.outer(g,t).transpose() + v).transpose()
    splus = ((np.outer(g, np.power(t,2)).transpose() / 2) + np.outer(v, t).transpose() + s).transpose()

    print('s', s, 'splus', splus)
    return splus

def ballPredictionError(ball_before, ball_now, tbefore, tnow):
    s_before = ball_before.position
    v_before = ball_before.velocity
    g = np.array([0,0,-650])
    dt = tnow - tbefore -0.03 #???????????is there some sort of timing issue with the function? is it behind or something?
    v_predict = g*dt + v_before
    s_predict = (g*(dt**2) / 2) + v_predict*dt + s_before
    square_error = (s_predict - ball_now.position)**2
    return square_error
