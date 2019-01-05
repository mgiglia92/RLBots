  #imports for LQR functions
from __future__ import division, print_function

import math
from pyquaternion import Quaternion

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
import scipy.linalg

#penis suck ma balls
class BallController:
    def __init__(self):
        self.location = None
        self.velocity = None
        self.counter1 = 0
        self.release = 0
    def update(self, data):
        self.location = data.physics.location
        self.x = data.physics.location.x
        self.y = data.physics.location.y
        self.z = data.physics.location.z
        self.vx = data.physics.velocity.x
        self.vy = data.physics.velocity.y
        self.vz = data.physics.velocity.z
    def rotateAboutZ(self, center, angVel):
        #center point of rotation of ball
        w = np.matrix([0,0,angVel])
        xc = center.item(0)
        yc = center.item(1)
        zc = center.item(2)
        xb = self.location.x
        yb = self.location.y
        zb = self.location.z
        Pb = np.matrix([self.location.x, self.location.y, 0]) #point of the ball
        Pc = np.matrix([xc, yc, 0]) #point of the center
        Pbc = np.subtract(Pb, Pc)
        #get desired tangential velocity
        Vt = np.cross(w, Pbc)
        return Vt
    def oscillateX(self, p1, p2, vel):
        if(self.location.x <= p1):
            return vel
        if(self.location.x >= p2):
            return -1*vel
        else:
            return self.vx

    def bounce(self, x, y, z, vi):
        self.counter1 = self.counter1 + 1
        if(self.counter1> 50):
            self.release = 1
            return x, y, z, vi
        else:
            return x, y, z, vi
