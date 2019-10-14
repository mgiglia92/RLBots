import math
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import CarState

class Controller():

    def __init__(self):
        self.on = False # Turn the controller on or off with this variable. Essentially this allows the controller to set inputs on the joypad
        self.t0 = 0.0 # The initial game time that the trajectory is started
        self.t_now = 0.0 # The running realtime of the game to follow the trajectory

        # current car state
        self.currentState = CarState()

        # Trajectory Data Types
        self.ts = []
        self.sx = []
        self.sy = []
        self.vx = []
        self.vy = []
        self.yaw = []
        self.omega = []
        self.curvature = []

        # Input Data Types
        self.a = []
        self.turning = []
        self.joypadState = SimpleControllerState() # Updating joypad state to read in get_output function for base agent

    def openLoop(self):
        if(self.on == True):
            # Find the correct input data to use based on time
            deltaT = self.t_now - self.t0
            idx = 0

            for i in range(0, len(self.ts) - 1):
                idx = i
                if(idx == len(self.ts)):
                    break
                if(deltaT < self.ts[idx+1]):
                    break
            # idx = np.searchsorted(self.ts, deltaT, side="left")
            # if (idx > 0) and (idx == len(self.ts) or math.fabs(t - self.ts[idx-1]) < math.fabs(t - self.ts[idx])):
            #     index = idx-1
            # else:
            #     index = idx

            # Stop trajectory
            if(deltaT > self.ts[-1]):
                self.on = False

            print("index: ", idx)
            self.joypadState.steer = self.getTurning(self.vx[idx], self.vy[idx], self.curvature[idx])
            # self.joypadState.steer = self.turning[idx]

            if(self.a[idx] > 0.9):
                self.joypadState.boost = 1
                self.joypadState.throttle = 1
            else:
                self.joypadState.boost = 0
                self.joypadState.throttle = 0.03

            return self.joypadState
        else:
            return SimpleControllerState()

    def getTurning(self, vx, vy, curvature):
        #getting the actual steering value since optmizer uses a polynomial approximation
        curvature_max = np.array([0.0069, 0.00398, 0.00235, 0.001375, 0.0011, 0.00088])
        v_for_steer = np.array([0, 500, 1000, 1500, 1750, 2300])
        velocity = np.linalg.norm(np.array([vx, vy]))
        curvature_current = np.interp(velocity, v_for_steer, curvature_max)

        c = curvature / curvature_current
        print('c: ', c)
        return -1*c

    def getHeadingError(self, yaw_ref):
        # get unit veclotiy vector
        heading_ref = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0])

        yaw = self.currentState.physics.rotation.yaw
        heading_current = np.array([np.cos(yaw), np.sin(yaw), 0])

        heading_error = np.cross(heading_ref, heading_current)

        heading_mag = np.linalg.norm(heading_ref)
        print("heading ref: ", heading_ref)
        print("heading mag: ", heading_mag)
        print("heading error: ", heading_error)
        return heading_error[2]

    def getVelocityError(self, vx, vy):
        v_mag_ref = np.sqrt((vx*vx) + (vy*vy))
        vx_cur = self.currentState.physics.velocity.x
        vy_cur = self.currentState.physics.velocity.y
        v_mag_cur = np.sqrt((vx_cur * vx_cur) + (vy_cur * vy_cur))
        v_error = v_mag_ref - v_mag_cur
        print("v_error: ", v_error)
        return v_error

    def getLateralError(self, traj_idx):
        # Reference position of trajectory
        x_ref = self.sx[traj_idx]
        y_ref = self.sy[traj_idx]
        pos_ref = np.array([x_ref, y_ref, 0])

        # current car position
        x_cur = self.currentState.physics.location.x
        y_cur = self.currentState.physics.location.y
        pos_cur = np.array([x_cur, y_cur, 0])

        # delta phi (delta yaw)
        vx_ref = self.vx[traj_idx]
        vy_ref = self.vy[traj_idx]
        v_ref = np.array([vx_ref, vy_ref, 0])
        v_ref_mag = np.linalg.norm(v_ref)
        vx_cur = self.currentState.physics.velocity.x
        vy_cur = self.currentState.physics.velocity.y
        v_cur = np.array([vx_cur, vy_cur, 0])
        v_cur_mag = np.linalg.norm(v_cur)
        phi = np.arcsin((np.cross(v_ref, v_cur)/(v_ref_mag * v_cur_mag))[2]) # Getting heading error angle

        # error vector
        if(v_ref_mag != 0):
            v_ref_unit = np.array([vx_ref/v_ref_mag, vy_ref/v_ref_mag, 0]) # Velocity reference unit to find lateral component of error
        else:
            v_ref_unit = np.array([0, 0, 0])
        error = np.array([pos_cur - pos_ref]) # total error vector
        lat_error = np.linalg.norm(np.subtract(error, (v_ref_unit * np.dot(error, v_ref_unit)))) # Getting lateral componnent of error scalar value

        # Calculate lateral error sign
        lat_error_sign = np.sign(np.cross(pos_cur, pos_ref)[2]) # Getting lateral error sign
        print("phi: ", phi)
        print("lateral e: ", lat_error)
        return (lat_error * lat_error_sign) + (phi * v_cur_mag) # Lateral error plus lookahead (using v_cur_mag as a variable lookahead value it gets larger as your velocity gets larger this may help)


    def feedFoward(self):
        None

    def feedBack(self):
        # Determine which index we are at in trajectory
        if(self.on == True):
            # Find the correct input data to use based on time
            deltaT = self.t_now - self.t0
            idx = 0

            for i in range(0, len(self.ts) - 1):
                idx = i
                if(idx == len(self.ts)):
                    break
                if(deltaT < self.ts[idx+1]):
                    break
            # idx = np.searchsorted(self.ts, deltaT, side="left")
            # if (idx > 0) and (idx == len(self.ts) or math.fabs(t - self.ts[idx-1]) < math.fabs(t - self.ts[idx])):
            #     index = idx-1
            # else:
            #     index = idx

            # Stop trajectory
            if(deltaT > self.ts[-1]):
                self.on = False

            # Totally Band Bang control attempt
            # Feedback gains
            ks = -0.5 # heading error feedback gain
            ke = -0.001 # lateral error feedback gain
            ka = 1 # Acceleratoin feedback gain

            # Set steering and boost
            steer = (np.clip((ks * self.getHeadingError(self.yaw[idx])) + (self.getLateralError(idx) * ke), -1, 1))
            boost = np.clip(np.sign(self.getVelocityError(self.vx[idx], self.vy[idx])), 0, 1)

            print("steer: ", steer)
            print("boost: ", boost)

            #Set joypad values
            self.joypadState.steer = float(steer)
            self.joypadState.boost = float(boost)

            return self.joypadState
        else:
            return SimpleControllerState()

    def setTrajectoryData(self, ts, sx, sy, vx, vy, yaw, omega, curvature):
        self.ts = ts
        self.sx = sx
        self.sy = sy
        self.vx = vx
        self.vy = vy
        self.yaw = yaw
        self.omega = omega
        self.curvature = curvature

    def setInputData(self, a, turning):
        self.a = a
        self.turning = turning

    def setTNOW(self, t):
        self.t_now = t

    def setCurrentState(self, cs):
        self.currentState = cs
