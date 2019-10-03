import math
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState

class Controller():

    def __init__(self):
        self.on = False # Turn the controller on or off with this variable. Essentially this allows the controller to set inputs on the joypad
        self.t0 = 0.0 # The initial game time that the trajectory is started
        self.t_now = 0.0 # The running realtime of the game to follow the trajectory

        # Trajectory Data Types
        self.ts = []
        self.sx = []
        self.sy = []
        self.vx = []
        self.vy = []
        self.yaw = []
        self.omega = []

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
            self.joypadState.steer = round(self.turning[idx] - 1.0 , 4)
            self.joypadState.boost = int(self.a[idx])
            return self.joypadState
        else:
            return SimpleControllerState()

    def setTrajectoryData(self, ts, sx, sy, vx, vy, yaw, omega):
        self.ts = ts
        self.sx = sx
        self.sy = sy
        self.vx = vx
        self.vy = vy
        self.yaw = yaw
        self.omega = omega

    def setInputData(self, a, turning):
        self.a = a
        self.turning = turning

    def setTNOW(self, t):
        self.t_now = t
