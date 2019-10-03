import numpy as np
import math

# RLBOT classes and services
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import CarState
from rlbot.utils.game_state_util import Physics
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.game_state_util import Rotator
from rlbot.utils.game_state_util import BallState


class EnvironmentManipulator:
    def __init__(self):
        self.ball_initial_state = BallState()
        self.car_initial_state = CarState()
        self.start_trajectory = False

    def getInitialStates(self):
        return self.car_initial_state, self.ball_initial_state

    def setValue(self, val):
        self.value = val

    def setInitialStates(self, carState, ballState):
        self.ball_initial_state = ballState
        self.car_initial_state = carState

    def startTrajectory(self):
        self.start_trajectory = True
