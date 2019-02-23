import math
import numpy as np

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import CarState
from rlbot.utils.game_state_util import Physics
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.game_state_util import Rotator
from rlbot.utils.game_state_util import BallState

class Controller():

    def __init__(self):
        self.boostPercent = 0.0
        self.torques = np.array([0.0, 0.0, 0.0])
        self.jump = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.turn = 0.0
