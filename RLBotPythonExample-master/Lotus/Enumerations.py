# RLBOT classes and services
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import CarState
from rlbot.utils.game_state_util import Physics
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.game_state_util import Rotator
from rlbot.utils.game_state_util import BallState


class CarStateEnumeration:
    def __init__(self):
        #define all different car states here
        self.carStateBasic = CarState(boost_amount=100,
                         physics=Physics(location = Vector3(0, 0, 17.01),velocity=Vector3(0, 0, 0), rotation = Rotator(pitch = 0, yaw = 0, roll = 0), angular_velocity = Vector3(0,0,0)))



    def getCarState(self, enum):
        None
        #return an enumeration car state

class BallStateEnumeration:
    def __init__(self):
        #define all different ball states here
        self.ballStateBasic = BallState(physics=Physics(location = Vector3(0, 0, 92.75),velocity=Vector3(0, 0, 0), angular_velocity = Vector3(0,0,0)))
        self.ballStateHigh = BallState(physics=Physics(location = Vector3(0, 0, 900),velocity=Vector3(0, 0, 0), angular_velocity = Vector3(0,0,0)))

    def getBallState(self, enum):
        None
        #return an enumeration ball state

    def createBallState(self):
        None
