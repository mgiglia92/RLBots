  #imports for LQR functions
from __future__ import division, print_function

import math
from CoordinateSystems import CoordinateSystems
from BallController import BallController
from pyquaternion import Quaternion
from TrajectoryGeneration import Trajectory
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

class Test1(BaseAgent):

    def initialize_agent(self):
        #This runs once before the bot starts up
        self.controller_state = SimpleControllerState()
        self.resetFlag = 0;
        self.setCarState()
        self.car = Car()
        self.fbController = FeedbackController2()
        self.orientationCounter = orientationCounter()
        self.boostCounter = BoostCounter()
        self.boostCounter2 = BoostCounter2()
        self.counter = TestCounter(50, 100)
        self.counter2 = TestCounter(50, 100)
        self.ball = Ball()
        self.BallController = BallController()
        self.CoordinateSystems = CoordinateSystems()
        self.Trajectory = Trajectory()
        self.Trajectory.startTrajectory('circular')

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        #Original Code
        my_car = packet.game_cars[self.index]
        ball_location = Vector2(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y)
        car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
        car_direction = get_car_facing_vector(my_car)
        car_to_ball = ball_location - car_location


        #update class data
        self.car.update(packet.game_cars[self.index])
        self.ball.update(packet.game_ball)
        self.CoordinateSystems.update(self.car, self.ball)
        self.BallController.update(packet.game_ball)

        self.ball.x = packet.game_ball.physics.location.x
        self.ball.y = packet.game_ball.physics.location.y
        self.ball.z = packet.game_ball.physics.location.z
        #print(packet.game_cars[self.index])

        carBallAngle = getAngle(packet.game_ball.physics.location.x - self.car.x, packet.game_ball.physics.location.z - self.car.z, 0, 0)


        # print('z:', self.car.z, 'vz:', self.car.vz, 'y:', self.car.y, 'vy:', self.car.vy, 'x:', self.car.x, 'vx:', self.car.vx, 'pitch:', self.car.pitch, 'roll: ', self.car.roll, 'yaw:', self.car.yaw)

        #fixing pitch since RL uses euler angles
        pi = math.pi
        p = float(self.car.pitch)
        r = float(self.car.roll)
        y = float(self.car.yaw)

        curang = float(getXZangle(p, r, y))
        # print('actual pitchL', actualPitch[0])
        #Create desired and current vectors [x, z, theta, vx, vz, omegay]
        # self.current = np.matrix([self.car.z, self.car.x, actualPitch, self.car.vz, self.car.vx, self.car.wy])
        # self.desired = np.matrix([800, self.car.x, math.pi/4, self.car.vz, self.car.vx, self.car.wy])
        self.current = np.matrix([curang, self.car.wy])
        self.desired1 = np.matrix([-1*float(carBallAngle[0]),0])
        self.desired2 = self.desired1 #np.matrix([-2, 0])


        #calculate Adesired
        As = 900 #uu/s2
        Ag = np.matrix([0,-650 ])
        theta = carBallAngle
        Adesired = np.matrix([(As*math.cos(theta)), (As * math.sin(theta))])
        Aa = Adesired - Ag

        Ax = float(Aa.item(0))
        Az = float(Aa.item(1))
        Ax2 = math.pow(Ax, 2)
        Az2 = math.pow(Az, 2)

        A = (math.sqrt(Ax2 + Az2))
        boostPercent = A / 991.666 * 100
        boostPercent = max(min(boostPercent, 100), 0)

        thetaActual = getAngle(Aa.item(0), Aa.item(1), 0, 0)
        # print('Theta actual:', thetaActual, 'Aactual: ', Aa, 'boost percent: ', boostPercent)
        thetaDesired = np.matrix([-1*float(thetaActual), 0])


        #Quaternion stuff
        q0 = Quaternion(axis = [1.,0.,0.], radians = self.car.roll)
        q1 = Quaternion(axis = [0.,1.,0.], radians = self.car.pitch)
        q2 = Quaternion(axis = [0.,0.,1.], radians = -1*self.car.yaw)
        q3 = q0*q1*q2
        xprime = q3.rotate([1,0,0])

        qcar = q3.rotate([1.,0.,0.])

        # print('xprime: ', xprime, 'q3:', q3)

        #get point vectors for car and ball
        ux = np.array([1.,0.,0.])
        uy = np.array([0.,1.,0.])
        uz = np.array([0.,0.,1.])
        Pb = np.array([packet.game_ball.physics.location.x, packet.game_ball.physics.location.y, packet.game_ball.physics.location.z])
        Pc = np.array([self.car.x, self.car.y, self.car.z]) #negate z because z axis for car is pointed downwards
        Pbc = np.subtract(Pb, Pc) #Get vector to ball from car in car coordinates

        xyz = np.cross(ux, Pbc) #xyz of quaternion is rotation between Pbc and unit x vector
        w = math.sqrt(1 * ((Pbc.item(0) ** 2) + (Pbc.item(1) ** 2) + (Pbc.item(2) ** 2))) + np.dot(ux, Pbc) #scalr of quaternion
        qbcworld = Quaternion(w = w, x = xyz.item(0), y = xyz.item(1), z = xyz.item(2))
        eulerAngles = toEulerAngle(qbcworld.unit) #convert quaternion to euler angles


        #getting quaternion describing car orientation
        qw2c = Quaternion(w=0, x = self.car.roll, y = self.car.pitch, z = self.car.yaw)
        # Pcar = qw2c.unit.rotate(ux)
        q0 = Quaternion(axis = [1.,0.,0.], radians = self.car.roll)
        q1 = Quaternion(axis = [0.,1.,0.], radians = self.car.pitch)
        q2 = Quaternion(axis = [0.,0.,-1.], radians = self.car.yaw)
        q3 = q0.unit*q1.unit*q2.unit
        Pcar = q3.unit.rotate(ux)
        # print('pcar', xprime, 'pitch', self.car.pitch)
        xyzCar = np.cross(ux, Pcar) #xyz of quaternion is rotation between Pbc and unit x vector
        wCar = math.sqrt(1 * ((Pcar.item(0) ** 2) + (Pcar.item(1) ** 2) + (Pcar.item(2) ** 2))) + np.dot(ux, Pcar) #scalr of quaternion
        qcarworld = Quaternion(w = wCar, x = xyzCar.item(0), y = xyzCar.item(1), z = xyzCar.item(2))

        #Angular Velocity Inputs to the control algorithm
        omegades = np.matrix([0,0,0])
        omegacur = self.CoordinateSystems.w_car

        #get current position of trajectory
        position = self.Trajectory.circularTrajectory(1000, 800, 4)
        self.Trajectory.progress()
        print('trajectory', position)

        #CONTROL SYSTEM ALGORITHMS
        #get acceleration vector
        # Pdes = np.array([packet.game_ball.physics.location.x, packet.game_ball.physics.location.y, packet.game_ball.physics.location.z])
        Pcur = np.array([self.car.x, self.car.y, self.car.z])
        # Pdes = np.array([0,self.ball.y,800])
        Pdes = position

        # Vdes = np.array([self.car.vx,self.car.vy,100])
        Vdes = np.array([0,0,0])
        Vcur = np.array([self.car.vx, self.car.vy, self.car.vz])
        acc, accMagnitude = getAccelerationVector(Pdes, Pcur, Vdes, Vcur)
        accfix = np.array([-1*acc.item(0), acc.item(1), acc.item(2)])
        boostPercent = accMagnitude / 991.666 * 100
        boostPercent = max(min(boostPercent, 100), 0)
        # print('acc:', acc, 'accmag', accMagnitude)
        Qworld_to_acceleration_vector = self.CoordinateSystems.createQuaternion_world_at_car(accfix)
        torques = getTorques(self.CoordinateSystems.Qworld_to_car, Qworld_to_acceleration_vector, omegades, omegacur)

        #Send Data to feedbackcontroller
        #self.car.printVals();
        if(self.counter.count < self.counter.lap1):
            self.fbController.pitchControl(thetaDesired, self.current)
            #try only controlling z velocity
            # self.fbController.pitchControl(self.car.z, self.car.x, self.car.pitch, self.car.vz, self.car.vx, self.car.wz, self.car.z, self.car.x, self.car.pitch, self.car.vz, self.car.vx, self.car.wz)
            self.counter.increment()
        if(self.counter.count >= self.counter.lap1) and (self.counter.count < self.counter.lap2):
            self.fbController.pitchControl(thetaDesired, self.current)
            # self.fbController.pitchControl(0, 0, 0, 10, 0, 0, self.car.z, self.car.x, self.car.pitch, self.car.vz, self.car.vx, self.car.wz)

            self.counter.increment()
        if(self.counter.count >= self.counter.lap2):
            self.counter.reset()


        #Setting Controller state from values in controller
        #boost value
        # self.controller_state.boost = 1 #self.boostCounter.boost(self.fbController.boostPercent)
        # self.controller_state.boost = self.boostCounter.boost(self.fbController.boostPercent)
        self.controller_state.boost = self.boostCounter.boost(boostPercent)

        #roll, pitch, yaw values
        self.controller_state.pitch = max(min(torques.item(1), 1), -1)
        self.controller_state.roll = max(min(torques.item(0), 1), -1)
        self.controller_state.yaw =  -1*max(min(torques.item(2), 1), -1) #changes in rotations about coordinate system cause z axis changes

        # self.controller_state.pitch = 0.0#max(min(torques.item(1), 1), -1) #self.fbController.pitchPercent
        # self.controller_state.roll = 0.0#max(min(torques.item(0), 1), -1)
        # self.controller_state.yaw = 1#max(min(torques.item(2), 1), -1)

        #Make car jump if its on the floor
        if(packet.game_cars[self.index].has_wheel_contact):
            self.controller_state.jump = True
        else:
            self.controller_state.jump = False
        #Contol ball for testing functions
        x, y, z, vi = self.BallController.bounce(500,500,1000,1000)
        Vt = self.BallController.rotateAboutZ(np.matrix([0,0,0]), math.pi/10)
        # vx = self.BallController.oscillateX(-1500, 0, 1000)
        vx = Vt.item(0)
        vy = Vt.item(1)
        # vz = 0

        #Set ball and car states to set game state for testing
        # ball_state = BallState(Physics(location=Vector3(x, y, z), velocity = Vector3(0, 0, vi)))
        ball_state = BallState(Physics(velocity = Vector3(vx, vy, 1)))
        # ,
        # car_state = CarState(jumped=True, double_jumped=False, boost_amount=0,
        #                  physics=Physics(location = Vector3(-1000, 0, 500),velocity=Vector3(0, 0, 0), rotation = Rotator(pitch = eulerAngles.item(1), yaw = eulerAngles.item(2), roll = eulerAngles.item(0))))
        # car_state = CarState(jumped=True, double_jumped=False, boost_amount=0,
        #                  physics=Physics(location = Vector3(00, 0, 500),velocity=Vector3(0, 0, z=0)))#, rotation = Rotator(pitch = 0, yaw = 0, roll = 0)))
        car_state = CarState(jumped=True, double_jumped=False, boost_amount=1,
                         physics=Physics(location = Vector3(500, 0, 500),velocity=Vector3(0, 0, 1100), rotation = Rotator(yaw = math.pi/2, pitch = -1*math.pi/2, roll = math.pi/2)))


        if(self.BallController.release == 0):
            game_state = GameState(ball = ball_state, cars = {self.index: car_state})
            self.set_game_state(game_state)
        game_state = GameState(ball = ball_state)#, cars = {self.index: car_state})
        self.set_game_state(game_state)
        #RENDERING
        self.renderer.begin_rendering()

        #helpful vectors for rendering
        car = np.array([self.car.x, self.car.y, self.car.z]).flatten()
        ball = np.array([self.ball.x, self.ball.y, self.ball.z]).flatten()
        origin = np.array([0,0,0])
        #World coordinate system
        self.renderer.draw_line_3d(np.array([0,0,0]), np.array([500,0,0]), self.renderer.red())
        self.renderer.draw_line_3d(np.array([0,0,0]), np.array([0,500,0]), self.renderer.green())
        self.renderer.draw_line_3d(np.array([0,0,0]), np.array([0,0,500]), self.renderer.blue())

        #Car coordinate system
        headingx = 100*self.CoordinateSystems.toWorldCoordinates(np.array([1,0,0])) #multiply by 100 to make line longer
        headingy = 100*self.CoordinateSystems.toWorldCoordinates(np.array([0,1,0]))
        headingz = 100*self.CoordinateSystems.toWorldCoordinates(np.array([0,0,1]))
        self.renderer.draw_line_3d(np.array([self.car.x, self.car.y, self.car.z]), np.array([self.car.x + headingx.item(0), self.car.y + headingx.item(1), self.car.z + headingx.item(2)]), self.renderer.red())
        self.renderer.draw_line_3d(np.array([self.car.x, self.car.y, self.car.z]), np.array([self.car.x + headingy.item(0), self.car.y + headingy.item(1), self.car.z + headingy.item(2)]), self.renderer.green())
        self.renderer.draw_line_3d(np.array([self.car.x, self.car.y, self.car.z]), np.array([self.car.x + headingz.item(0), self.car.y + headingz.item(1), self.car.z + headingz.item(2)]), self.renderer.blue())

        #Car direction vector on world coordinate system
        self.renderer.draw_line_3d(np.array([0,0,0]), np.array([headingx.item(0), headingx.item(1), headingx.item(2)]), self.renderer.red())
        self.renderer.draw_line_3d(np.array([0,0,0]), np.array([headingy.item(0), headingy.item(1), headingy.item(2)]), self.renderer.green())
        self.renderer.draw_line_3d(np.array([0,0,0]), np.array([headingz.item(0), headingz.item(1), headingz.item(2)]), self.renderer.blue())

        #Draw position of ball after converting from Pball_car to Pball_world
        # desired = np.array(self.CoordinateSystems.getVectorToBall_world()).flatten()
        # self.renderer.draw_line_3d(car, car + desired, self.renderer.yellow())

        #Car to ball vector

        # self.renderer.draw_rect_3d(car+desired, 100, 100, 0, self.renderer.teal())

        #Acceleration vector
        a = np.array([accfix.item(0), accfix.item(1),-1*accfix.item(2)])
        self.renderer.draw_line_3d(car, car + a/10, self.renderer.pink())

        #trajectory vector
        self.renderer.draw_line_3d(origin, position, self.renderer.orange())
        self.renderer.end_rendering()


        #printing
        # print('wx:', self.car.wx, 'wy:', self.car.wy, 'wz:', self.car.wz)
        return self.controller_state

    def reset(self):
        game_state = GameState()
        self.set_game_state(game_state)
        car_state = CarState(physics(location = Vector3 (x=-1000, y=0, z=250), velocity=Vector3(x=0, y=0, z=0), rotation=Rotator(0, 0, 0), angular_velocity=Vector3(0, 0, 0)))

        ball_state = BallState(Physics(location=Vector3(500, 0, None)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

    def setBall(self):
        ball_state = BallState(Physics(location=Vector3(1000,0,800)))
        game_state = GameState(ball=ball_state)
        self.set_game_state(game_state)

    def setCarState(self):
        game_state = GameState()
        self.set_game_state(game_state)
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=100,
                         physics=Physics(location = Vector3 (x=-1800, y=0, z=550), velocity=Vector3(x=0, y=0, z=0), rotation=Rotator(math.pi / 2, 0, 0), angular_velocity=Vector3(0, 0, 0)))

        ball_state = BallState(Physics(location=Vector3(1000, 0, 800), velocity=Vector3(x=0, y=0, z=0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)


class Vector2:
    def __init__(self, x=0, y=0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, val):
        return Vector2(self.x + val.x, self.y + val.y)

    def __sub__(self, val):
        return Vector2(self.x - val.x, self.y - val.y)

    def correction_to(self, ideal):
        # The in-game axes are left handed, so use -x
        current_in_radians = math.atan2(self.y, -self.x)
        ideal_in_radians = math.atan2(ideal.y, -ideal.x)

        correction = ideal_in_radians - current_in_radians

        # Make sure we go the 'short way'
        if abs(correction) > math.pi:
            if correction < 0:
                correction += 2 * math.pi
            else:
                correction -= 2 * math.pi

        return correction


class Ball:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
    def update(self, data):
        self.x = data.physics.location.x
        self.y = data.physics.location.y
        self.z = data.physics.location.z
        self.vx = data.physics.velocity.x
        self.vy = data.physics.velocity.y
        self.vz = data.physics.velocity.z

class Car:
    def __init__(self):
        #Member variables initialized

        #mass
        self.mass = None
        #position
        self.x = None
        self.y = None
        self.z = None

        #velocity
        self.vx = None
        self.vy = None
        self.vz = None

        #Pitch Roll yaw
        self.pitch = None
        self.roll = None
        self.yaw = None

        #angular velocities
        self.wx = None
        self.wy = None
        self.wz = None

        #MUST CHECK THSESE TO MAKE SURE THEY CORRELATE PROPERLY

    def update(self, data):
        #Member variables initialized
        #position
        self.x = data.physics.location.x
        self.y = data.physics.location.y
        self.z = data.physics.location.z

        #velocity
        self.vx = data.physics.velocity.x
        self.vy = data.physics.velocity.y
        self.vz = data.physics.velocity.z

        #Pitch Roll yaw
        self.pitch = data.physics.rotation.pitch
        self.roll = data.physics.rotation.roll
        self.yaw = data.physics.rotation.yaw

        #angular velocities
        self.wx = data.physics.angular_velocity.x
        self.wy = data.physics.angular_velocity.y
        self.wz = data.physics.angular_velocity.z

    def printVals(self):
        print("x:", int(self.x), "y:", self.y, "z:", self.z, "wx:", self.wx, "wy:", self.wy, "wz:", self.wz)

class FeedbackController: #This is the controller algorithm possibly full state feedback possibly state observer?
    def __init__(self):
        self.values = None
        self.boostPercent = 0.0 #boost Percentage 0 - 100
        self.lastz = None #the tick priors value of position
        self.lastvz = None #the tick priors value of velocity
        self.lastError = 0.0 #prior error
        self.errorzI = 0.0 #summation of values for integral of error
        self.errorzICOUNTER = 1.0

        #Rocket League physics (using unreal units (uu))
        self.gravity = 650 #uu/s^2

        #State Space matrix coefficients
        self.mass = 14.2
        self.M = 1/14.2
        self.A = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.B = np.array([[0.0], [1.0]])
        self.C = np.array([[1, 0], [0, 0]])
        self.D = np.array([[0], [0]])
        self.system = control.ss(self.A, self.B, self.C, self.D, None)
        #print(self.B)
        self.poles = np.array([-2.5, -2.8])
        self.K = control.place(self.A, self.B, self.poles)
        self.k1 = self.K[0, 0]
        self.k2 = self.K[0, 1]
        self.kr = 1.0

        #print(self.K)
        #print(control.pole(self.system))

    def defineStateSpaceSystem(self, A, B, C, D, dt):
        self.system = control.ss(A, B, C, D, dt)

    def heightControl(self, desz, desvz, z, vz):

        #define matricies


        errorzP = desvz - vz #error of position
        self.errorzICOUNTER = self.errorzICOUNTER + 1.0
        self.errorzI = (errorzP + self.errorzI) / self.errorzICOUNTER
        errorzD = errorzP + self.lastError

        P = 1
        I = 0
        D = 0


        #output = (errorzP * P) + (self.errorzI * I) + (errorzD * D)
        output = errorzP * P
        self.boostPercent = output

        #Print data
        #print(desvz,'|', vz, '|', output, '|', errorzD * D)
        #print(errorzP, '|', vz)

    def gainFlightControl(self, desz, desvz, z, vz):
        u = (-self.k1*(z - desz)) + (-self.k2 * (vz - desvz))#add desz here since equation considers xequilibrium point as center
        #print('u:', int(u), '/', 'z:', int(z), '/', 'vz:', int(vz), '/', 'desz', int(desz), '/', 'desvz', int(desvz))
        self.boostPercent = (u)


class FeedbackController2: #This is the controller algorithm possibly full state feedback possibly state observer?
    def __init__(self):
        self.values = None
        self.boostPercent = 0.0 #boost Percentage 0 - 100
        self.pitchPercent = 0 #pitch percentage
        self.lastz = None #the tick priors value of position
        self.lastvz = None #the tick priors value of velocity
        self.lastError = 0.0 #prior error
        self.errorzI = 0.0 #summation of values for integral of error
        self.errorzICOUNTER = 1.0

        #Rocket League physics (using unreal units (uu))
        self.g = 650 #uu/s^2
        self.Dp = -2.798194258050845 #Drag coeff for pitch
        self.Tp = 12.14599781908070
        T_r = -36.07956616966136; # torque coefficient for roll
        T_p = -12.14599781908070; # torque coefficient for pitch
        T_y =   8.91962804287785; # torque coefficient for yaw
        D_r =  -4.47166302201591; # drag coefficient for roll
        D_p = -2.798194258050845; # drag coefficient for pitch
        D_y = -1.886491900437232; # drag coefficient for yaw
        self.I = 1
        self.m = 180 #mass of the car arbitrary units


        #State Space matrix coefficients
        self.A = np.matrix([[0, 1], [0, 0]])
        self.B = np.matrix([[0],[self.Tp]])
        self.C = np.matrix([[1, 0], [0,1]])
        self.D = np.matrix([[0],[0]])
        self.system = control.ss(self.A, self.B, self.C, self.D, None)
        self.controllability = control.ctrb(self.A, self.B)
        print("ctrb:", self.controllability)
        #print(self.B)

        #print(control.ctrb(self.A, self.B))
        self.poles = np.array([-1000 , -5])
        print('poles: ', self.poles)
        #self.eigen= control.pole(self.system)
        self.eigen = self.system.pole()

        print("Eigen: ", self.eigen)
        self.K = control.place(self.A, self.B, self.poles)
        print("\nK: ", self.K)

    def defineStateSpaceSystem(self, A, B, C, D, dt):
        self.system = control.ss(A, B, C, D, dt)

    def heightControl(self, desz, desvz, z, vz):

        #define matricies


        errorzP = desvz - vz #error of position
        self.errorzICOUNTER = self.errorzICOUNTER + 1.0
        self.errorzI = (errorzP + self.errorzI) / self.errorzICOUNTER
        errorzD = errorzP + self.lastError

        P = 1
        I = 0
        D = 0


        #output = (errorzP * P) + (self.errorzI * I) + (errorzD * D)
        output = errorzP * P
        self.boostPercent = output

        #Print data
        #print(desvz,'|', vz, '|', output, '|', errorzD * D)
        #print(errorzP, '|', vz)

    def gainFlightControl(self, desz, desvz, z, vz):
        u = (-self.k1*(z - desz)) + (-self.k2 * (vz - desvz))#add desz here since equation considers xequilibrium point as center
        #print('u:', int(u), '/', 'z:', int(z), '/', 'vz:', int(vz), '/', 'desz', int(desz), '/', 'desvz', int(desvz))
        self.boostPercent = (u)

    def pitchControl(self, desired, current):

        # current = np.matrix([ z,  x,  theta,  vz,  vx,  omega])
        # desired = np.matrix( [desz,  desx,  destheta,  desvz,  desvx ,  desomega] )
        # print('current: ', current, ' desired: ', desired)
        # print("des - cur:", np.subtract(desired, current))
        u = -1*np.matmul(self.K, np.subtract(desired, current).T)
        #print("u:", u)
        self.torque = u

        self.pitchPercent = u
        self.pitchPercent =  max(min(self.pitchPercent, 1), -1)

        # print('current: ', current, ' desired: ', 'u: ', u)

        # print('err: ', np.subtract(desired, current), 'u: ', u, 'cur:', current[0])
        return self.boostPercent, self.pitchPercent
        #print("err_z: ", (desz - z), "err_x: ", (desx - x), "err_theta: ", (destheta - theta), "err_vz: ", (desvz - vz), "err_vx: ", (desvx - x), "err_vomega: ", (desomega - omega) )

class FeedbackController3: #This is the controller algorithm possibly full state feedback possibly state observer?
    def __init__(self):
        self.values = None
        self.boostPercent = 0.0 #boost Percentage 0 - 100
        self.pitchPercent = 0 #pitch percentage
        self.lastz = None #the tick priors value of position
        self.lastvz = None #the tick priors value of velocity
        self.lastError = 0.0 #prior error
        self.errorzI = 0.0 #summation of values for integral of error
        self.errorzICOUNTER = 1.0

        #Rocket League physics (using unreal units (uu))
        self.g = 650 #uu/s^2
        self.D = 2.798194258050845 #Drag coeff for pitch
        self.I = 1 #=
        self.m = 180 #mass of the car arbitrary units


        #State Space matrix coefficients
        Asvd = np.matrix([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0,0.0, 0.0, 0.0], [ self.g, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, (-1*self.D/self.I)]])

        self.A = np.matrix([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0,0.0,0.0, 0.0, 0.0], [0.0, 0.0, self.g, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, (-1*self.D/self.I)]])
        self.B = np.matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [(1 / self.m), 0.0], [0.0, 0.0], [0.0, 1.0]])
        self.C = np.matrix([[1,1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        self.D = np.matrix([[0, 0], [0, 0]])
        self.Q = np.matrix([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,5]])
        self.R = np.matrix([[1, 0], [0, 1]])
        self.system = control.ss(self.A, self.B, self.C, self.D, None)
        self.controllability = control.ctrb(self.A, self.B)
        print("ctrb:", self.controllability)
        #print(self.B)

        #print(control.ctrb(self.A, self.B))
        self.poles = np.array([-100000,-100,-20, -30, -1000000000000000000000, -100000000])
        #self.eigen= control.pole(self.system)
        self.eigen = self.system.pole()
        #Asvd = np.matrix([[0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0], [0.0,0.0, 0.0, 0.0], [self.g, 0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0, (-1*self.D/self.I)]])

        U,S,V = np.linalg.svd(self.A)
        #print("H: ", H)
        print("U: ", U, "\nS: ", S, "\nV: ", V)
        print("Eigen: ", self.eigen)
        #self.poles = np.array([0.00000000e+00+0.00000000e+00j, -3.58188374e+00+0.00000000e+00j, 2.54338639e-06+4.40528914e-06j, -2.54338639e-06-4.40528914e-06j, 5.08677277e-06+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j])
        self.K = control.place(self.A, self.B, self.eigen)
        print("\nK: ", self.K)
        #self.Klqr, self.state, self.eigen = lqr(self.A, self.B, self.Q, self.R)
        #self.K, self.X, self.eigen = lqr(self.A, self.B, self.Q, self.R)
        #print('K matrix:', self.K, '\nState matrix: ', self.state, '\nEigen values: ', self.eigen)

        self.k1 = self.K[0, 0]
        self.k2 = self.K[0, 1]
        self.k3 = self.K[0, 2]
        self.k4 = self.K[0, 3]
        self.k5 = self.K[0, 4]
        self.k6 = self.K[0, 5]
        self.k7 = self.K[1, 0]
        self.k8 = self.K[1, 1]
        self.k9 = self.K[1, 2]
        self.k10 = self.K[1, 3]
        self.k11 = self.K[1, 4]
        self.k12 = self.K[1, 5]
        self.kr = 1.0

        #inputs to the car
        # self.thrust = 0
        # self.torque = 0
        #print(self.K)
        #print(control.pole(self.system))

    def defineStateSpaceSystem(self, A, B, C, D, dt):
        self.system = control.ss(A, B, C, D, dt)

    def heightControl(self, desz, desvz, z, vz):

        #define matricies


        errorzP = desvz - vz #error of position
        self.errorzICOUNTER = self.errorzICOUNTER + 1.0
        self.errorzI = (errorzP + self.errorzI) / self.errorzICOUNTER
        errorzD = errorzP + self.lastError

        P = 1
        I = 0
        D = 0


        #output = (errorzP * P) + (self.errorzI * I) + (errorzD * D)
        output = errorzP * P
        self.boostPercent = output

        #Print data
        #print(desvz,'|', vz, '|', output, '|', errorzD * D)
        #print(errorzP, '|', vz)

    def gainFlightControl(self, desz, desvz, z, vz):
        u = (-self.k1*(z - desz)) + (-self.k2 * (vz - desvz))#add desz here since equation considers xequilibrium point as center
        #print('u:', int(u), '/', 'z:', int(z), '/', 'vz:', int(vz), '/', 'desz', int(desz), '/', 'desvz', int(desvz))
        self.boostPercent = (u)

    def pitchControl(self, desired, current):
        #convert theta counted counter clockwise positive from the +z axis, game gives us ccw from the +x axis
        #theta = theta -( math.pi/2)

        #input values considering all state variables
        # u1 = (self.g*self.m) + (-self.k1*(z- desz)) + (-self.k2 * (vz-desvz)) + (-self.k3 * (x - desx)) + (-self.k4 * (vx - desvx)) + (-self.k5 * (theta - destheta)) + (-self.k6 * (omega - desomega))
        # u2 = (-self.k7*(z- desz)) + (-self.k8 * (vz-desvz)) + (-self.k9 * (x - desx)) + (-self.k10 * (vx - desvx)) + (-self.k11 * (theta - destheta)) + (-self.k12 * (omega - desomega))

        #input values considering only z and xs
        # u1 = (self.g*self.m) + (-self.k1*(z- desz)) + (-self.k2 * (vz-desvz)) + (-self.k3 * (x - desx)) + (-self.k4 * (vx - desvx))
        # u2 = (-self.k7*(z- desz)) + (-self.k8 * (vz-desvz)) + (-self.k9 * (x - desx)) + (-self.k10 * (vx - desvx))


        # current = np.matrix([ z,  x,  theta,  vz,  vx,  omega])
        # desired = np.matrix( [desz,  desx,  destheta,  desvz,  desvx ,  desomega] )
        #print("des - cur:", desired - current)
        u = -1*np.matmul(self.K, np.transpose(desired - current))
        #print("u:", u)
        self.thrust = u[0]# + (self.T/self.m) #u[0] is u1equil, so add T/m to bring it back to u1 [du1e = u1 + u1e]

        self.torque =  u[1]

        #Convert desired thrust and torque to percent thrust
        acc = self.thrust/self.m
        self.boostPercent = (self.thrust/991.666) * 100
        self.boostPercent = max(min(self.boostPercent, 100), 0)
        self.pitchPercent = (self.torque/12.46)
        self.pitchPercent =  max(min(self.pitchPercent, 1), -1)

        # print('current: ', current, ' desired: ', desired, 'bP: ',  int(self.boostPercent), 'pP: ', float(self.pitchPercent))

        return self.boostPercent, self.pitchPercent
        #print("err_z: ", (desz - z), "err_x: ", (desx - x), "err_theta: ", (destheta - theta), "err_vz: ", (desvz - vz), "err_vx: ", (desvx - x), "err_vomega: ", (desomega - omega) )

class orientationCounter:
    def __init__(self):
        self.counter = 0.0
        self.max = 10.0

    def pitch(self, desiredPitchPercentage):
        #print(desiredBoostPercentage)
        if(self.counter >= self.max): #If counter is at max make sure to make it zero before sending boost confimation
            if((self.counter / self.max) > (desiredPitchPercentage / 100.0)):
                self.counter = 0
                return 0
            else:
                self.counter = 0
                return 1
        if((self.counter / self.max) > (desiredPitchPercentage / 100.0)):
            #Turn on boost
            self.counter = self.counter + 1
            return 0
        else:
            self.counter = self.counter + 1

            return 1
class BoostCounter: #
    def __init__(self):
        self.counter = 0.0
        self.max = 10.0

    def boost(self, desiredBoostPercentage):
        #print(desiredBoostPercentage)
        if(self.counter >= self.max): #If counter is at max make sure to make it zero before sending boost confimation
            if((self.counter / self.max) > (desiredBoostPercentage / 100.0)):
                self.counter = 0
                return 0
            else:
                self.counter = 0
                return 1
        if((self.counter / self.max) > (desiredBoostPercentage / 100.0)):
            #Turn on boost
            self.counter = self.counter + 1
            return 0
        else:
            self.counter = self.counter + 1

            return 1

class BoostCounter2:
    def __init__(self):
        self.boost_counter = 0.0
        self.B_max = 1000.0

    def boost(self, boostPercentage):
        boostPercentage = clamp(boostPercentage, 0, 1000)
        use_boost = 0.0
        use_boost -= round(self.boost_counter)
        self.boost_counter += (boostPercentage) / self.B_max
        use_boost += round(self.boost_counter)

        #print(self.boost_counter, boostPercentage, use_boost)
        if(use_boost):
            return 1
        else:
            return 0

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class TestCounter:
    def __init__(self, l1, l2):
        self.count = 0
        self.lap1 = l1
        self.lap2 = l2
    def reset(self):
        self.count = 0
    def increment(self):
        self.count = self.count + 1
    def decrement(self):
        self.count = self.count - 1

def get_car_facing_vector(data):
    pitch = float(data.physics.rotation.pitch)
    yaw = float(data.physics.rotation.yaw)

    facing_x = math.cos(pitch) * math.cos(yaw)
    facing_y = math.cos(pitch) * math.sin(yaw)

    return Vector2(facing_x, facing_y)


#Helpful function/classes for the future
def turn_radius(v):
    if v == 0:
        return 0
    return 1.0 / curvature(v)

# v is the magnitude of the velocity in the car's forward direction
def curvature(v):
    if 0.0 <= v < 500.0:
        return 0.006900 - 5.84e-6 * v
    elif 500.0 <= v < 1000.0:
        return 0.005610 - 3.26e-6 * v
    elif 1000.0 <= v < 1500.0:
        return 0.004300 - 1.95e-6 * v
    elif 1500.0 <= v < 1750.0:
        return 0.003025 - 1.10e-6 * v
    elif 1750.0 <= v < 2500.0:
        return 0.001800 - 0.40e-6 * v
    else:
        return 0.0

#LQR functions from http://www.mwm.im/lqr-controllers-with-python/
def lqr(A,B,Q,R):
    # """Solve the continuous time lqr controller.
    #
    # dx/dt = A x + B u
    #
    # cost = integral x.T*Q*x + u.T*R*u
    # """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))

    eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return K, X, eigVals

def dlqr(A,B,Q,R):
    # """Solve the discrete time lqr controller.
    #
    # x[k+1] = A x[k] + B u[k]
    #
    # cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))

    eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return K, X, eigVals

def getXZangle(pitch, roll, yaw):
    #Create rotation matricies from euler angles, then pick out data necessary to get desired 2D angle

    Ry = np.matrix([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])

    Rp = np.matrix([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    Rr = np.matrix([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])

    R =  Rr @ Rp @ Ry

    unitx = np.matrix([1.0,0.0,0.0]) #X axis unit vector
    vp = np.dot(R, unitx.transpose()) #new direction vector of body of car
    #print(vp)

    v1 = np.matrix([1,0]) #XZ plane x unit vector
    v2 = np.matrix([vp.item(0), vp.item(2)] ) #rotated vector projected onto XZ plane
    n_v1 = np.linalg.norm(v1, ord=1)
    n_v2 = np.linalg.norm(v2, ord=1)
    cosang = np.inner(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    angle = np.arctan2(sinang, cosang)

    angle = np.sign(vp[2]) * angle #make sure angle is negative if we go past 180def, do this my multiplying angle by the sign of the z position of the resulstant orientation vector vp
    # print('v1: ', v1, " v2: ", v2, ' sangle:', sinang, ' cang: ', cosang, ' angle:', angle)
    return angle #returning angle to x axis

def getAngle(bx,bz,cx,cz):
    c = np.matrix([cx, cz])
    b = np.matrix([bx, bz])
    v1 = np.matrix([1, 0])
    v2 = np.matrix(np.subtract(b, c))
    cosang = np.inner(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    angleCtoB = np.arctan2(sinang, cosang)
    angleCtoB = np.sign(v2.item(1)) * angleCtoB #fix angle
    return angleCtoB

#Quaternion functions
def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = cos(theta)
    x = x * sin(theta)
    y = y * sin(theta)
    z = z * sin(theta)
    return w, x, y, z

def q_to_axisangle(q):
    w, v = q[0], q[1:]
    theta = acos(w) * 2.0
    return normalize(v), theta

def toEulerAngle(q):
    w = q.scalar
    x = q.vector[0]
    y = q.vector[1]
    z = q.vector[2]

    #roll (x axis rotation)
    sinr_cosp = 2.0*((w*x)+(y*z))
    cosr_cosp = 1.0-(2.0*((x*x)+(y*y)))
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    #pitch y axis rotation
    sinp = +2.0 * ((w * y) - (z * x))
    if (math.fabs(sinp) >= 1):
        pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = +2.0 * ((w*z) + (x*y))
    cosy_cosp = +1.0 - 2.0 * ((y*y) + (z*z))
    yaw = np.arctan2(siny_cosp, cosy_cosp)


    return np.matrix([roll, pitch, yaw])

def getTorques(Qw2c, Qw2b, wdes, wcur):
    #define gains
    kq = np.matrix([100., 100., 100.]).T #quaternion error gain
    kw = np.matrix([10, 10, 10]).T #omega error gain

    T_r = -36.07956616966136; # torque coefficient for roll
    T_p = -12.14599781908070; # torque coefficient for pitch
    T_y =   8.91962804287785; # torque coefficient for yaw
    #define local quaternions
    Qdes = Quaternion(Qw2b.normalised)
    Qcur = Quaternion(Qw2c.normalised)

    #get error Quaternion
    Qerr = Qcur.normalised * Qdes.normalised.conjugate.normalised
    # print('Qerr:', Qerr)

    #check for q0 < 0 and if this is true use Q* (conjugate) to give closest rotation
    if(Qerr.scalar < 0):     #use the conjugate of Qerr
        Qerr = Qerr.unit.conjugate.unit

    #renormalize quaternion
    Qerr = Qerr.unit

    #trying different method to find error
    theta = np.arccos(Qerr[0]) * 2
    w = Qerr[0]
    q1err = w + Qerr[1]*math.sin(theta/2)
    q2err = w + Qerr[2]*math.sin(theta/2)
    q3err = w + Qerr[2]*math.sin(theta/2)

    qerrnew = np.array([q1err, q2err, q3err])
    #get omega errors
    werr = np.matrix(np.subtract(wdes, wcur))

    #get torques
    q = np.matrix([Qerr.vector])
    # q = np.matrix(qerrnew)
    torques = -1*(kq * q) - (kw * werr)
    # print('theta', theta, 'torques', torques, 'qerr', Qerr.unit, 'qw2c', Qw2c.unit, 'qc2b', Qw2b.unit)
    return torques

def getAccelerationVector(Pdesired, Pcurrent, Vdesired, Vcurrent):
    #gains for state feedback control
    kp = np.array([2, 2, 3])
    kv = np.array([2, 2, 5])
    #gravity vector
    gravity = np.array([0,0,-650])

    #error vectors
    Perr = Pdesired - Pcurrent
    Verr = Vdesired - Vcurrent

    acc = -1*np.multiply(kp, Perr) - (np.multiply(kv, Verr))
    acc_gravityfix = acc + gravity
    accMagnitude = np.linalg.norm(acc_gravityfix)

    return acc_gravityfix, accMagnitude
