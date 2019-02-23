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

class Test2(BaseAgent):

    def initialize_agent(self):
        #This runs once before the bot starts up
        self.packet = None
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
        self.data = Plotting.data()
        self.TPN = TPN()
        self.state = s.State()
        self.controller = con.Controller()

        self.err_previous_PID = 0
        self.integration_previous_PID = 0

        self.s_before = np.array([0,0,0])
        self.s_now = np.array([0,0,0])
        self.v_before = np.array([0,0,0])
        self.v_now = np.array([0,0,0])
        self.t0 = 0 #Time at initial time step
        self.t1 = 0 #Time at time step tk+1
        self.p0 = np.array([0,0,0]) #position vector at initial time
        self.p1 = np.array([0,0,0]) #position vector at tk+1
        self.v0 = np.array([0,0,0]) #velocity at initial time
        self.v1 = np.array([0,0,0])
        self.q0 = Quaternion() #Orientation at initial time
        self.q1 = Quaternion()
        self.w0 = np.array([0,0,0]) #angular velocity at initial time
        self.w1 = np.array([0,0,0])
        self.a0 = np.array([0,0,0]) #accelreation vector at initial time
        self.a1 = np.array([0,0,0])
        self.T0 = np.array([0,0,0]) #Torque vector at initial time
        self.T1 = np.array([0,0,0])

        self.plotTime0 = None
        self.plotTime1 = None
        self.plotFlag = False

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        #update class data
        self.update_data(packet)

        #shift t1 to previous time t0, and set t1 to the new current time
        self.t0 = self.t1
        self.t1 = packet.game_info.seconds_elapsed

        #Set car state (forcing to whichever algo i am working on)
        self.state.set_state(0)

        controller_flight = self.flight_interception()

        controller_ground = self.ground_interception()

        #Set the controller state depending on which state the car is in
        if(self.state.current_state == 0): #Car is in driving state
            self.setControllerState(controller_ground)
        if(self.state.current_state == 1):
            self.setControllerState(controller_flight)

        #Control ball for testing functionss
        x, y, z, vi = self.BallController.bounce(500,500,300,1500)
        Vt = self.BallController.rotateAboutZ(np.matrix([0,0,0]), math.pi/10)
        p = np.array([00, 1500, 100])
        v = np.array([-600, -500, 1500])
        ballpos, ballvel = self.BallController.projectileMotion(p, v)
        # vx = self.BallController.oscillateX(-1500, 0, 1000)
        vx = Vt.item(0)
        vy = Vt.item(1)
        # vz = 0

        #Set ball and car states to set game state for testing
        ball_state1 = BallState(Physics(location=Vector3(x, y, z), velocity = Vector3(0, 0, vi)))
        ball_state2 = BallState(Physics(location=Vector3(ballpos[0],ballpos[1],ballpos[2]), velocity = Vector3(ballvel[0],ballvel[1],ballvel[2])))
        ball_stateHold = BallState(Physics(location=Vector3(0, 0, 800), velocity = Vector3(0, 0, 0)))
        ball_state = BallState(Physics(velocity = Vector3(vx, vy, 1)))
        ball_state_high = BallState(Physics(location=Vector3(0, 0, 1200), velocity = Vector3(0, 0, 300)))
        ball_state_none = BallState()
        ball_state_linear = BallState(Physics(velocity = Vector3(0, -200, 300)))

        # car_state = CarState(jumped=True, double_jumped=False, boost_amount=0,
        #                  physics=Physics(location = Vector3(-1000, 0, 500),velocity=Vector3(0, 0, 0), rotation = Rotator(pitch = eulerAngles.item(1), yaw = eulerAngles.item(2), roll = eulerAngles.item(0))))
        car_state = CarState(jumped=True, double_jumped=False, boost_amount=1,
                         physics=Physics(location = Vector3(-1000, -3000, 100),velocity=Vector3(0, -300, 200), rotation = Rotator(pitch = math.pi/8, yaw = math.pi/2, roll = 0), angular_velocity = Vector3(0,0,0)))
        car_state_hold = CarState(jumped=True, double_jumped=False, boost_amount=1,
                         physics=Physics(location = Vector3(00, 00, 500), velocity = Vector3(0,0,0)))
        car_state_falling = CarState(jumped=True, double_jumped=False, boost_amount=0)
        car_state_high = CarState(jumped=True, double_jumped=False, boost_amount=0,
                         physics=Physics(location = Vector3(300, 500, 1000), velocity = Vector3(0,0,800)))
        # car_state2 = CarState(jumped=True, double_jumped=False, boost_amount=0,
        #                  physics=Physics(location = Vector3(00, 0, 500),velocity=Vector3(0, 0, 0)))

        #Pointed down car state for maximum initial error
        # car_state = CarState(jumped=True, double_jumped=False, boost_amount=1,
        #                  physics=Physics(location = Vector3(500, 0, 500),velocity=Vector3(0, 0, 1100), rotation = Rotator(yaw = math.pi/2, pitch = -1*math.pi/2, roll = math.pi/2)))


        if(self.BallController.release == 0):
            # game_state = GameState(ball = ball_state2, cars = {self.index: car_state_hold})
            game_state = GameState(ball = ball_state2, cars = {self.index: car_state})
            self.set_game_state(game_state)
        # else:
        #
        #     # game_state = GameState(cars = {self.index: car_state_hold})
        #     game_state = GameState(ball = ball_state_linear)
        #     # game_state = GameState(ball = ball_state2, cars = {self.index: car_state_hold})
        #
        # self.set_game_state(game_state)

        #Reset to initial states after counter runs
        if(self.BallController.counter1 > 800):
            self.BallController.release = 0
            self.BallController.counter1 = 0
        else:
            # game_state = GameState(ball = ball_state, cars = {self.index: car_stateHoldPosition})
            # game_state = GameState(cars = {self.index: car_state_falling})
            game_state = GameState(ball = ball_state_none)
            self.set_game_state(game_state)

        #Predictions
        self.prediction()

        #RENDERING
        self.render()

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

    def update_data(self, packet):
        self.packet = packet
        self.car.update(packet.game_cars[self.index])
        self.ball.update(packet.game_ball)
        self.CoordinateSystems.update(self.car, self.ball)
        self.BallController.update(packet.game_ball)
        self.TPN.update(self.car, self.ball, packet.game_info.seconds_elapsed)

    def plotData(self, data):
        self.fig1 = plt.figure()
        self.ax1  = self.fig1.gca()
        self.ax1.plot(data.d4, data.d3)
        # self.fig2 = plt.figure()
        # self.ax2  = self.fig3.gca()
        # self.ax2.plot(data.d4, data.d2)
        # self.fig3 = plt.figure()
        # self.ax3  = self.fig3.gca()
        # self.ax3.plot(data.d4, data.d3)

        # self.ax1.plot(data.d4, data.d2, 'r')
        # self.ax1.plot(data.d4, data.d3, 'o')
        # self.ax2.plot(data.d4, data.d2)
        # self.ax3.plot(data.d4, data.d3)
        plt.show()

    def state_controller(self):
        #Do algorithm to determine what state the car should be in
        None

    def setControllerState(self, controller):
        # self.controller_state.boost = self.boostCounter.boost(boostPercent)
        self.controller_state.boost = self.boostCounter2.boost(controller.boostPercent)
        # #roll, pitch, yaw values
        self.controller_state.pitch = max(min(controller.torques.item(1), 1), -1)
        self.controller_state.roll = max(min(controller.torques.item(0), 1), -1)
        self.controller_state.yaw =  -1*max(min(controller.torques.item(2), 1), -1) #changes in rotations about coordinate system cause z axis changes
        self.controller_state.jump = controller.jump
        self.controller_state.throttle = controller.throttle
        self.controller_state.brake = controller.brake
        self.controller_state.turn = controller.turn

    def ground_interception(self):
        controller = con.Controller()
        return controller

    def flight_interception(self):
        controller = con.Controller()
        #True Proportionanl navigation
        gravity = np.array([0,0,-650])
        Vb_para, Vc_para, Vb_perp, Vc_perp, latax = self.TPN.getNavigationValues()
        latax_unit = latax / np.linalg.norm(latax)
        latax_mag = np.linalg.norm(latax)

        # if(latax_mag > 500):
        #     latax = latax_unit * 500 #prevent latax from exceeding 500

        # latax = np.array([latax[0], latax[1], -1 * latax[2]])
        # latax = self.CoordinateSystems.Qcar_to_world.rotate(latax)
        unit_to_ball = (self.ball.position - self.car.position) / np.linalg.norm(self.ball.position - self.car.position)
        # Acc_to_ball = ( unit_to_ball) * np.linalg.norm(self.ball.position - self.car.position) * 0.5

        Pcur = np.array([self.car.x, self.car.y, self.car.z])
        Vcur = self.car.velocity
        # Vdes = np.array([0,0,100])
        Vdes = unit_to_ball * 1500
        Pdes = np.array([self.car.x, self.car.y, self.car.z])
        Acc_to_ball, accMag = getAccelerationVector_TPN_gains(Pdes, Pcur, Vdes, Vcur)
        TotalAcceleration = Acc_to_ball + latax
        # TotalAcceleration = -gravity
        Acc_unit = TotalAcceleration / np.linalg.norm(TotalAcceleration)
        TotalAccelerationMag = np.linalg.norm(TotalAcceleration)
        boostPercent = TotalAccelerationMag / 991.666 * 100
        boostPercent = max(min(boostPercent, 100), 0)

        # Qdesired = self.CoordinateSystems.createQuaternion_world_at_car(unit_to_ball)
        Qdesired= self.CoordinateSystems.createQuaternion_from_point_and_roll(TotalAcceleration, self.car.roll)
        omegades = np.matrix([0,0,0])
        omegacur = self.CoordinateSystems.w_car

        torques = getTorques(self.CoordinateSystems.Qworld_to_car, Qdesired, omegades, omegacur)
        # torques, self.err_previous_PID, self.integration_previous_PID = getTorquesPID(self.CoordinateSystems.Qworld_to_car, Qworld_to_acceleration_vector, self.t0, self.t1, self.err_previous_PID, self.integration_previous_PID)
        #Testing quaternion
        toBall = self.ball.position - self.car.position

        #Save data to self variables for rendering
        self.Acc_to_ball = Acc_to_ball
        self.latax = latax
        self.TotalAcceleration = TotalAcceleration

        if(self.packet.game_cars[self.index].has_wheel_contact):
            jump = True
        else:
            jump = False

        controller.torques = torques
        controller.boostPercent = boostPercent
        controller.jump = jump

        return controller

    def prediction(self):
        #PREDICTIONS
        #torque coefficients
        T_r = 36.07956616966136; # torque coefficient for roll
        T_p = 12.14599781908070; # torque coefficient for pitch
        T_y =   8.91962804287785; # torque coefficient for yaw
        #boost vector in car coordinates
        boostVector = np.array([self.controller_state.boost * 991.666, 0, 0])
        #Get values at tk and tk - 1
        self.s_before = self.s_now
        self.s_now = self.ball.position
        self.v_before = self.v_now
        self.v_now = self.ball.velocity
        self.p0 = self.p1 #position vector at initial time
        self.p1 = self.car.position #position vector at tk+1
        self.v0 = self.v1 #velocity at prior frame
        self.v1 = self.car.velocity
        self.q0 = self.q1 #Orientation at prior frame
        self.q1 = self.CoordinateSystems.Qworld_to_car.conjugate.normalised
        self.w0 = self.w1 #angular velocity at prior frame
        self.w1 = self.car.angular_velocity
        self.a0 = self.a1 #accelreation vector at prior frame
        self.a1 = self.CoordinateSystems.toWorldCoordinates(boostVector)
        self.T0 = self.T1 #Torque vector at prior frame
        self.T1 = np.array([self.controller_state.roll * T_r, self.controller_state.pitch * T_p, self.controller_state.yaw * -1 * T_y])

        aavg = (self.a1 + self.a0) / 2
        vavg = (self.v1 + self.v0 / 2)
        predictedp1, predictedv1 = Predictions.predict(self.p0, self.v0, self.q0, self.w0, aavg, self.T0, self.t0, self.t1)
        self.ballposition = Predictions.predictBallTrajectory(self.ball, self.t1)
        self.ballerror = Predictions.ballPredictionError(self.s_before, self.s_now, self.v_before, self.v_now, self.t0, self.t1)
        self.ballerror = self.ballerror**(1/2)
        self.errorv = (predictedv1 - self.v1)**2
        self.errorp = (predictedp1 - self.p1)**2
        # print("error^2 v:", errorv, "error^2 p:", errorp)
        # print("z actual:", self.car.z, "z predicted:", predictedp1[2])
        self.data.add(self.v1[0], predictedv1[0], self.errorv[0], self.t1)

    def render(self):
        try:
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
            # headingx = 100*self.CoordinateSystems.toWorldCoordinates(np.array([1,0,0])) #multiply by 100 to make line longer
            # headingy = 100*self.CoordinateSystems.toWorldCoordinates(np.array([0,1,0]))
            # headingz = 100*self.CoordinateSystems.toWorldCoordinates(np.array([0,0,1]))
            # self.renderer.draw_line_3d(np.array([self.car.x, self.car.y, self.car.z]), np.array([self.car.x + headingx.item(0), self.car.y + headingx.item(1), self.car.z + headingx.item(2)]), self.renderer.red())
            # self.renderer.draw_line_3d(np.array([self.car.x, self.car.y, self.car.z]), np.array([self.car.x + headingy.item(0), self.car.y + headingy.item(1), self.car.z + headingy.item(2)]), self.renderer.green())
            # self.renderer.draw_line_3d(np.array([self.car.x, self.car.y, self.car.z]), np.array([self.car.x + headingz.item(0), self.car.y + headingz.item(1), self.car.z + headingz.item(2)]), self.renderer.blue())

            #Car direction vector on world coordinate system
            # self.renderer.draw_line_3d(np.array([0,0,0]), np.array([headingx.item(0), headingx.item(1), headingx.item(2)]), self.renderer.red())
            # self.renderer.draw_line_3d(np.array([0,0,0]), np.array([headingy.item(0), headingy.item(1), headingy.item(2)]), self.renderer.green())
            # self.renderer.draw_line_3d(np.array([0,0,0]), np.array([headingz.item(0), headingz.item(1), headingz.item(2)]), self.renderer.blue())

            #TPN vectors
            #LOS Vector
            LOS, LOSunit = self.TPN.getUnitVectors()
            self.renderer.draw_line_3d(LOS + self.car.position, self.car.position, self.renderer.pink())
            # self.renderer.draw_line_3d(LOS + self.car.position, self.car.position, self.renderer.yellow())

            self.renderer.draw_line_3d(self.Acc_to_ball + self.car.position, self.car.position, self.renderer.yellow())
            #V parallel vectors
            # Vb_para, Vc_para, Vb_perp, Vc_perp, TPN_acceleration = self.TPN.getNavigationValues()
            # self.renderer.draw_line_3d(Vb_para + self.ball.position, self.ball.position, self.renderer.pink())
            # self.renderer.draw_line_3d(Vc_para + self.car.position, self.car.position, self.renderer.pink())
            # self.renderer.draw_line_3d(Vb_perp + Vb_para + self.ball.position, Vb_para + self.ball.position, self.renderer.cyan())
            # self.renderer.draw_line_3d(Vc_perp + Vc_para + self.car.position, Vc_para + self.car.position, self.renderer.cyan())
            #draw velocity vectors for sanity check
            # self.renderer.draw_line_3d(self.ball.velocity + self.ball.position, self.ball.position, self.renderer.orange())
            # self.renderer.draw_line_3d(self.car.velocity + self.car.position, self.car.position, self.renderer.orange())
            self.renderer.draw_line_3d(self.TotalAcceleration + self.car.position, self.car.position, self.renderer.white())
            self.renderer.draw_line_3d(self.latax + self.car.position + self.Acc_to_ball, self.car.position + self.Acc_to_ball, self.renderer.cyan())
            self.renderer.draw_line_3d(self.latax + self.car.position, self.car.position, self.renderer.cyan())
            self.renderer.draw_line_3d(self.TPN.omega*500 + self.car.position, self.car.position, self.renderer.red())

            # self.renderer.draw_line_3d(Acc_to_ball + self.car.position, self.car.position, self.renderer.yellow())
            # self.renderer.draw_line_3d(-1*gravity + self.car.position, self.car.position, self.renderer.green())
            # self.renderer.draw_line_3d(self.TPN.Nt + self.car.position, self.car.position, self.renderer.orange())

            # vec = self.CoordinateSystems.Qcar_to_world.rotate(np.array([500,0,0]))
            # self.renderer.draw_line_3d(vec + self.car.position, self.car.position, self.renderer.black())


            # self.renderer.draw_line_3d(toBall + self.car.position, self.car.position, self.renderer.white())


            #Ball trajectory PREDICTIONS

            self.renderer.draw_line_3d(self.ballposition[:, 1], self.ball.position, self.renderer.black())
            self.renderer.draw_line_3d(self.ballposition[:, 2], self.ball.position, self.renderer.black())
            self.renderer.draw_line_3d(self.ballposition[:, 3], self.ball.position, self.renderer.black())
            self.renderer.draw_line_3d(self.ballposition[:, 4], self.ball.position, self.renderer.black())
            self.renderer.draw_line_3d(self.ballposition[:, 5], self.ball.position, self.renderer.black())
            self.renderer.draw_line_3d(self.ballposition[:, 6], self.ball.position, self.renderer.black())
            self.renderer.draw_line_3d(self.ballposition[:, 7], self.ball.position, self.renderer.black())
            self.renderer.draw_line_3d(self.ballposition[:, 8], self.ball.position, self.renderer.black())
            self.renderer.draw_line_3d(self.ballposition[:, 9], self.ball.position, self.renderer.black())
            self.renderer.draw_line_3d(self.ballposition[:, 10], self.ball.position, self.renderer.black())

            self.renderer.draw_line_3d(self.ballposition[:, 0], self.ballposition[:, 1], self.renderer.yellow())
            self.renderer.draw_line_3d(self.ballposition[:, 1], self.ballposition[:, 2], self.renderer.cyan())
            self.renderer.draw_line_3d(self.ballposition[:, 2], self.ballposition[:, 3], self.renderer.pink())
            self.renderer.draw_line_3d(self.ballposition[:, 3], self.ballposition[:, 4], self.renderer.orange())
            self.renderer.draw_line_3d(self.ballposition[:, 4], self.ballposition[:, 5], self.renderer.green())
            self.renderer.draw_line_3d(self.ballposition[:, 5], self.ballposition[:, 6], self.renderer.yellow())
            self.renderer.draw_line_3d(self.ballposition[:, 6], self.ballposition[:, 7], self.renderer.cyan())
            self.renderer.draw_line_3d(self.ballposition[:, 7], self.ballposition[:, 8], self.renderer.pink())
            self.renderer.draw_line_3d(self.ballposition[:, 8], self.ballposition[:, 9], self.renderer.orange())
            self.renderer.draw_line_3d(self.ballposition[:, 9], self.ballposition[:, 10], self.renderer.green())

            #Draw position of ball after converting from Pball_car to Pball_world
            # desired = np.array(self.CoordinateSystems.getVectorToBall_world()).flatten()
            # self.renderer.draw_line_3d(car, car + desired, self.renderer.yellow())

            #Car to ball vector

            # self.renderer.draw_rect_3d(car+desired, 100, 100, 0, self.renderer.teal())

            #Acceleration vector
            # a = np.array([accfix.item(0), -1*accfix.item(1),-1*accfix.item(2)])
            # self.renderer.draw_line_3d(car, car + a/10, self.renderer.pink())
            #
            # #trajectory vector
            # self.renderer.draw_line_3d(origin, Pdes, self.renderer.orange())
            #
            #error rectangles velocities
            self.renderer.draw_rect_2d(10,10,int(self.errorv[0]**(1/2)), 30, True, self.renderer.cyan())
            self.renderer.draw_rect_2d(10,40,int(self.errorv[1]**(1/2)), 30, True, self.renderer.cyan())
            self.renderer.draw_rect_2d(10,70,int(self.errorv[2]**(1/2)), 30, True, self.renderer.cyan())
            #text for velocity errors
            self.renderer.draw_string_2d(10, 10, 1, 1, "X velocity Error: " + str(self.errorv[0]**(1/2)), self.renderer.white())
            self.renderer.draw_string_2d(10, 40, 1, 1, "Y velocity Error: " + str(self.errorv[1]**(1/2)), self.renderer.white())
            self.renderer.draw_string_2d(10, 70, 1, 1, "Z velocity Error: " + str(self.errorv[2]**(1/2)), self.renderer.white())
            #positions
            self.renderer.draw_rect_2d(10,100,int(self.errorp[0]**(1/2)), 30, True, self.renderer.red())
            self.renderer.draw_rect_2d(10,130,int(self.errorp[1]**(1/2)), 30, True, self.renderer.red())
            self.renderer.draw_rect_2d(10,190,int(self.errorp[2]**(1/2)), 30, True, self.renderer.red())

            self.renderer.draw_string_2d(10, 100, 1, 1, 'X position Error: ' + str(self.errorp[0]**(1/2)), self.renderer.white())
            self.renderer.draw_string_2d(10, 130, 1, 1, "Y position Error: " + str(self.errorp[1]**(1/2)), self.renderer.white())
            self.renderer.draw_string_2d(10, 160, 1, 1, "Z position Error: " + str(self.errorp[2]**(1/2)), self.renderer.white())
            #ball error
            self.renderer.draw_rect_2d(10,190,int(self.ballerror[0]), 30, True, self.renderer.red())
            self.renderer.draw_rect_2d(10,210,int(self.ballerror[1]), 30, True, self.renderer.red())
            self.renderer.draw_rect_2d(10,240,int(self.ballerror[2]), 30, True, self.renderer.red())

            self.renderer.draw_string_2d(10, 190, 1, 1, 'X ball Error: ' + str(self.ballerror[0]), self.renderer.white())
            self.renderer.draw_string_2d(10, 210, 1, 1, "Y ball Error: " + str(self.ballerror[1]), self.renderer.white())
            self.renderer.draw_string_2d(10, 240, 1, 1, "Z ball Error: " + str(self.ballerror[2]), self.renderer.white())

            self.renderer.end_rendering()
        except Exception as e:
            print ('Exception in rendering:', e)
            traceback.print_exc()


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
        self.x = 0
        self.y = 0
        self.z = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.position = np.array([0,0,0])
        self.velocity = np.array([0,0,0])
        self.Inertia = np.array([[1,0,0], [0,1,0], [0,0,1]])

    def update(self, data):
        self.x = data.physics.location.x
        self.y = data.physics.location.y
        self.z = data.physics.location.z
        self.vx = data.physics.velocity.x
        self.vy = data.physics.velocity.y
        self.vz = data.physics.velocity.z

        self.position = np.array([self.x,self.y,self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])

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
        self.position = np.array([0,0,0])
        self.velocity = np.array([0,0,0])
        self.angular_velocity = np.array([0,0,0])
        self.attitude = np.array([0,0,0])
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

        self.position = np.array([self.x,self.y,self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])
        self.angular_velocity = np.array([self.wx, self.wy, self.wz])
        self.attitude = np.array([self.roll, self.pitch, self.yaw])

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
        self.B_max = 100.0

    def boost(self, boostPercentage):
        boostPercentage = clamp(boostPercentage, 0, 100.0)
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
    kq = np.matrix([90., 90., 90.]).T #quaternion error gain
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
    # print(q)
    return torques

def getTorquesPID(Qcur, Qdes, t0, t1, err0, integration0):
    kp = -12
    ki = 0.00001
    kd = -9

    Qerr = Qcur.normalised * Qdes.normalised.conjugate.normalised
    #check for q0 < 0 and if this is true use Q* (conjugate) to give closest rotation
    if(Qerr.scalar < -1*math.pi):     #use the conjugate of Qerr
        Qerr = Qerr.unit.conjugate.unit

    err = np.matrix([Qerr.vector])
    integration_err = integration0 + (1/2) * (err - err0) * (t1-t0)
    derivative_err = (err-err0) / (t1-t0)

    torques = (kp*err) + (ki*integration_err) + (kd*derivative_err)
    return torques, err, integration_err

def getAccelerationVector_trajectory_gains(Pdesired, Pcurrent, Vdesired, Vcurrent):
    #gravity vector
    gravity = np.array([0,0,-650])

    #error vectors
    Perr = Pdesired - Pcurrent
    Verr = Vdesired - Vcurrent

    # print('perr:', Perr, 'Verr:', Verr)
    #Gain Scheduling formula
    if(Perr.item(2) > 0):
        kpx = 2
        kpy = 2
        kpz = Perr.item(2)/10
    else:
        kpx = 5
        kpy = 5
        kpz = 2
    if(Verr.item(2) > 0):
        kvz = Verr.item(2)/5
    else:
        kvz = 2

    #gains for state feedback control
    kp = np.array([kpx, kpy, kpz])
    kv = np.array([2, 2, kvz])

    #input vector calculation
    acc = -1*np.multiply(kp, Perr) - (np.multiply(kv, Verr))
    acc_gravityfix = acc + gravity
    accMagnitude = np.linalg.norm(acc_gravityfix)

    return acc_gravityfix, accMagnitude

def getAccelerationVector_TPN_gains(Pdesired, Pcurrent, Vdesired, Vcurrent):
    #gravity vector
    gravity = np.array([0,0,-650])

    #error vectors
    Perr = Pdesired - Pcurrent
    Verr = Vdesired - Vcurrent

    #gains for state feedback control
    kp = np.array([3,3,3])
    kv = np.array([2,2,2])

    #input vector calculation
    acc = np.multiply(kp, Perr) + (np.multiply(kv, Verr))
    acc_gravityfix = acc - gravity
    accMagnitude = np.linalg.norm(acc_gravityfix)

    return acc_gravityfix, accMagnitude
