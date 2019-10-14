  #imports for LQR functions
from __future__ import division, print_function
import csv
import sys
import os


import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import Plotting
import Predictions
from CoordinateSystems import CoordinateSystems
# from BallController import BallController
from pyquaternion import Quaternion
# from TrueProportionalNavigation import TPN
from Controller import Controller
# import State as s
# import DrivingEquations
# import DiscreteDynamicModel as ddm
import Optimization2
from Trajectory import Trajectory
# from Car import Car
from GUI import GUI
from tkinter import Tk, Label, Button, StringVar, Entry, Listbox

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
#import slycot

#imports for LQR functions
# from __future__ import division, print_function
import numpy as np
import scipy.linalg
import time

import traceback
import copy
import queue

import queue_testing
from EnvironmentManipulator import EnvironmentManipulator


class Lotus(BaseAgent):

    def initialize_agent(self):
        #Initialize GUI
        self.EM = EnvironmentManipulator()
        self.root = Tk()
        self.g = GUI(self.root, self.EM)

        # Controller clsas with Algorithm to follow trajectory
        self.controller = Controller()

        # self.csv_write_flag = 0
        #This runs once before the bot starts up
        self.packet = None
        self.controller_state = SimpleControllerState()

        # # Set ball initial conditions
        # self.ball_si = [-1200.0, 400.0]
        # self.ball_vi = [1200.0, 1200.0]
        #
        # # Ball and car states that you can call for controlling environement
        # #Set ball and car states to set game state for testing
        # # self.ball_state1 = BallState(Physics(location=Vector3(x, y, z), velocity = Vector3(0, 0, vi)))
        # # self.ball_state2 = BallState(Physics(location=Vector3(ballpos[0],ballpos[1],ballpos[2]), velocity = Vector3(ballvel[0],ballvel[1],ballvel[2])))
        # self.ball_stateHold = BallState(Physics(location=Vector3(0, 0, 800), velocity = Vector3(0, 0, 0)))
        # # self.ball_state = BallState(Physics(velocity = Vector3(vx, vy, 1)))
        # self.ball_high_pos = Vector3(1000, -1000, 1500)
        # self.ball_state_high = BallState(Physics(location=self.ball_high_pos, velocity = Vector3(0, 0, 300)))
        # self.ball_state_none = BallState()
        # self.ball_state_linear = BallState(Physics(velocity = Vector3(0, -200, 300)))
        # self.ball_state_optimizer_test = BallState(physics = Physics(location=Vector3(self.ball_si[0], 0, self.ball_si[1]), velocity = Vector3(self.ball_vi[0], 0, self.ball_vi[1])))
        # # car_state = CarState(jumped=True, double_jumped=False, boost_amount=0,
        # #                  physics=Physics(location = Vector3(-1000, 0, 500),velocity=Vector3(0, 0, 0), rotation = Rotator(pitch = eulerAngles.item(1), yaw = eulerAngles.item(2), roll = eulerAngles.item(0))))
        # self.car_state = CarState(boost_amount=1,
        #                  physics=Physics(location = Vector3(-1000, -3000, 100),velocity=Vector3(0, -300, 200), rotation = Rotator(pitch = math.pi/8, yaw = math.pi/2, roll = 0), angular_velocity = Vector3(0,0,0)))
        # self.car_state_hold = CarState(boost_amount=1,
        #                  physics=Physics(location = Vector3(00, 00, 500), velocity = Vector3(0,0,0)))
        # self.car_state_falling = CarState(boost_amount=0)
        # self.car_state_high = CarState( boost_amount=0,
        #                  physics=Physics(location = Vector3(2200, 00, 1500), velocity = Vector3(0,0,00), rotation = Rotator(pitch = self.r_ti, yaw = 0.0, roll = 0.0)))
        # self.car_state_optimizer_test = CarState(
        #                  physics=Physics(location = Vector3(self.s_ti[0], 0, self.s_ti[1]), velocity = Vector3(self.v_ti[0],0,self.v_ti[1]), rotation = Rotator(pitch = self.r_ti, yaw = 0.0, roll = 0.0)))
        # # Reference starting car state
        # self.car_start = Car()
        # self.car_start.x = -1500.0
        # self.car_start.y = 2500.0
        # self.car_start.vx = 0.0
        # self.car_start.vy = 0.0
        # self.car_start.yaw = -math.pi/2
        #
        # # Desired car state
        # self.car_desired = Car()
        #
        # self.car_desired.x = self.ball_high_pos.x
        # self.car_desired.y = self.ball_high_pos.y
        # self.car_desired.position = np.array([self.car_desired.x, self.car_desired.y, 0])
        # self.car_desired.vx = 1000
        # self.car_desired.vy = 1000
        #
        # self.car_state_optimizer_driving_test = CarState(
        #          physics=Physics(location = Vector3(self.car_start.x, self.car_start.y, 17), velocity = Vector3(self.car_start.vx,self.car_start.vy,0), rotation = Rotator(pitch = 0, yaw = self.car_start.yaw, roll = 0.0)))
        #
        # game_state = GameState(cars = {self.index: self.car_state}, ball = self.ball_state_high)
        # self.set_game_state(game_state)
        #
        # # self.optimizer.solving = False
        # game_state = GameState(cars = {self.index: self.car_state_optimizer_driving_test}, ball = self.ball_state_high)
        # self.set_game_state(game_state)
        #
        # self.flag = True


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        #Update GUI and pass important values to GUI
        self.g.label_tnow.config(text=str(round(float(packet.game_info.seconds_elapsed), 2)))
        self.root.update()

        # Check if Trajectory should be started and start trajectory
        if(self.EM.start_trajectory == True):

            print('initializing environment')
            self.g.label_t0.config(text=str(round(float(packet.game_info.seconds_elapsed), 2))) #Set the t0 label on the GUI to current time
            self.EM.start_trajectory = False # Reset trajectory starting flag to prevent re-initializeation
            self.EM.setEnvironment(self) # Send self to environment manager to allow it to manuiplate the game state

            self.controller.on = True #Turn the realtime controller on

            # Send trajectory and input data to controller
            ts, sx, sy, vx, vy, yaw, omega, curvature = self.g.getTrajectoryData()
            self.controller.setTrajectoryData(ts, sx, sy, vx, vy, yaw, omega, curvature)
            a, turning = self.g.getInputData()
            self.controller.setInputData(a, turning)
            self.controller.t0 = packet.game_info.seconds_elapsed

            # Set Environment

            game_state = GameState(cars = {self.index: self.EM.car_initial_state}, ball = self.EM.ball_initial_state)
            self.set_game_state(game_state)

        # Update data to controller
        self.controller.setTNOW(float(packet.game_info.seconds_elapsed))
        self.controller.setCurrentState(packet.game_cars[self.index])
        # self.controller_state = self.controller.openLoop() # Get controller value from openLoop algorithm
        self.controller_state = self.controller.feedBack() # Get controller value from feedback algo

        vx = packet.game_cars[self.index].physics.velocity.x
        vy= packet.game_cars[self.index].physics.velocity.y
        vmag = np.sqrt((vx*vx) + (vy*vy))
        print(vmag)
        # print(self.controller.t_now)

        # print("ENvironmentmanipulator value: ", self.EM.getValue())

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
        car_state = CarState(boost_amount=100,
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

    def get_current_u_star(self, ti, tf):
        #Get the current thrust value from the optimal control vector, dependent on what the current time is
        controller = con.Controller()

        if(ti != None):
            if((tf - ti) > self.t[-1]):
                print('completed optimal trajectory')
                controller.boostPercent = self.thrust.value[-1]
                controller.steer = self.steer.value[-1]
                controller.throttle = self.throttle.value[-1]
                return controller
            t = (tf - ti)# - 1/60

            # Search time array for the index of the closest value
            idx = np.searchsorted(self.t, t, side="left")
            if (idx > 0) and (idx == len(self.t) or math.fabs(t - self.t[idx-1]) < math.fabs(t - self.t[idx])):
                index = idx-1
            else:
                index = idx

            print('index', idx)
            # u_thrust_current = self.u_thrust_star[index]
            # u_pitch_current = self.u_pitch_star[index]
            # u_thrust_current = self.u_thrust_star[idx]
            # u_pitch_current = self.u_pitch_star[idx]
            # if(self.optimizer.u_thrust_d.value[idx] > 0.8):
            #     u_thrust_current = 1.0
            # else:
            #     u_thrust_current = 0

            # u_throttle_current = self.optimizer.u_throttle_d.value[idx]
            # u_turning_current = self.optimizer.u_turning_d.value[idx]
            # u_pitch_current = (tf - self.t_star[index]) * (self.u_pitch_star[index + 1] - self.u_pitch_star[index]) / (self.t_star[index+1] - self.t_star[index])


            # print('u thrust current:', u_thrust_current, 'u pitch:', u_pitch_current, 'time', t)
            # get bboost percent
            # boostPercent = u_thrust_current / (991.666 + 60) * 100
            # boostPercent = max(min(boostPercent, 100), 0)

            #Set controller values
            # controller.boostPercent = u_thrust_current
            # controller.torques = np.array([0, u_pitch_current, 0])

            #Driving controller values
            controller.boostPercent = self.thrust.value[idx]
            controller.steer = self.steer.value[idx]
            controller.throttle = self.throttle.value[idx]

        else:
            controller.boostPercent = 0
            controller.torques = np.array([0, 0, 0])

        return controller

    def state_controller(self):
        #Do algorithm to determine what state the car should be in
        None

    def setControllerState(self, controller):
        # self.controller_state.boost = self.boostCounter.boost(boostPercent)
        # self.controller_state.boost = self.boostCounter.boost(controller.boostPercent)
        self.controller_state.boost = controller.boostPercent
        # #roll, pitch, yaw values
        self.controller_state.pitch = max(min(controller.torques.item(1), 1), -1)
        self.controller_state.roll = max(min(controller.torques.item(0), 1), -1)
        self.controller_state.yaw =  -1*max(min(controller.torques.item(2), 1), -1) #changes in rotations about coordinate system cause z axis changes
        self.controller_state.jump = controller.jump
        self.controller_state.throttle = controller.throttle
        self.controller_state.steer = controller.steer

    def ground_interception(self):
        controller = con.Controller()
        controller.throttle = 0.5
        controller.steer = 0.75
        k_min, k_actual = DrivingEquations.get_turning_radius(self.car, controller)
        # print('k_min', k_min, 'k_actual', k_actual, 'omegaz', self.car.wz, 'vel_mag', np.linalg.norm(self.car.velocity))

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
        boostPercent = TotalAccelerationMag / (991.666 + 60) * 100
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

    def save_trajectory_data(self):
        # Car and ball have been released, set model_state_t0 to current state
        if(self.car_set == False):
            self.model_states = None
        if(self.car_set == True):
            if(self.model_states == None):
                self.model_states = [ddm.ModelState(self.car, ddm.ControlVariables(self.controller_state), self.CoordinateSystems, self.packet.game_info.seconds_elapsed, self.ball)]
                print(self.packet.game_ball)
                pass
            if(self.ti != None and self.completed_optimal_trajectory == False):
                # print('model state length', len(self.model_states))
                self.model_states.append(ddm.ModelState(self.car, ddm.ControlVariables(self.controller_state), self.CoordinateSystems, self.packet.game_info.seconds_elapsed, self.ball))
            elif(self.completed_optimal_trajectory == True and self.csv_write_flag == False):
                print('exporting csv')
                self.export_model_states_as_csv()
                self.csv_write_flag = 1
                # self.model_states.append(ddm.ModelState(self.car, ddm.ControlVariables(self.controller_state), self.CoordinateSystems, self.packet.game_info.seconds_elapsed))
                # self.model_states.pop()
        # print(len(self.model_states))

    def export_model_states_as_csv(self):
        f = open('test_data.csv', 'w', newline = "")
        writer = csv.writer(f)
        writer.writerow(['ti', 'time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'quaternion', 'boost', 'u roll', 'u pitch', 'u yaw', 'theta roll', 'theta pitch', 'theta yaw', 'wx', 'wy', 'wz', 'ball sx', 'ball sy', 'ball sz', 'ball vx', 'ball vy', 'ball vz'])
        for i in range(len(self.model_states)):
            obj = self.model_states[i]
            row = [self.ti, obj.time,
                obj.position_world[0], obj.position_world[1], obj.position_world[2],
                obj.velocity_world[0], obj.velocity_world[1], obj.velocity_world[2],
                obj.acceleration_world[0], obj.acceleration_world[1], obj.acceleration_world[2],
                obj.coordinate_system.Qcar_to_world, obj.control_variables.boost,
                obj.control_variables.torques[0], obj.control_variables.torques[1], obj.control_variables.torques[2],
                obj.attitude[0],obj.attitude[1],obj.attitude[2],
                obj.angular_velocity[0],obj.angular_velocity[1],obj.angular_velocity[2],
                obj.ball_position[0],obj.ball_position[1], obj.ball_position[2],
                obj.ball_velocity[0], obj.ball_velocity[1], obj.ball_velocity[2]]
            writer.writerow(row)

        f = open('optimization_data.csv', 'w', newline = "")
        writer = csv.writer(f)
        writer.writerow(['time', 'sx', 'sz', 'ball sx', 'ball sz', 'opt ball vz', 'opt pitch']) # , 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'quaternion', 'boost', 'roll', 'pitch', 'yaw'])
        for i in range(len(self.t_star)):
            row = [self.t_star[i], self.sx_star.value[i], self.sz_star.value[i], self.ball_sx.value[i], self.ball_sz.value[i], self.ball_vz.value[i], self.pitch_star.value[i]]
            writer.writerow(row)
            print('wrote row', row)

    def predict_car(self):
        #torque coefficients
        T_r = 36.07956616966136; # torque coefficient for roll
        T_p = 12.14599781908070; # torque coefficient for pitch
        T_y =   8.91962804287785; # torque coefficient for yaw

        boostVector = np.array([float(self.controller_state.boost) * (991.666 + 60), 0, 0])
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
        vavg = (self.v1 + self.v0) / 2
        # def predict(position, velocity, q, omega, a, torques, t0, t1):
        predictedp1, predictedv1 = Predictions.predict(self.p0, self.v0, self.q0, self.w0, aavg, self.T0, self.t0, self.t1)
        self.errorv = ((predictedv1 - self.v1)*100 / self.v1 )**2
        self.errorp = ((predictedp1 - self.p1)*100 / self.p1)**2


    def predict_ball(self):
        #PREDICTIONS
        #torque coefficients
        T_r = 36.07956616966136; # torque coefficient for roll
        T_p = 12.14599781908070; # torque coefficient for pitch
        T_y =   8.91962804287785; # torque coefficient for yaw
        #boost vector in car coordinates
        # boostVector = np.array([self.controller_state.boost * 991.666, 0, 0])
        #Get values at tk and tk - 1
        self.s_before = self.s_now
        self.s_now = self.ball.position
        self.v_before = self.v_now
        self.v_now = self.ball.velocity
        # self.p0 = self.p1 #position vector at initial time
        # self.p1 = self.car.position #position vector at tk+1
        # self.v0 = self.v1 #velocity at prior frame
        # self.v1 = self.car.velocity
        # self.q0 = self.q1 #Orientation at prior frame
        # self.q1 = self.CoordinateSystems.Qworld_to_car.conjugate.normalised
        # self.w0 = self.w1 #angular velocity at prior frame
        # self.w1 = self.car.angular_velocity
        # self.a0 = self.a1 #accelreation vector at prior frame
        # self.a1 = self.CoordinateSystems.toWorldCoordinates(boostVector)
        # self.T0 = self.T1 #Torque vector at prior frame
        # self.T1 = np.array([self.controller_state.roll * T_r, self.controller_state.pitch * T_p, self.controller_state.yaw * -1 * T_y])

        # aavg = (self.a1 + self.a0) / 2
        # vavg = (self.v1 + self.v0 / 2)
        # predictedp1, predictedv1 = Predictions.predict(self.p0, self.v0, self.q0, self.w0, aavg, self.T0, self.t0, self.t1)
        self.ballposition = Predictions.predictBallTrajectory(self.ball, self.t1)
        self.ballerror = Predictions.ballPredictionError(self.s_before, self.s_now, self.v_before, self.v_now, self.t0, self.t1)
        self.ballerror = self.ballerror**(1/2)

        self.time_to_ground, self.s_ground = Predictions.predict_ball_to_ground(self.ball, self.t1)
        # self.errorv = (predictedv1 - self.v1)**2
        # self.errorp = (predictedp1 - self.p1)**2
        # print("error^2 v:", errorv, "error^2 p:", errorp)
        # print("z actual:", self.car.z, "z predicted:", predictedp1[2])
        # self.data.add(self.v1[0], predictedv1[0], self.errorv[0], self.t1)

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


class BoostCounter: #
    def __init__(self):
        self.counter = 0.0
        self.max = 10.0

    def boost(self, desiredBoostPercentage):
        #print(desiredBoostPercentage)
        desiredBoostPercentage = clamp(desiredBoostPercentage, 0.0, 100.0)
        if(desiredBoostPercentage == 0):
            return 0
        if(self.counter >= self.max): #If counter is at max make sure to make it zero before sending boost confimation
            if((self.counter / self.max) > (desiredBoostPercentage / 100.0)):
                self.counter = 0
                return 0
            else:
                self.counter = 0
                return 0
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

class BoostCounter3:
    def __init__(self):
        self.boost_counter = 0.0
        self.B_max = 1000.0

    def boost(self, avgAcceleration):
        use_boost = 0.0
        use_boost -= round(self.boost_counter)
        self.boost_counter += (avgAcceleration) / self.B_max
        use_boost += round(self.boost_counter)

        #print(self.boost_counter, boostPercentage, use_boost)
        if(use_boost):
            return 1
        else:
            return 0
