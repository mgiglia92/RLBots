import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import integrate as int
import scipy.linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
from gekko import GEKKO
import csv
from mpl_toolkits.mplot3d import Axes3D
# This optimizer will minimize time to get to desired end points

# Next I will need to add to the cost function the error between the rocket and the desired set point

class Optimizer():
    def __init__(self):
#################GROUND DRIVING OPTIMIZER SETTTINGS##############
        self.d = GEKKO(remote=False) # Driving on ground optimizer

        ntd = 7
        self.d.time = np.linspace(0, 1, ntd) # Time vector normalized 0-1

        # options
        # self.d.options.NODES = 3
        self.d.options.SOLVER = 3
        self.d.options.IMODE = 6# MPC mode
        # m.options.IMODE = 9 #dynamic ode sequential
        self.d.options.MAX_ITER = 200
        self.d.options.MV_TYPE = 0
        self.d.options.DIAGLEVEL = 0

        # final time for driving optimizer
        self.tf = self.d.FV(value=1.0,lb=0.1,ub=100.0)

        # allow gekko to change the tfd value
        self.tf.STATUS = 1

        # Scaled time for Rocket league to get proper time

        # Acceleration variable
        self.a = self.d.MV(value = 1, lb = 0, ub = 1, integer=True)
        self.a.STATUS = 1
        self.a.DCOST = 1e-10

        # # Boost variable, its integer type since it can only be on or off
        # self.u_thrust_d = self.d.MV(value=0,lb=0,ub=1, integer=False) #Manipulated variable integer type
        # self.u_thrust_d.STATUS = 0
        # self.u_thrust_d.DCOST = 1e-5
        #
        # # Throttle value, this can vary smoothly between 0-1
        # self.u_throttle_d = self.d.MV(value = 1, lb = 0.02, ub = 1)
        # self.u_throttle_d.STATUS = 1
        # self.u_throttle_d.DCOST = 1e-5

        # Turning input value also smooth
        self.u_turning_d = self.d.MV(value = 1, lb = 0, ub = 2) # 0-2 since using -1 to +1 was causing issues. I shift the value in the differential equations. Worked well
        self.u_turning_d.STATUS = 1
        self.u_turning_d.DCOST = 1e-10

        # end time variables to multiply u2 by to get total value of integral
        self.p_d = np.zeros(ntd)
        self.p_d[-1] = 1.0
        self.final = self.d.Param(value = self.p_d)

        # integral over time for u_pitch^2
        # self.u2_pitch = self.d.Var(value=0)
        # self.d.Equation(self.u2.dt() == 0.5*self.u_pitch**2)

    def optimize2D(self, si, sf, vi, vf, ri, omegai): #these are 1x2 vectors s or v [x, z]
        #NOTE: I should make some data structures to easily pass this data around as one variable instead of so many variables


        #Trajectory to follow
        w = 0.5 # radians/sec of rotation
        amp = 100 #amplitude
        traj_sx = amp * self.d.cos(w * self.d.time)
        # self.t_sx = self.d.Var(value = amp) # Pre-Defined trajectory
        # self.d.Equation(self.t_sx.dt() == w * amp * self.d.sin(w * self.d.time))
        # self.t_sz = self.d.Var(value = 100) # Pre-Defined trajectory
        # self.d.Equation(self.t_sz.dt() == -1 * w * amp * self.d.cos(w * self.d.time))

        # variables intial conditions are placed here
            # Position and Velocity in 2d
        self.sx = self.d.Var(value=si[0], lb=-4096, ub=4096) #x position
        # self.vx = self.d.Var(value=vi[0]) #x velocity
        self.sy = self.d.Var(value=si[1], lb=-5120, ub=5120) #y position
        # self.vy = self.d.Var(value=vi[1]) #y velocity
            # Pitch rotation and angular velocity
        self.yaw = self.d.Var(value = ri) #orientation yaw angle
        self.omega = self.d.Var(value = omegai)
        # self.omega_yaw = self.d.Var(value=omegai, lb=-5.5, ub=5.5) #angular velocity
        # self.v_mag = self.d.Intermediate(self.d.sqrt((self.vx**2) + (self.vy**2)))
        self.v_mag = self.d.Var(value = self.d.sqrt((vi[0]**2) + (vi[1]**2)),  ub = 2500)

        self.curvature = self.d.Intermediate((0.0069 - ((7.67e-6) * self.v_mag) + ((4.35e-9)*self.v_mag**2) - ((1.48e-12) * self.v_mag**3) + ((2.37e-16) * self.v_mag**4)))

        self.vx = self.d.Intermediate(self.v_mag * self.d.cos(self.yaw))
        self.vy = self.d.Intermediate(self.v_mag * self.d.sin(self.yaw))
        # Errors to minimize trajectory error, take the derivative to get the final error value
        # self.errorx = self.d.Var(value = 0)
        # self.d.Equation(self.errorx.dt() == 0.5*(self.sx - self.t_sx)**2)


        # Differental equations
        self.d.Equation(self.v_mag.dt() == self.tf * self.a * (991.666+60))
        self.d.Equation(self.sx.dt() == self.tf * ((self.v_mag * self.d.cos(self.yaw))))
        self.d.Equation(self.sy.dt() == self.tf * ((self.v_mag * self.d.sin(self.yaw))))
        self.d.Equation(self.yaw.dt() == self.tf * ((self.u_turning_d - 1.0) * self.curvature * self.v_mag))



        # self.d.fix(self.sz, pos = len(self.d.time) - 1, val = 1000)


        #Soft constraints for the end point
        # Uncomment these 4 objective functions to get a simlple end point optimization
        #sf[1] is z position @ final time etc...
        self.d.Obj(self.final*1e4*(self.sy-sf[1])**2) # Soft constraints
        # self.d.Obj(self.final*1e3*(self.vy-vf[1])**2)
        self.d.Obj(self.final*1e4*(self.sx-sf[0])**2) # Soft constraints
        # self.d.Obj(self.final*1e3*(self.vx-vf[0])**2)


        #Objective function to minimize time
        self.d.Obj(self.tf*1e5)

        #Objective functions to follow trajectory
        # self.d.Obj(self.final * (self.errorx **2) * 1e3)

        # self.d.Obj(self.final*1e3*(self.sx-traj_sx)**2) # Soft constraints
        # self.d.Obj(self.errorz)
        # self.d.Obj(( self.all * (self.sx - trajectory_sx) **2) * 1e3)
        # self.d.Obj(((self.sz - trajectory_sz)**2) * 1e3)

        # minimize thrust used
        # self.d.Obj(self.u2*self.final*1e3)

        # minimize torque used
        # self.d.Obj(self.u2_pitch*self.final)

        #solve
        # self.d.solve('http://127.0.0.1') # Solve with local apmonitor server
        self.d.solve()

        # NOTE: another data structure type or class here for optimal control vectors
        # Maybe it should have some methods to also make it easier to parse through the control vector etc...
        # print('time', np.multiply(self.d.time, self.tf.value[0]))
        # time.sleep(3)

        self.ts = np.multiply(self.d.time, self.tf.value[0])

        return self.a, self.yaw, self.ts

    def getTrajectoryData(self):
        return [self.ts, self.sx, self.sy, self.vx, self.vy, self.yaw, self.omega]

    def getInputData(self):
        return [self.ts, self.a]
#
# # Main Code
#
# opt = Optimizer()
#
# s_ti = [-2000.0, -1000.0]
# v_ti = [00.0, 800.0]
# s_tf = [000.0, 000.0]
# v_tf = [00.00, 00.0]
# r_ti = -1*math.pi/2 # inital orientation of the car
# omega_ti = 0.0 # initial angular velocity of car
#
# acceleration, turning, t_star = opt.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti)
#
# # print('u', acceleration.value)
# # print('tf', opt.tf.value)
# # print('tf', opt.tf.value[0])
# print('vx', opt.vx.value)
# print('vy', opt.vy.value)
# print('a', opt.a.value)
#
# ts = opt.d.time * opt.tf.value[0]
# # plot results
# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection='3d')
# # plt.subplot(2, 1, 1)
# Axes3D.plot(ax, opt.sx.value, opt.sy.value, ts, c='r', marker ='o')
# plt.ylim(-2500, 2500)
# plt.xlim(-2500, 2500)
# plt.ylabel('Position y')
# plt.xlabel('Position x')
# ax.set_zlabel('time')
#
# fig = plt.figure(3)
# ax = fig.add_subplot(111, projection='3d')
# # plt.subplot(2, 1, 1)
# Axes3D.plot(ax, opt.vx.value, opt.vy.value, ts, c='r', marker ='o')
# plt.ylim(-2500, 2500)
# plt.xlim(-2500, 2500)
# plt.ylabel('velocity y')
# plt.xlabel('Velocity x')
# ax.set_zlabel('time')
#
# plt.figure(1)
# plt.subplot(3,1,1)
# plt.plot(ts, opt.a, 'r-')
# plt.ylabel('acceleration')
#
# plt.subplot(3,1,2)
# plt.plot(ts, np.multiply(opt.yaw, 1/math.pi), 'r-')
# plt.ylabel('turning input')
#
# plt.subplot(3, 1, 3)
# plt.plot(ts, opt.v_mag, 'b-')
# plt.ylabel('vmag')
# # plt.figure(1)
# #
# # plt.subplot(7,1,1)
# # plt.plot(ts,opt.sz.value,'r-',linewidth=2)
# # plt.ylabel('Position z')
# # plt.legend(['sz (Position)'])
# #
# # plt.subplot(7,1,2)
# # plt.plot(ts,opt.vz.value,'b-',linewidth=2)
# # plt.ylabel('Velocity z')
# # plt.legend(['vz (Velocity)'])
# #
# # # plt.subplot(4,1,3)
# # # plt.plot(ts,mass.value,'k-',linewidth=2)
# # # plt.ylabel('Mass')
# # # plt.legend(['m (Mass)'])
# #
# # plt.subplot(7,1,3)
# # plt.plot(ts,opt.u_thrust.value,'g-',linewidth=2)
# # plt.ylabel('Thrust')
# # plt.legend(['u (Thrust)'])
# #
# # plt.subplot(7,1,4)
# # plt.plot(ts,opt.sx.value,'r-',linewidth=2)
# # plt.ylabel('Position x')
# # plt.legend(['sx (Position)'])
# #
# # plt.subplot(7,1,5)
# # plt.plot(ts,opt.vx.value,'b-',linewidth=2)
# # plt.ylabel('Velocity x')
# # plt.legend(['vx (Velocity)'])
# #
# # # plt.subplot(4,1,3)
# # # plt.plot(ts,mass.value,'k-',linewidth=2)
# # # plt.ylabel('Mass')
# # # plt.legend(['m (Mass)'])
# #
# # plt.subplot(7,1,6)
# # plt.plot(ts,opt.u_pitch.value,'g-',linewidth=2)
# # plt.ylabel('Torque')
# # plt.legend(['u (Torque)'])
# #
# # plt.subplot(7,1,7)
# # plt.plot(ts,opt.pitch.value,'g-',linewidth=2)
# # plt.ylabel('Theta')
# # plt.legend(['p (Theta)'])
# #
# # plt.xlabel('Time')
#
# # plt.figure(2)
# #
# # plt.subplot(2,1,1)
# # plt.plot(opt.m.time,m.t_sx,'r-',linewidth=2)
# # plt.ylabel('traj pos x')
# # plt.legend(['sz (Position)'])
# #
# # plt.subplot(2,1,2)
# # plt.plot(opt.m.time,m.t_sz,'b-',linewidth=2)
# # plt.ylabel('traj pos z')
# # plt.legend(['vz (Velocity)'])
# # #export csv
# #
# # f = open('optimization_data.csv', 'w', newline = "")
# # writer = csv.writer(f)
# # writer.writerow(['time', 'sx', 'sz', 'vx', 'vz', 'u thrust', 'theta', 'omega_pitch', 'u pitch']) # , 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'quaternion', 'boost', 'roll', 'pitch', 'yaw'])
# # for i in range(len(opt.m.time)):
# #     row = [opt.m.time[i], opt.sx.value[i], opt.sz.value[i], opt.vx.value[i], opt.vz.value[i], opt.u_thrust.value[i], opt.pitch.value[i],
# #     opt.omega_pitch.value[i], opt.u_pitch.value[i]]
# #     writer.writerow(row)
# #     print('wrote row', row)
#
#
# plt.show()
