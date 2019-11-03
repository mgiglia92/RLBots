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
import sys
import linecache
# This optimizer will minimize time to get to desired end points

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

# Next I will need to add to the cost function the error between the rocket and the desired set point
def plotData(figNum, ts, sx, sy, vx, vy, yaw, vmag, acceleration):
    # plot results
    fig = plt.figure(figNum)
    ax = fig.add_subplot(111, projection='3d')
    # plt.subplot(2, 1, 1)
    Axes3D.plot(ax, sx, sy, ts, c='r', marker ='o')
    plt.ylabel('Position/Velocity y')
    plt.xlabel('Position/Velocity x')
    ax.set_zlabel('time')

    # fig = plt.figure(3)
    # ax = fig.add_subplot(111, projection='3d')
    # plt.subplot(2, 1, 1)
    Axes3D.plot(ax, vx, vy, ts, c='b', marker ='.')

    plt.figure(figNum+1)
    plt.subplot(3,1,1)
    plt.plot(ts, vmag, 'r-')
    plt.ylabel('v_mag')

    plt.subplot(3,1,2)
    plt.plot(ts, np.multiply(yaw, 1/math.pi), 'r-')
    plt.ylabel('yaw')

    plt.subplot(3,1,3)
    plt.plot(ts, acceleration, 'o-')
    plt.ylabel('acceleration')



class Optimizer():
    def __init__(self):
#################GROUND DRIVING OPTIMIZER SETTTINGS##############
        self.d = GEKKO(remote=False) # Driving on ground optimizer

# nt1 nt2 are how many the first half and second half of time will be split
# t1 t2 are then concatenated to make the time vector which has variable deltaT

        # nt1 = 7
        # nt2 = 5
        # t1 = np.linspace(0,0.5, nt1)
        # t2 = np.linspace(0.55, 1.0, nt2)
        # self.d.time = np.concatenate((t1,t2), axis=0)

        ntd = 21

        self.d.time = np.linspace(0, 1, ntd) # Time vector normalized 0-1

        # options
        self.d.options.NODES = 2
        self.d.options.SOLVER = 3
        self.d.options.IMODE = 6# MPC mode
        # m.options.IMODE = 9 #dynamic ode sequential
        self.d.options.MAX_ITER = 300
        self.d.options.MV_TYPE = 0
        self.d.options.DIAGLEVEL = 0

        # final time for driving optimizer
        self.tf = self.d.FV(value=1.0,lb=0.1,ub=100.0)

        # allow gekko to change the tfd value
        self.tf.STATUS = 1

        # Scaled time for Rocket league to get proper time

        # Acceleration variable
        self.a = self.d.MV(value = 1, lb = -3.5, ub = 1)
        self.a.STATUS = 1
        self.a.DCOST = 1e-1

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
        self.u_turning_d = self.d.MV(lb = -1, ub = 1)
        self.u_turning_d.STATUS = 1
        self.u_turning_d.DCOST = 1e-5

        # end time variables to multiply u2 by to get total value of integral
        # Time vector length is nt1 and nt2
        self.p_d = np.zeros(ntd)
        self.p_d[-1] = 1.0
        self.final = self.d.Param(value = self.p_d)

        # integral over time for u_pitch^2
        # self.u2_pitch = self.d.Var(value=0)
        # self.d.Equation(self.u2.dt() == 0.5*self.u_pitch**2)

    def optimize2D(self, si, sf, vi, vf, ri, omegai): #these are 1x2 vectors s or v [x, z]

        # variables intial conditions are placed here
            # Position and Velocity in 2d
        self.sy = self.d.Var(value=si[1], lb=-5120, ub=5120) #y position
        self.sx = self.d.Var(value=si[0], lb=-4096, ub=4096) #x position

        # self.sx = self.d.Var(value=0, lb=-4096, ub=4096) #x position
        # self.sy = self.d.Var(value=0, lb=-5120, ub=5120) #y position
        #Delta s variable from v_mag, maybe will quicken soltuion solving time
        self.delta_s = self.d.Var(value = 0, lb = 0)

            # Pitch rotation and angular velocity
        self.yaw = self.d.Var(value = ri) #orientation yaw angle
        self.omega = self.d.Var(value=omegai, lb=-5.5, ub=5.5) #angular velocity
        # self.v_mag = self.d.Intermediate(self.d.sqrt((self.vx**2) + (self.vy**2)))

        self.v_mag = self.d.Var(value = vi, ub = 2300, lb =0)
        self.vx = self.d.Var(value = (np.cos(ri) * vi)) #x velocity
        self.vy = self.d.Var(value = (np.sin(ri) * vi)) #y velocity

        # curvature as a polynomical approximation
        self.curvature = self.d.Intermediate(0.0069 - ((7.67e-6) * self.v_mag) + ((4.35e-9)*(self.v_mag**2)) - ((1.48e-12) * (self.v_mag**3)) + ((2.37e-16) * (self.v_mag**4)))

        # curvature as a piecewise linear function
        # self.curvature = self.d.Var()
        # self.steer = np.array([0.0069, 0.00398, 0.00235, 0.001375, 0.0011, 0.00088])
        # self.v_for_steer = np.array([0, 500, 1000, 1500, 1750, 2300])
        # self.d.pwl(self.v_mag, self.curvature, self.v_for_steer, self.steer)

        # self.v_for_throttle = np.array([0, 1400, 1410])
        # self.a_for_throttle = np.array([1600, 160, 0])
        # self.throttle_acceleration = self.d.Var(value = 1, lb = 0, ub = 1)
        # self.d.pwl(self.v_mag, self.throttle_acceleration, self.v_for_throttle, self.a_for_throttle)

#piecewise linear
        # self.v_for_throttle = [0, 1400, 1410, 2300]
        # self.a_for_throttle = [1600, 160, 0, 0]
        # self.vp = [self.d.Param(value=self.v_for_throttle[i]) for i in range(4)]
        # self.ap = [self.d.Param(value=self.a_for_throttle[i]) for i in range(4)]
        # self.throttle_acceleration = [self.d.Var(lb=self.vp[i], ub=self.vp[i+1]) for i in range(3)]



# Differental equations
        self.d.Equation(self.v_mag.dt() == self.tf * self.a * (991.666))

        # self.d.Equation(self.vx.dt() == self.tf * (self.a * (991.666+60) * self.d.cos(self.yaw)))
        # self.d.Equation(self.vy.dt() == self.tf * (self.a * (991.666+60) * self.d.sin(self.yaw)))
        # self.d.Equation(self.vx.dt()==self.tf *(self.a * ((-1600 * self.v_mag/1410) +1600) * self.d.cos(self.yaw)))
        # self.d.Equation(self.vy.dt()==self.tf *(self.a * ((-1600 * self.v_mag/1410) +1600) * self.d.sin(self.yaw)))
        # self.d.Equation(self.vx == (self.v_mag * self.d.cos(self.yaw)))
        # self.d.Equation(self.vy == (self.v_mag * self.d.sin(self.yaw)))

        # self.d.Equation(self.yaw.dt() == self.tf * ((self.u_turning_d) * self.curvature * self.v_mag))

        # position vector from velocity componenets
        # self.d.Equation(self.sx.dt() == self.tf * self.vx)
        # self.d.Equation(self.sy.dt() == self.tf * self.vy)

        # delta position vector
        # self.d.Equation(self.delta_s.dt() == self.tf * self.v_mag)



        self.d.Equation(self.omega ==  ((self.u_turning_d) * self.curvature * self.v_mag))
        self.d.Equation(self.yaw.dt() == self.tf * self.omega)
        # self.d.fix(self.sz, pos = len(self.d.time) - 1, val = 1000)

        #Get position component vectors
        self.d.Equation(self.sx.dt() == (self.tf * self.v_mag * self.d.cos(self.yaw)))
        self.d.Equation(self.sy.dt() == (self.tf * self.v_mag * self.d.sin(self.yaw)))

        #Soft constraints for the end point
        # Uncomment these 4 objective functions to get a simlple end point optimization
        #sf[1] is z position @ final time etc...
        self.d.Obj(self.final*1e8*((self.sy-sf[1])**2)) # Soft constraints
        # self.d.Obj(self.final*1e3*(self.vz-vf[1])**2)
        self.d.Obj(self.final*1e8*((self.sx-sf[0])**2)) # Soft constraints
        # self.d.Obj(self.final*1e3*(self.vx-vf[0])**2)

        #Objective function to minimize time
        self.d.Obj(self.tf*1e7)

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
        self.d.solve(disp=True)

        # NOTE: another data structure type or class here for optimal control vectors
        # Maybe it should have some methods to also make it easier to parse through the control vector etc...
        # print('time', np.multiply(self.d.time, self.tf.value[0]))
        # time.sleep(3)

        self.ts = np.multiply(self.d.time, self.tf.value[0])

        return self.a, self.u_turning_d, self.ts

    def getTrajectoryData(self):
        return [self.ts, self.sx, self.sy, self.vx, self.vy, self.yaw, self.omega]

    def getInputData(self):
        return [self.ts, self.a]

try:
    # Main Code

    opt = Optimizer()
    opt2 = Optimizer()
    opt3 = Optimizer()

# First Trajectory
    s_ti = [-2000.0, -2000.0]
    v_ti = 300.0
    s_tf = [000.0, 000.0]
    v_tf = [00.00, 00.0]
    r_ti = np.pi/2 # inital orientation of the car
    omega_ti = 0.0 # initial angular velocity of car

    t_star = opt.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti)

    ts1 = opt.ts
    sx1 = np.array(opt.sx)
    sy1 = np.array(opt.sy)
    vx1 = np.array(opt.vx)
    vy1 = np.array(opt.vy)
    yaw1 = np.array(opt.yaw)
    vmag1 = np.array(opt.v_mag)
    omega1 = np.array(opt.omega)
    acc1 = np.array(opt.a)

    # plotData(1, ts1, sx1, sy1, vx1, vy1, yaw1, vmag1)
    plt.ion()
    plt.show()
    plt.pause(0.001)

# Second Trajectory
    s_ti = [sx1[-1], sy1[-1]]
    v_ti = vmag1[-1]
    s_tf = [0, 0]
    v_tf = [0.00, 0.0]
    r_ti = yaw1[-1] # inital orientation of the car
    omega_ti = omega1[-1] # initial angular velocity of car

    print(s_ti, v_ti, s_tf, v_tf, r_ti, omega_ti)
    # s_ti = [-2000.0, -2000.0]
    # v_ti = 0.0
    # s_tf = [000.0, 000.0]
    # v_tf = [00.00, 00.0]
    # r_ti = np.pi/2 # inital orientation of the car
    # omega_ti = 0.0 # initial angular velocity of car


    t_star = opt2.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti)

    ts2 = ts1[-1] + opt2.ts
    sx2 = np.array(opt2.sx)
    sy2 = np.array(opt2.sy)
    vx2 = np.array(opt2.vx)
    vy2 = np.array(opt2.vy)
    yaw2 = np.array(opt2.yaw)
    vmag2 = np.array(opt2.v_mag)
    acc2 = np.array(opt2.a)
    omega2 = np.array(opt2.omega)
    print(vx2, '\n', vy2, '\n', vmag2, '\n', yaw2)

    # plotData(3, ts2, sx2, sy2, vx2, vy2, yaw2, vmag2)

# Thirds Trajectory
    s_ti = [sx2[-1], sy2[-1]]
    v_ti = vmag2[-1]
    s_tf = [0, -2000]
    v_tf = [0.00, 0.0]
    r_ti = yaw2[-1] # inital orientation of the car
    omega_ti = omega2[-1] # initial angular velocity of car

    print(s_ti, v_ti, s_tf, v_tf, r_ti, omega_ti)
    # s_ti = [-2000.0, -2000.0]
    # v_ti = 0.0
    # s_tf = [000.0, 000.0]
    # v_tf = [00.00, 00.0]
    # r_ti = np.pi/2 # inital orientation of the car
    # omega_ti = 0.0 # initial angular velocity of car


    t_star = opt3.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti)

    ts3 = ts2[-1] + opt3.ts
    sx3 = np.array(opt3.sx)
    sy3 = np.array(opt3.sy)
    vx3 = np.array(opt3.vx)
    vy3= np.array(opt3.vy)
    yaw3 = np.array(opt3.yaw)
    vmag3 = np.array(opt3.v_mag)
    acc3 = np.array(opt3.a)
    omega3 = np.array(opt3.omega)
    print(vx3, '\n', vy3, '\n', vmag3, '\n', yaw3)

    # plotData(3, ts2, sx2, sy2, vx2, vy2, yaw2, vmag2)

    # CONCATENATE THE TRAJECTORY DATA
    t = np.concatenate([ts1, ts2[1:], ts3[1:]])
    sx = np.concatenate([sx1, sx2[1:], sx3[1:]])
    sy = np.concatenate([sy1, sy2[1:], sy3[1:]])
    vx = np.concatenate([vx1, vx2[1:], vx3[1:]])
    vy = np.concatenate([vy1, vy2[1:], vy3[1:]])
    yaw = np.concatenate([yaw1, yaw2[1:], yaw3[1:]])
    vmag = np.concatenate([vmag1, vmag2[1:], vmag3[1:]])
    acceleration = np.concatenate([acc1, acc2[1:], acc3[1:]])
    print("t: ", t)
    plotData(5, t, sx, sy, vx, vy, yaw, vmag, acceleration)

except BaseException as e:
    PrintException()
plt.ioff()
plt.show()


# plt.figure(1)
#
# plt.subplot(7,1,1)
# plt.plot(ts,opt.sz.value,'r-',linewidth=2)
# plt.ylabel('Position z')
# plt.legend(['sz (Position)'])
#
# plt.subplot(7,1,2)
# plt.plot(ts,opt.vz.value,'b-',linewidth=2)
# plt.ylabel('Velocity z')
# plt.legend(['vz (Velocity)'])
#
# # plt.subplot(4,1,3)
# # plt.plot(ts,mass.value,'k-',linewidth=2)
# # plt.ylabel('Mass')
# # plt.legend(['m (Mass)'])
#
# plt.subplot(7,1,3)
# plt.plot(ts,opt.u_thrust.value,'g-',linewidth=2)
# plt.ylabel('Thrust')
# plt.legend(['u (Thrust)'])
#
# plt.subplot(7,1,4)
# plt.plot(ts,opt.sx.value,'r-',linewidth=2)
# plt.ylabel('Position x')
# plt.legend(['sx (Position)'])
#
# plt.subplot(7,1,5)
# plt.plot(ts,opt.vx.value,'b-',linewidth=2)
# plt.ylabel('Velocity x')
# plt.legend(['vx (Velocity)'])
#
# # plt.subplot(4,1,3)
# # plt.plot(ts,mass.value,'k-',linewidth=2)
# # plt.ylabel('Mass')
# # plt.legend(['m (Mass)'])
#
# plt.subplot(7,1,6)
# plt.plot(ts,opt.u_pitch.value,'g-',linewidth=2)
# plt.ylabel('Torque')
# plt.legend(['u (Torque)'])
#
# plt.subplot(7,1,7)
# plt.plot(ts,opt.pitch.value,'g-',linewidth=2)
# plt.ylabel('Theta')
# plt.legend(['p (Theta)'])
#
# plt.xlabel('Time')

# plt.figure(2)
#
# plt.subplot(2,1,1)
# plt.plot(opt.m.time,m.t_sx,'r-',linewidth=2)
# plt.ylabel('traj pos x')
# plt.legend(['sz (Position)'])
#
# plt.subplot(2,1,2)
# plt.plot(opt.m.time,m.t_sz,'b-',linewidth=2)
# plt.ylabel('traj pos z')
# plt.legend(['vz (Velocity)'])
# #export csv
#
# f = open('optimization_data.csv', 'w', newline = "")
# writer = csv.writer(f)
# writer.writerow(['time', 'sx', 'sz', 'vx', 'vz', 'u thrust', 'theta', 'omega_pitch', 'u pitch']) # , 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'quaternion', 'boost', 'roll', 'pitch', 'yaw'])
# for i in range(len(opt.m.time)):
#     row = [opt.m.time[i], opt.sx.value[i], opt.sz.value[i], opt.vx.value[i], opt.vz.value[i], opt.u_thrust.value[i], opt.pitch.value[i],
#     opt.omega_pitch.value[i], opt.u_pitch.value[i]]
#     writer.writerow(row)
#     print('wrote row', row)
