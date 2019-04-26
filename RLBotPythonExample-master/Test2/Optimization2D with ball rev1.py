import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import integrate as int
import scipy.linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
from gekko import GEKKO
import csv
import time

# This optimizer will minimize time to get to desired end points

# Next I will need to add to the cost function the error between the rocket and the desired set point
class Optimizer():
    def __init__(self):


        self.m = GEKKO(remote=False)
        nt = 11
        self.m.time = np.linspace(0, 1, nt)

        # options
        # self.m.options.NODES = 3
        self.m.options.SOLVER = 3
        self.m.options.IMODE = 6# MPC mode
        # m.options.IMODE = 9 #dynamic ode sequential
        self.m.options.MAX_ITER = 200
        self.m.options.MV_TYPE = 0
        self.m.options.DIAGLEVEL = 0

        # final time
        self.tf = self.m.FV(value=1.0,lb=0.1,ub=100.0)

        # tf = m.FV(value=5.0)
        self.tf.STATUS = 1

        # Scaled time for Rocket league to get proper time
        self.ts = np.multiply(self.m.time, self.tf)

        # some constants
        self.g = 650 # Gravity (my data says its more like 670)
        # v_end = 500

        # force (thruster)
        self.u_thrust = self.m.FV(value=1.0,lb=0.0,ub=1.0) #Fixed variable, stays on entire time
        # self.u_thrust = self.m.MV(value=1, lb=0, ub=1) # Manipulated variable non integaer
        # self.u_thrust = self.m.MV(value=0,lb=0,ub=1, integer=True) #Manipulated variable integer type
        # self.u_thrust.STATUS = 1
        # self.u_thrust.DCOST = 1e-5

        # angular acceleration
        self.u_pitch = self.m.MV(value=0.0, lb=-1.0, ub=1.0)
        self.u_pitch.STATUS = 1
        self.u_pitch.DCOST = 1e-5

        self.Tp = 12.14599781908070 # torque coefficient for pitch
        self.Dp = -2.798194258050845 # drag coefficient fo rpitch


        # integral over time for u^2
        self.u2 = self.m.Var(value=0.0)
        self.m.Equation(self.u2.dt() == 0.5*self.u_thrust**2)

        # integral over time for u_pitch^2
        # self.u2_pitch = self.m.Var(value=0)
        # self.m.Equation(self.u2.dt() == 0.5*self.u_pitch**2)

        # end time variables to multiply u2 by to get total value of integral
        self.p = np.zeros(nt)
        self.p[-1] = 1.0
        self.final = self.m.Param(value = self.p)

    def optimize2D(self, si, sf, vi, vf, ri, omegai, ball_si, ball_vi): #these are 1x2 vectors s or v [x, z]
        #NOTE: I should make some data structures to easily pass this data around as one variable instead of so many variables

        # variables intial conditions are placed here
        # CAR VARIABLES
        # NOTE: maximum velocites, need to be total velocity magnitude, not max on indididual axes, as you can max on both axes but actually be above the true max velocity of the game
            # Position and Velocity in 2d
        self.sx = self.m.Var(value=si[0], lb=-2500.0, ub=2500.0) #x position
        self.vx = self.m.Var(value=vi[0], lb=-1*2300, ub=2300.0) #x velocity
        self.sz = self.m.Var(value=si[1], lb = 0.0, ub = 4000.0) #z position
        self.vz = self.m.Var(value=vi[1],lb=-1*2300.0,ub=2300.0) #z velocity

            # Pitch rotation and angular velocity
        self.pitch = self.m.Var(value = ri) #orientation pitch angle
        self.omega_pitch = self.m.Var(value=omegai, lb=-5.5, ub=5.5) #angular velocity

        # BALL VARIABLES
        # NOTE: same issue with max velocity as car, will fix later
        self.ball_sx = self.m.Var(value=ball_si[0])
        self.ball_sz = self.m.Var(value=ball_si[1])
        self.ball_vx = self.m.Var(value=ball_vi[0])
        self.ball_vz = self.m.Var(value=ball_vi[1])



        # differential equations scaled by tf
        # CARS DIFFERENTIAL EQUATIONS
            #position and velocity
        self.m.Equation(self.sz.dt()==self.tf * self.vz)
        self.m.Equation(self.vz.dt()==self.tf * ((self.u_thrust*(991.666+66.666) * self.m.sin(self.pitch)) - self.g)) #testing different acceleration value that i get from data
        self.m.Equation(self.sx.dt()==self.tf * self.vx)
        self.m.Equation(self.vx.dt()==self.tf * ((self.u_thrust*(991.666+66.666) * self.m.cos(self.pitch)))) #testing different acceleration value that i get from data
            # pitch rotation
        self.m.Equation(self.pitch.dt()==self.tf * self.omega_pitch)
        self.m.Equation(self.omega_pitch.dt()== self.tf * ((self.u_pitch*self.Tp) + (self.omega_pitch*self.Dp*(1.0-self.m.sqrt(self.u_pitch*self.u_pitch)))))

        # BALLS DIFFERENTIAL EQUATIONS
        self.m.Equation(self.ball_sx.dt()==self.tf * self.ball_vx)
        self.m.Equation(self.ball_sz.dt()==self.tf * self.ball_vz)
        self.m.Equation(self.ball_vz.dt()==self.tf * (-1.0*self.g))
        self.m.Equation(self.ball_vx.dt()==self.tf * 0.0)
        # self.m.Equation(self.error == self.sx - trajectory_sx)

        # hard constraints
        # self.m.fix(self.sz, pos = len(self.m.time) - 1, val = 1000)


        #Soft constraints for the end point
        # Uncomment these 4 objective functions to get a simlple end point optimization
        #sf[1] is z position @ final time etc...
        # self.m.Obj(self.final*1e3*(self.sz-sf[1])**2) # Soft constraints
        # self.m.Obj(self.final*1e3*(self.vz-vf[1])**2)
        # self.m.Obj(self.final*1e3*(self.sx-sf[0])**2) # Soft constraints
        # self.m.Obj(self.final*1e3*(self.vx-vf[0])**2)

        # Objective values to hit into ball in minimal time
        self.m.Obj(self.final*1e4*(self.sz-self.ball_sz)**2) # Soft constraints
        self.m.Obj(self.final*1e4*(self.sx-self.ball_sx)**2) # Soft constraints

        #Objective function to minimize time
        self.m.Obj(self.tf * 1e3)

        #Objective functions to follow trajectory
        # self.m.Obj(self.final * (self.errorx **2) * 1e3)

        # self.m.Obj(self.final*1e3*(self.sx-traj_sx)**2) # Soft constraints
        # self.m.Obj(self.errorz)
        # self.m.Obj(( self.all * (self.sx - trajectory_sx) **2) * 1e3)
        # self.m.Obj(((self.sz - trajectory_sz)**2) * 1e3)

        # minimize thrust used
        # self.m.Obj(self.u2*self.final*1e3)

        # minimize torque used
        # self.m.Obj(self.u2_pitch*self.final)

        #solve
        # self.m.solve('http://127.0.0.1') # Solve with local apmonitor server
        self.m.solve()

        # NOTE: another data structure type or class here for optimal control vectors
        # Maybe it should have some methods to also make it easier to parse through the control vector etc...
        # print('time', np.multiply(self.m.time, self.tf.value[0]))
        # time.sleep(3)

        self.ts = np.multiply(self.m.time, self.tf.value[0])
        print('ts', self.ts)
        # time.sleep(0.10)
        return self.u_thrust, self.u_pitch, self.ts, self.sx, self.sz, self.ball_sx, self.ball_sz, self.ball_vz, self.pitch



# Main Code

opt = Optimizer()

# Set optimizer values for optimal control algo for car
s_ti = [2200.0, 100.0]
v_ti = [-00.0, 00.0]
s_tf = [-2200.0, 1200.0]
v_tf = [-500.00, 200.0]
r_ti = math.pi/2 # inital orientation of the car
omega_ti = 0.0 # initial angular velocity of car

# Set ball initial conditions
ball_si = [-1200.0, 200.0]
ball_vi = [0.0, 1200.0]

u_thrust_star, u_pitch_star, t_star, carx, carz, ballx, ballz, ballvz, pitch= opt.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti, ball_si, ball_vi)

print('u', u_thrust_star.value)
# print('tf', opt.tf.value)
print('tf', opt.tf.value[0])
print('ball x', opt.ball_sx.value)
print('car x', opt.sx.value)
print('ball z', opt.ball_sz.value)
print('car z', opt.sz.value)

ts = opt.m.time * opt.tf.value[0]
# plot results
plt.figure(1)

plt.subplot(9,1,1)
plt.plot(ts,opt.sz.value,'r-',linewidth=2)
plt.ylabel('Position z')
plt.legend(['sz (Position)'])

plt.subplot(9,1,2)
plt.plot(ts,opt.vz.value,'b-',linewidth=2)
plt.ylabel('Velocity z')
plt.legend(['vz (Velocity)'])

# plt.subplot(4,1,3)
# plt.plot(ts,mass.value,'k-',linewidth=2)
# plt.ylabel('Mass')
# plt.legend(['m (Mass)'])

plt.subplot(9,1,3)
plt.plot(ts,opt.u_thrust.value,'g-',linewidth=2)
plt.ylabel('Thrust')
plt.legend(['u (Thrust)'])

plt.subplot(9,1,4)
plt.plot(ts,opt.sx.value,'r-',linewidth=2)
plt.ylabel('Position x')
plt.legend(['sx (Position)'])

plt.subplot(9,1,5)
plt.plot(ts,opt.vx.value,'b-',linewidth=2)
plt.ylabel('Velocity x')
plt.legend(['vx (Velocity)'])

# plt.subplot(4,1,3)
# plt.plot(ts,mass.value,'k-',linewidth=2)
# plt.ylabel('Mass')
# plt.legend(['m (Mass)'])

plt.subplot(9,1,6)
plt.plot(ts,opt.u_pitch.value,'g-',linewidth=2)
plt.ylabel('Torque')
plt.legend(['u (Torque)'])

plt.subplot(9,1,7)
plt.plot(ts,opt.pitch.value,'g-',linewidth=2)
plt.ylabel('Theta')
plt.legend(['p (Theta)'])

plt.subplot(9,1,8)
plt.plot(ts,opt.ball_sx,'g-',linewidth=2)
plt.ylabel('ball sx')
plt.legend(['ball pos x'])

plt.subplot(9,1,9)
plt.plot(ts,opt.ball_sz,'g-',linewidth=2)
plt.ylabel('ball sz')
plt.legend(['ball pos z'])


plt.xlabel('Time')

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
f = open('optimization_data.csv', 'w', newline = "")
writer = csv.writer(f)
writer.writerow(['time', 'sx', 'sz', 'vx', 'vz', 'u thrust', 'theta', 'omega_pitch', 'u pitch', 'ball sx', 'ball sz', 'ballvx', 'ballvz']) # , 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'quaternion', 'boost', 'roll', 'pitch', 'yaw'])
for i in range(len(opt.m.time)):
    row = [opt.ts[i], opt.sx.value[i], opt.sz.value[i], opt.vx.value[i], opt.vz.value[i], opt.u_thrust.value[i], opt.pitch.value[i],
    opt.omega_pitch.value[i], opt.u_pitch.value[i], opt.ball_sx.value[i], opt.ball_sz.value[i], opt.ball_vx.value[i], opt.ball_vz.value[i]]
    writer.writerow(row)
    print('wrote row', row)


plt.show()
