import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import integrate as int
import scipy.linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
from gekko import GEKKO
import csv
from pyquaternion import Quaternion



class Optimizer():
    def __init__(self):
        self.m = GEKKO(remote=False)
        nt = 11
        self.m.time = np.linspace(0, 3, nt)

        # options
        # self.m.options.NODES = 10 #When i remove this the solution is 4x faster
        self.m.options.SOLVER = 1
        self.m.options.IMODE = 6 # MPC mode
        # m.options.IMODE = 9 #dynamic ode sequential
        self.m.options.MAX_ITER = 100
        self.m.options.MV_TYPE = 0
        self.m.options.DIAGLEVEL = 0

        # final time
        self.tf = self.m.MV(value=1.0,lb=0.0,ub=100)

        # tf = m.FV(value=5.0)
        self.tf.STATUS = 1

        # some constants
        self.g = 650 # Gravity
        # v_end = 500

        # force (thruster)
        self.u_thrust = self.m.FV(value=1,lb=0,ub=1) #Fixed variable, stays on entire time
        # self.u_thrust = self.m.MV(value=1, lb=0, ub=1) # Manipulated variable non integaer
        # self.u_thrust = self.m.MV(value=0,lb=0,ub=1, integer=True) #Manipulated variable integer type
        # self.u_thrust.STATUS = 1
        # self.u_thrust.DCOST = 1e-5

        # angular acceleration
        self.u_pitch = self.m.MV(value=0, lb=-1, ub=1)
        self.u_pitch.STATUS = 1
        self.u_pitch.DCOST = 1e-5

        self.Tp = 12.14599781908070 # torque coefficient for pitch
        self.Dp = -2.798194258050845 # drag coefficient fo rpitch


        # integral over time for u^2
        self.u2 = self.m.Var(value=0)
        self.m.Equation(self.u2.dt() == 0.5*self.u_thrust**2)

        # integral over time for u_pitch^2
        # self.u2_pitch = self.m.Var(value=0)
        # self.m.Equation(self.u2.dt() == 0.5*self.u_pitch**2)

        # end time variables to multiply u2 by to get total value of integral
        self.p = np.zeros(nt)
        self.p[-1] = 1.0
        self.final = self.m.Param(value = self.p)

    def optimize(self, s_ti, s_tf, v_ti, v_tf):
        # variables intial conditions are placed here
        self.s = self.m.Var(value=s_ti, lb = 0, ub = 4000)
        self.v = self.m.Var(value=v_ti,lb=-1*2400,ub=2400)

        # differential equations
        self.m.Equation(self.s.dt()==self.v)
        # self.m.Equation(self.v.dt()==((self.u*991.666) - self.g))
        self.m.Equation(self.v.dt()==((self.u_thrust*(991.666+60)) - self.g)) #testing different acceleration value that i get from data

        #Set constraints
        # specify endpoint conditions
        self.m.Obj(self.final*1e3*(self.s-s_tf)**2) # Soft constraints
        self.m.Obj(self.final*1e3*(self.v-v_tf)**2)

        # minimize thrust used
        # self.m.Obj(self.u2*self.final)

        #solve
        self.m.solve()

        #

        return self.u_thrust, self.m.time

    def optimize2D(self, si, sf, vi, vf, ri, omegai): #these are 1x2 vectors s or v [x, z]
        #NOTE: I should make some data structures to easily pass this data around as one variable instead of so many variables

        # variables intial conditions are placed here
            # Position and Velocity in 2d
        self.sz = self.m.Var(value=si[1], lb = 0, ub = 4000) #z position
        self.vz = self.m.Var(value=vi[1],lb=-1*2300,ub=2300) #z velocity
        self.sx = self.m.Var(value=si[0], lb=-1500, ub=1500) #x position
        self.vx = self.m.Var(value=vi[0], lb=-1*2300, ub=2300) #x velocity
            # Pitch rotation and angular velocity
        self.pitch = self.m.Var(value = ri) #orientation pitch angle
        self.omega_pitch = self.m.Var(value=omegai, lb=-5.5, ub=5.5) #angular velocity


        # differential equations
            #position and velocity
        self.m.Equation(self.sz.dt()==self.vz)
        self.m.Equation(self.vz.dt()==((self.u_thrust*(991.666+60) * self.m.sin(self.pitch)) - self.g)) #testing different acceleration value that i get from data
        self.m.Equation(self.sx.dt()==self.vx)
        self.m.Equation(self.vx.dt()==((self.u_thrust*(991.666+60) * self.m.cos(self.pitch)))) #testing different acceleration value that i get from data
            # pitch rotation
        self.m.Equation(self.pitch.dt()==self.omega_pitch)
        self.m.Equation(self.omega_pitch.dt()==(self.u_pitch*self.Tp) + (self.omega_pitch*self.Dp*(1-self.m.sqrt(self.u_pitch*self.u_pitch))))
        # self.m.Equation(self.omega_pitch.dt()==(self.u_pitch*self.Tp) + (self.Dp))


        #Soft constraints for the end point
        #sf[1] is z position @ final time etc...
        self.m.Obj(self.final*1e3*(self.sz-sf[1])**2) # Soft constraints
        self.m.Obj(self.final*1e3*(self.vz-vf[1])**2)
        self.m.Obj(self.final*1e3*(self.sx-sf[0])**2) # Soft constraints
        self.m.Obj(self.final*1e3*(self.vx-vf[0])**2)

        # minimize thrust used
        # self.m.Obj(self.u2*self.final*1e3)

        # minimize torque used
        # self.m.Obj(self.u2_pitch*self.final)

        #solve
        # self.m.solve('http://127.0.0.1') # Solve with local apmonitor server (need to install apache, php, and apm on computer to do this)
        self.m.solve() # Solve with public apmonitor servers

        # NOTE: another data structure type or class here for optimal control vectors
        # Maybe it should have some methods to also make it easier to parse through the control vector etc...
        return self.u_thrust, self.u_pitch, self.m.time


m = Optimizer()


s_ti = [0.0, 100.0]
v_ti = [0.0, 100.0]
s_tf = [500.0, 1000.0]
v_tf = [500.00, 300.0]
r_ti = math.pi/2 # inital orientation of the car
omega_ti = 0.0 # initial angular velocity of car
u_thrust_star, u_pitch_star, t_star = m.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti)

print(u_thrust_star.value)

# plot results
plt.figure(1)

plt.subplot(7,1,1)
plt.plot(m.m.time,m.sz.value,'r-',linewidth=2)
plt.ylabel('Position z')
plt.legend(['sz (Position)'])

plt.subplot(7,1,2)
plt.plot(m.m.time,m.vz.value,'b-',linewidth=2)
plt.ylabel('Velocity z')
plt.legend(['vz (Velocity)'])

# plt.subplot(4,1,3)
# plt.plot(ts,mass.value,'k-',linewidth=2)
# plt.ylabel('Mass')
# plt.legend(['m (Mass)'])

plt.subplot(7,1,3)
plt.plot(m.m.time,m.u_thrust.value,'g-',linewidth=2)
plt.ylabel('Thrust')
plt.legend(['u (Thrust)'])

plt.subplot(7,1,4)
plt.plot(m.m.time,m.sx.value,'r-',linewidth=2)
plt.ylabel('Position x')
plt.legend(['sx (Position)'])

plt.subplot(7,1,5)
plt.plot(m.m.time,m.vx.value,'b-',linewidth=2)
plt.ylabel('Velocity x')
plt.legend(['vx (Velocity)'])

# plt.subplot(4,1,3)
# plt.plot(ts,mass.value,'k-',linewidth=2)
# plt.ylabel('Mass')
# plt.legend(['m (Mass)'])

plt.subplot(7,1,6)
plt.plot(m.m.time,m.u_pitch.value,'g-',linewidth=2)
plt.ylabel('Torque')
plt.legend(['u (Torque)'])

plt.subplot(7,1,7)
plt.plot(m.m.time,m.pitch.value,'g-',linewidth=2)
plt.ylabel('Theta')
plt.legend(['p (Theta)'])

plt.xlabel('Time')

# #export csv
#
f = open('optimization_data.csv', 'w', newline = "")
writer = csv.writer(f)
writer.writerow(['time', 'sx', 'sz', 'vx', 'vz', 'u thrust', 'theta', 'omega_pitch', 'u pitch']) # , 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'quaternion', 'boost', 'roll', 'pitch', 'yaw'])
for i in range(len(m.m.time)):
    row = [m.m.time[i], m.sx.value[i], m.sz.value[i], m.vx.value[i], m.vz.value[i], m.u_thrust.value[i], m.pitch.value[i],
    m.omega_pitch.value[i], m.u_pitch.value[i]]
    writer.writerow(row)
    print('wrote row', row)


plt.show()
