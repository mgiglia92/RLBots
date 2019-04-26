# This optimization will use bang-bang controls (making u an integer)

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import integrate as int
import scipy.linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
import csv
from gekko import GEKKO


# create GEKKO model
m = GEKKO()

# make time from 0 to 5 seconds
nt = 21
m.time = np.linspace(0,3,nt)

# options
m.options.NODES = 100
m.options.SOLVER = 3
m.options.IMODE = 6 # MPC mode
# m.options.IMODE = 9 #dynamic ode sequential
m.options.MAX_ITER = 500
m.options.MV_TYPE = 0
m.options.DIAGLEVEL = 0

# final time
tf = m.MV(value=1.0,lb=0.0,ub=100)

# tf = m.FV(value=5.0)
tf.STATUS = 1

# some constants
g = 650 # Gravity
# v_end = 500

# force (thruster)
u_thrust = m.MV(value=1, lb=0,ub=1, integer=True)
u_thrust.STATUS = 1
u_thrust.DCOST = 1e-5

# angular acceleration
u_pitch = m.MV(value=0, lb=-1, ub=1)
u_pitch.STATUS = 1
u_thrust.DCOST = 1e-5

Tp = 12.14599781908070 # torque coefficient for pitch
Dp = -2.798194258050845 # drag coefficient fo rpitch

# variables intial conditions are placed here
    # Position and Velocity in 2d
sz = m.Var(value=100, lb = 0, ub = 4000)
vz = m.Var(value=100.0,lb=-1*2300,ub=2300)
sx = m.Var(value=0, lb=-1500, ub=1500)
vx = m.Var(value=0, lb=-1*2300, ub=2300)
    # Pitch rotation and angular velocity
pitch = m.Var(value = math.pi/2)
omega_pitch = m.Var(value=0, lb=-5.5, ub=5.5)

# integral over time for u^2
u2 = m.Var(value=0)
m.Equation(u2.dt() == 0.5*u_thrust**2 + 0.5*u_pitch**2)

# differential equations
    #position and velocity
m.Equation(sz.dt()==vz)
m.Equation(vz.dt()==((u_thrust*(991.666+60) * m.sin(pitch)) - g)) #testing different acceleration value that i get from data
m.Equation(sx.dt()==vx)
m.Equation(vx.dt()==((u_thrust*(991.666+60) * m.cos(pitch)))) #testing different acceleration value that i get from data
    # pitch rotation
m.Equation(pitch.dt()==omega_pitch)
m.Equation(omega_pitch.dt()==(u_pitch*Tp) - (Dp*(1-m.sqrt(u_pitch*u_pitch))))

# end time variables to multiply u2 by to get total value of integral
p = np.zeros(nt)
p[-1] = 1.0
final = m.Param(value = p)

# halfway cell number
nhalf = len(m.time) / 2
print(nhalf)
nhalf = round(nhalf)

# specify endpoint conditions
# m.fix(s, pos=nhalf,val=200.0) #Hard constraints, makes the derivative zero also
# m.fix(s, pos=nhalf+30,val=500.0) #Hard constraints, makes the derivative zero also

# m.fix(v, pos=nhalf,val=0.0)
m.Obj(final*1e3*(sz-1000)**2) # Soft constraints
m.Obj(final*1e3*(vz-0)**2)
m.Obj(final*1e3*(sx-300)**2) # Soft constraints
m.Obj(final*1e3*(vx-0)**2)

# minimize thrust used
# m.Obj(u2*final)
# m.Obj(tf) #final time objective

# Optimize launch
m.solve()

print('Optimal Solution (final time): ' + str(tf.value[0]))

# scaled time
ts = m.time * tf.value[0]
print(u_thrust.value)

# plot results
plt.figure(1)

plt.subplot(7,1,1)
plt.plot(ts,sz.value,'r-',linewidth=2)
plt.ylabel('Position z')
plt.legend(['sz (Position)'])

plt.subplot(7,1,2)
plt.plot(ts,vz.value,'b-',linewidth=2)
plt.ylabel('Velocity z')
plt.legend(['vz (Velocity)'])

# plt.subplot(4,1,3)
# plt.plot(ts,mass.value,'k-',linewidth=2)
# plt.ylabel('Mass')
# plt.legend(['m (Mass)'])

plt.subplot(7,1,3)
plt.plot(ts,u_thrust.value,'g-',linewidth=2)
plt.ylabel('Thrust')
plt.legend(['u (Thrust)'])

plt.subplot(7,1,4)
plt.plot(ts,sx.value,'r-',linewidth=2)
plt.ylabel('Position x')
plt.legend(['sx (Position)'])

plt.subplot(7,1,5)
plt.plot(ts,vx.value,'b-',linewidth=2)
plt.ylabel('Velocity x')
plt.legend(['vx (Velocity)'])

# plt.subplot(4,1,3)
# plt.plot(ts,mass.value,'k-',linewidth=2)
# plt.ylabel('Mass')
# plt.legend(['m (Mass)'])

plt.subplot(7,1,6)
plt.plot(ts,u_pitch.value,'g-',linewidth=2)
plt.ylabel('Torque')
plt.legend(['u (Torque)'])

plt.subplot(7,1,7)
plt.plot(ts,pitch.value,'g-',linewidth=2)
plt.ylabel('Theta')
plt.legend(['p (Theta)'])

plt.xlabel('Time')

# #export csv
#
f = open('optimization_data.csv', 'w', newline = "")
writer = csv.writer(f)
writer.writerow(['time', 'sx', 'sz', 'vx', 'vz', 'u thrust', 'theta', 'omega_pitch', 'u pitch']) # , 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'quaternion', 'boost', 'roll', 'pitch', 'yaw'])
for i in range(len(m.time)):
    row = [m.time[i], sx.value[i], sz.value[i], vx.value[i], vz.value[i], u_thrust.value[i], pitch.value[i], omega_pitch.value[i], u_pitch.value[i]]
    writer.writerow(row)
    print('wrote row', row)


plt.show()
