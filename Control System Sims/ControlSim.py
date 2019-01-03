import control #Control system python library
import numpy as np
import time as time
from scipy import linspace
from scipy.signal import lti, step
from matplotlib import pyplot as p
#Rocket League physics (using unreal units (uu))
gravity = 650 #uu/s^2

#State Space matrix coefficients
mass = 14.2
M = 1/14.2
A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])
C = np.array([[1.0, 0.0], [0.0, 0.0]])
D = np.array([[0.0], [0.0]])
system = control.ss(A, B, C, D, None)
#print(B)
poles = np.array([-1.1, -1.0])
K = control.place(A, B, poles)
k1 = K[0, 0]
k2 = K[0, 1]
kr = 1.33333333

#u = (-k1*(z)) + (-k2 * vz) + (kr * desz) #add desz here since equation considers xequilibrium point as center
#print('u:', int(u), '/', 'z:', int(z), '/', 'vz:', int(vz), '/', int(k1), '/', int(k2))
#boostPercent = u

from scipy.signal import lti, step2

sys = control.ss(A, B, C, D, None)
t, y = step2(system)

p.plot(t, y)
time.sleep(10)
