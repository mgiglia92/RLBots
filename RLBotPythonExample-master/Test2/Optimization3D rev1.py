# This is the file that I can use to show the plots using python only (without Rocket league running)

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
import threading
import controller as con
import queue
import CoordinateSystems
from pyquaternion import Quaternion

def euler_to_quaternion(yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

def rotation_by_quaternion(quaternion):
    ux = np.array([1,0,0])
    q0 = quaternion[0].value
    q1 = quaternion[1].value
    q2 = quaternion[2].value
    q3 = quaternion[3].value
    print(q1)
    q = Quaternion(np.array([q0, q1, q2, q3]))
    direction_vector = q.rotate(ux)
    return direction_vector

def get_thrust_direction_x(q): #Assuming vector to be rotated is [1,0,0]
    x = (1 - (2*q[2]**2) - (2*q[3]**2))
def get_thrust_direction_y(q):
    y =(2*(q[1]*q[2] - q[0]*q[3]))
def get_thrust_direction_z(q):
    z = (2*(q[1]*q[3] + q[0]*q[2]))

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

        self.quaternion = None #Orientation quaterion
        self.quaternion_dt = None #Orientation quaternion derivative (from angular velocity)

        #angular velocities
        self.wx = None
        self.wy = None
        self.wz = None
        self.position = np.array([0,0,0])
        self.velocity = np.array([0,0,0])
        self.angular_velocity = np.array([0,0,0])
        self.attitude = np.array([0,0,0])
        #MUST CHECK THSESE TO MAKE SURE THEY CORRELATE PROPERLY

        self.is_demolished = False
        self.has_wheel_contact = True
        self.is_super_sonic = False
        self.jumped = False
        self.boost_left = '100'
        self.double_jumped = 'False'
# This optimizer will minimize time to get to desired end points

# Next I will need to add to the cost function the error between the rocket and the desired set point
class Optimizer():
    def __init__(self):
        # Queue for data thread safe
        self.data = queue.LifoQueue()
        self.control_data = queue.LifoQueue()

        # Create thread lock
        # self.lock = threading.Lock()

        # stop thread variable
        self.stop = False

        # variable to prevent multiple solutions form running
        self.solving = False

        # Create thread parameters
        self.MPC_thread = threading.Thread(target=self.run, args=(), daemon=True)

        # Controller state to pass into get_output function
        self.controllerState = con.Controller()

        # Optimal Control data updated by thread
        self.u_pitch_star = 0.0
        self.u_star = 0.0


        # time.sleep(1)
        # self.MPC_queue.put('worked')

        # Initialize optimization parameters
        self.initialize_optimization()


    def initialize_optimization(self):
################# AIRBORNE OPTIMIZER SETTINGS################ NOW IN 3D
        self.m = GEKKO(remote=False) # Airborne optimzer

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
        self.u_thrust = self.m.MV(value=1, lb=0, ub=1) # Manipulated variable non integaer
        # self.u_thrust = self.m.MV(value=0,lb=0,ub=1, integer=True) #Manipulated variable integer type
        self.u_thrust.STATUS = 1
        self.u_thrust.DCOST = 1e-5

        # angular acceleration for all 3 axes
        self.u_pitch = self.m.MV(value=0.0, lb=-1.0, ub=1.0)
        self.u_pitch.STATUS = 1
        self.u_pitch.DCOST = 1e-5

        self.u_roll = self.m.MV(value=0.0, lb=-1.0, ub=1.0)
        self.u_roll.STATUS = 1
        self.u_roll.DCOST = 1e-5

        self.u_yaw = self.m.MV(value=0.0, lb=-1.0, ub=1.0)
        self.u_yaw.STATUS = 1
        self.u_yaw.DCOST = 1e-5

        self.Tr = -36.07956616966136; # torque coefficient for roll
        self.Tp = -12.14599781908070; # torque coefficient for pitch
        self.Ty =   8.91962804287785; # torque coefficient for yaw
        self.Dr =  -4.47166302201591; # drag coefficient for roll
        self.Dp = -2.798194258050845; # drag coefficient for pitch
        self.Dy = -1.886491900437232; # drag coefficient for yaw

        # # integral over time for u^2
        # self.u2 = self.m.Var(value=0.0)
        # self.m.Equation(self.u2.dt() == 0.5*self.u_thrust**2)
        #
        # # integral over time for u_pitch^2
        # # self.u2_pitch = self.m.Var(value=0)
        # # self.m.Equation(self.u2.dt() == 0.5*self.u_pitch**2)
        
        # end time variables to multiply u2 by to get total value of integral
        self.p = np.zeros(nt)
        self.p[-1] = 1.0
        self.final = self.m.Param(value = self.p)


#################GROUND DRIVING OPTIMIZER SETTTINGS##############
        self.d = GEKKO(remote=False) # Driving on ground optimizer

        ntd = 9
        self.d.time = np.linspace(0, 1, ntd) # Time vector normalized 0-1

        # options
        # self.d.options.NODES = 2
        self.d.options.SOLVER = 1
        self.d.options.IMODE = 6# MPC mode
        # self.d.options.IMODE = 9 #dynamic ode sequential
        self.d.options.MAX_ITER = 500
        self.d.options.MV_TYPE = 0
        self.d.options.DIAGLEVEL = 0

        # final time for driving optimizer
        self.tf_d = self.d.FV(value=1.0,lb=0.1,ub=100.0)

        # allow gekko to change the tfd value
        self.tf_d.STATUS = 1

        # Scaled time for Rocket league to get proper time


        # Boost variable, its integer type since it can only be on or off
        self.u_thrust_d = self.d.MV(integer = True, lb=0,ub=1) #Manipulated variable integer type
        self.u_thrust_d.STATUS = 1
        # self.u_thrust_d.DCOST = 1e-5

        # Throttle value, this can vary smoothly between 0-1
        self.u_throttle_d = self.d.MV(value = 1, lb = 0.1, ub = 1)
        self.u_throttle_d.STATUS = 1
        self.u_throttle_d.DCOST = 1e-5

        # Turning input value also smooth
        self.u_turning_d = self.d.MV(value = 0, lb = -1, ub = 1)
        self.u_turning_d.STATUS = 1
        self.u_turning_d.DCOST = 1e-5

        # end time variables to multiply u2 by to get total value of integral
        self.p_d = np.zeros(ntd)
        self.p_d[-1] = 1.0
        self.final_d = self.d.Param(value = self.p_d)

    def MPC_optimize(self, car, ball):
        #NOTE: I should make some data structures to easily pass this data around as one variable instead of so many variables

        # variables intial conditions are placed here
        # CAR VARIABLES
        # NOTE: maximum velocites, need to be total velocity magnitude, not max on indididual axes, as you can max on both axes but actually be above the true max velocity of the game
#--------------------------------
        # Position of car vector
        self.s = self.m.Array(self.m.Var,(3))
        ig = [car.x,car.y, car.z]

        #Initialize values for each array element
        i = 0
        for xi in self.s:
            xi.value = ig[i]
            xi.lower = -2300.0
            xi.upper = 2300
            i += 1
#--------------------------------
        #velocity of car vector
        self.v = self.m.Array(self.m.Var,(3))
        ig = [car.vx,car.vy, car.vz]

        #Initialize values for each array element
        i = 0
        for xi in self.v:
            xi.value = ig[i]
            xi.lower = -2300.0
            xi.upper = 2300
            i += 1

        # Pitch rotation and angular velocity
        # self.roll = self.m.Var(value = car.roll)
        # self.pitch = self.m.Var(value = car.pitch) #orientation pitch angle
        # self.yaw = self.m.Var(value = car.yaw)
        # self.omega_roll = self.m.Var(value = car.wx, lb = -5.5, ub = 5.5)
        # self.omega_pitch = self.m.Var(value=car.wy, lb=-5.5, ub=5.5) #angular velocity
        # self.omega_yaw = self.m.Var(value = car.wz, lb = -5.5, ub = 5.5)
#--------------------------------
        #Orientation Quaternion
        self.q = self.m.Array(self.m.Var, (4))
        q = euler_to_quaternion(car.roll, car.pitch, car.yaw)
        ig = [q[0], q[1], q[2], q[3]]

        #Initialize values for each array element
        i = 0
        for xi in self.q:
            xi.value = ig[i]
            i += 1
#--------------------------------
        #Angular Velocity quaternion
        self.q_dt = self.m.Array(self.m.Var, (4))

        #Initialize values for each array element
        ig = [0, car.wx, car.wy, car.wz]

        #Initialize values for each array element
        i = 0
        for xi in self.q_dt:
            xi.value = ig[i]
            i += 1
#--------------------------------
        #Thrust direction vector from quaternion
        self.thrust_direction_x = self.m.Var(value = get_thrust_direction_x(self.q))
        self.thrust_direction_y = self.m.Var(value = get_thrust_direction_y(self.q))
        self.thrust_direction_z = self.m.Var(value = get_thrust_direction_z(self.q))

        #Rworld_to_car
        # r = self.roll #rotation around roll axis to get world to car frame
        # p = self.pitch #rotation around pitch axis to get world to car frame
        # y = -1*self.yaw #rotation about the world z axis to get world to car frame
        # self.Rx = np.matrix([[1, 0, 0], [0, math.cos(r), -1*math.sin(r)], [0, math.sin(r), math.cos(r)]])
        # self.Ry = np.matrix([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-1*math.sin(p), 0, math.cos(p)]])
        # self.Rz = np.matrix([[math.cos(y), -1*math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        # #Order of rotations from world to car is x then y then z
        # self.Rinter = np.matmul(self.Rx, self.Ry)
        # self.Rworld_to_car = np.matmul(self.Rinter, self.Rz)
        # self.q = Quaternion(matrix = self.Rworld_to_car).normalised #orientation quaternion created from rotation matrix derived from euler qngle "sensor"
        # self.qi = self.q.inverse

        # BALL VARIABLES
        # NOTE: same issue with max velocity as car, will fix later
        self.ball_s = self.m.Array(self.m.Var,(3)) #Ball position
        self.ball_s[0].value = ball.x
        self.ball_s[1].value = ball.y
        self.ball_s[2].value = ball.z

        self.ball_v = self.m.Array(self.m.Var,(3)) #Ball Velocity
        self.ball_v[0].value = ball.vx
        self.ball_v[1].value = ball.vy
        self.ball_v[2].value = ball.vz

        # differential equations scaled by tf
        thrust = np.array([991.666+66.666, 0, 0]) #Thrust lies along the car's local x axis only, so need to rotate this by quaternio car is rotated by to get thrust in each axis
        q = np.array([self.q[0],self.q[1],self.q[2]])
        # qi = np.array([self.qi[0],self.qi[1],self.qi[2]])
        # print(self.q.rotate(t))
        # self.thrust_direction = self.m.Array(self.m.Intermediate(self.u_thrust.value * self.q[0]))
        # self.thrust_direction_y = self.m.Intermediate(self.q.rotate(thrust)[1])
        # self.thrust_direction_z = self.m.Intermediate(self.q.rotate(t)[2])  #Intermediate thrust direction vector rotated by the quaternion


        # CARS DIFFERENTIAL EQUATIONS
            #angular orientatio quaternion and angular velocity quaternion

            #position and velocity
        self.m.Equation(self.s[0].dt()==self.tf * self.v[0])
        self.m.Equation(self.s[1].dt()==self.tf * self.v[1])
        self.m.Equation(self.s[2].dt()==self.tf * self.v[2])


        self.m.Equation(self.v[0].dt()==self.tf * ((self.u_thrust*self.thrust_direction_x)))
        self.m.Equation(self.v[1].dt()==self.tf * ((self.u_thrust*self.thrust_direction_y)))
        self.m.Equation(self.v[2].dt()==self.tf * ((self.u_thrust*self.thrust_direction_z)))

        # self.m.Equation(self.pitch.dt()==self.tf * self.omega_pitch)
        # self.m.Equation(self.omega_pitch.dt()== self.tf * ((self.u_pitch*self.Tp) + (self.omega_pitch*self.Dp*(1.0-self.m.sqrt(self.u_pitch*self.u_pitch)))))

        # BALLS DIFFERENTIAL EQUATIONS
        self.m.Equation(self.ball_s[0].dt()==self.tf * self.ball_v[0])
        self.m.Equation(self.ball_s[1].dt()==self.tf * self.ball_v[1])
        self.m.Equation(self.ball_s[2].dt()==self.tf * self.ball_v[2])

        self.m.Equation(self.ball_v[0].dt()==self.tf * 0)
        self.m.Equation(self.ball_v[1].dt()==self.tf * 0)
        self.m.Equation(self.ball_v[2].dt()==self.tf * (-1.0*self.g))

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
        self.m.Obj(self.final*1e4*(self.s[2]-self.ball_s[2])**2) # Soft constraints
        self.m.Obj(self.final*1e4*(self.s[1]-self.ball_s[1])**2) # Soft constraints
        self.m.Obj(self.final*1e4*(self.s[0]-self.ball_s[0])**2) # Soft constraints

        # Objective funciton to hit with a particular velocity
        # self.m.Obj(self.final*1e3*(self.vz/)**2)
        # self.m.Obj(self.final*1e4*(self.vx + 1000)**2)
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

        # self.ts = np.multiply(self.m.time, self.tf.value[0])
        # print('ts', self.ts)
        # print('ustar', self.u_pitch.value)
        # time.sleep(0.10)
        return self.u_thrust, self.u_pitch#, self.ts, self.sx, self.sz, self.ball_sx, self.ball_sz, self.ball_vz, self.pitch



    def run(self):
        while(self.stop == False):
            try:
                print('in run function')
                [car, car_desired] = self.data.get()
                print('car actual', car.position, 'car desired', car_desired.position)

                if(car != None and self.solving == False):
                    self.solving = True
                    self.optimizeDriving(copy.deepcopy(car), copy.deepcopy(car_desired))
                    print('t', self.ts_d, 'u_turn', self.u_turning_d.value)

                    #Push data to control data queue
                    self.control_data.put([self.ts_d, self.u_throttle_d, self.u_turning_d, self.u_thrust_d])

                    # Save local control vector and time that the control vector should start
            except Exception as e:
                print('Exception in optimization thread', e)
                traceback.print_exc()



# Main Code
opt = Optimizer()

# Set car and car desired
car = Car()
car.x = -1500.0
car.y = 2500.0
car.z = 17.0
car.vx = 0.0
car.vy = 0.0
car.vz = 0.0
car.roll = 0
car.pitch = math.pi/2
car.yaw = 0
car.wx = 0
car.wy = 0
car.wz = 0
car.quaternion = euler_to_quaternion(car.yaw, car.pitch, car.roll)
car.quaternion_dt = np.array([0, car.wx, car.wy, car.wz])

car_desired = Car()
car_desired.x = 1000.0
car_desired.y = -1000.0
car_desired.vx = 1000
car_desired.vy = 1000

ball = Ball()
ball.x = 1000
ball.y = 1000
ball.z = 1000
ball.vx = 100
ball.vy = 100
ball.vz = 100

# opt.optimizeDriving(car, car_desired)
# pi = np.array([100, 100])
opt.MPC_optimize(car, ball)

print('u throttle', opt.u_throttle_d.value)
print('tf_d', opt.tf_d.value)
# print('ts_d', opt.ts_d)

# opt = Optimizer()
#
# # Set optimizer values for optimal control algo for car
# s_ti = [2200.0, 100.0]
# v_ti = [-00.0, 00.0]
# s_tf = [-2200.0, 1200.0]
# v_tf = [-500.00, 200.0]
# r_ti = math.pi/2 # inital orientation of the car
# omega_ti = 0.0 # initial angular velocity of car
#
# # Set ball initial conditions
# ball_si = [-1200.0, 200.0]
# ball_vi = [0.0, 1200.0]
#
# u_thrust_star, u_pitch_star, t_star, carx, carz, ballx, ballz, ballvz, pitch= opt.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti, ball_si, ball_vi)
#
# print('u', u_thrust_star.value)
# # print('tf', opt.tf.value)
# print('tf', opt.tf.value[0])
# print('ball x', opt.ball_sx.value)
# print('car x', opt.sx.value)
# print('ball z', opt.ball_sz.value)
# print('car z', opt.sz.value)
#
ts = opt.m.time * opt.tf.value
# plot results
plt.figure(1)

plt.subplot(8,1,1)
plt.plot(ts,opt.sx.value,'r-',linewidth=2)
plt.ylabel('Position x')
plt.legend(['sx (Position)'])

plt.subplot(8,1,2)
plt.plot(ts,opt.vx.value,'b-',linewidth=2)
plt.ylabel('Velocity magnitude')
plt.legend(['v mag (Velocity)'])

# plt.subplot(4,1,3)
# plt.plot(ts,mass.value,'k-',linewidth=2)
# plt.ylabel('Mass')
# plt.legend(['m (Mass)'])

plt.subplot(8,1,3)
plt.plot(ts,opt.u_thrust.value,'g-',linewidth=2)
plt.ylabel('Throttle')
plt.legend(['u (Throttle)'])

plt.subplot(8,1,4)
plt.plot(ts,opt.sz.value,'r-',linewidth=2)
plt.ylabel('Position y')
plt.legend(['sy (Position)'])

plt.subplot(8,1,5)
plt.plot(ts,opt.vz.value,'b-',linewidth=2)
plt.ylabel('Velocity y')
plt.legend(['vy (Velocity)'])

# plt.subplot(4,1,3)
# plt.plot(ts,mass.value,'k-',linewidth=2)
# plt.ylabel('Mass')
# plt.legend(['m (Mass)'])

plt.subplot(8,1,6)
plt.plot(ts,opt.u_thrust.value,'g-',linewidth=2)
plt.ylabel('Turning')
plt.legend(['u (Turning)'])

plt.subplot(8,1,7)
plt.plot(ts,opt.u_thrust.value,'g-',linewidth=2)
plt.ylabel('yaw')
plt.legend(['y (Tyaw)'])

plt.subplot(8,1,8)
plt.plot(ts,opt.u_thrust.value,'g-',linewidth=2)
plt.ylabel('thrust')
plt.legend(['thrust'])

# plt.subplot(9,1,8)
# plt.plot(ts,opt.yaw.value,'g-',linewidth=2)
# plt.ylabel('ball sx')
# plt.legend(['ball pos x'])
#
# plt.subplot(9,1,9)
# plt.plot(ts,opt.yaw.value,'g-',linewidth=2)
# plt.ylabel('ball sz')
# plt.legend(['ball pos z'])


plt.xlabel('Time')
# plt.ylim(-1500, 1500)
plt.autoscale(False)

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
# writer.writerow(['time', 'sx', 'sz', 'vx', 'vz', 'u thrust', 'theta', 'omega_pitch', 'u pitch', 'ball sx', 'ball sz', 'ballvx', 'ballvz']) # , 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'quaternion', 'boost', 'roll', 'pitch', 'yaw'])
# for i in range(len(opt.m.time)):
#     row = [opt.ts[i], opt.sx.value[i], opt.sz.value[i], opt.vx.value[i], opt.vz.value[i], opt.u_thrust.value[i], opt.pitch.value[i],
#     opt.omega_pitch.value[i], opt.u_pitch.value[i], opt.ball_sx.value[i], opt.ball_sz.value[i], opt.ball_vx.value[i], opt.ball_vz.value[i]]
#     writer.writerow(row)
#     print('wrote row', row)
#
#
plt.show()
