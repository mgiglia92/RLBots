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
################# AIRBORNE OPTIMIZER SETTINGS################
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

    def optimizeDriving(self, car, car_desired):

        # Print what values we're using
        print('Values for this optimization---------------------------------')
        print('car', car.position, 'car desired', car_desired.position)
        # Desired positions
        self.x_desired = self.d.Var(value = car_desired.x)
        self.y_desired = self.d.Var(value = car_desired.y)

        # Desired velocities
        self.vx_desired = self.d.Var(value = car_desired.vx)
        self.vy_desired = self.d.Var(value = car_desired.vy)

        # Positions of driving plane
        self.sx_d = self.d.Var(value = car.x)
        self.sy_d = self.d.Var(value = car.y)
        self.yaw = self.d.Var(value = car.yaw)

        # Velocities of driving plane
        self.v_mag = self.d.Var(value = self.d.sqrt((car.vx**2) + (car.vy**2)), lb = 0.0, ub = 2300.0)
        self.vx_d = self.d.Intermediate(self.d.cos(self.yaw) * self.v_mag)
        self.vy_d = self.d.Intermediate(self.d.sin(self.yaw) * self.v_mag)

        # Curvature of turn dependant on velocity magnitude
        self.curvature = self.d.Intermediate(0.0069 - ((7.67e-6) * self.v_mag) + ((4.35e-9)*(self.v_mag**2)) - ((1.48e-12) * (self.v_mag**3)) + ((2.37e-16) * (self.v_mag**4)))
        self.turn_radius = self.d.Intermediate(153 - (0.0713*self.v_mag) + (4.83e-4 * (self.v_mag**2)) - (1.16e-7 * (self.v_mag**3)))

        # Heading unit vector from yaw
        self.heading_x = self.d.Intermediate(self.d.cos(self.yaw))
        self.heading_y = self.d.Intermediate(self.d.sin(self.yaw))
        # Velocity direction unit vector
        self.vx_direction = self.d.Intermediate(self.vx_d/self.v_mag)
        self.vy_direction = self.d.Intermediate(self.vy_d/self.v_mag)
        # Direction car is moving relative to heading vector, this will help determine what the throttle value will do in the current state
        self.v_relative = self.d.Intermediate((self.heading_x*self.vx_direction) + (self.heading_y*self.vy_direction))
        self.throttle_relative = self.d.Intermediate(self.v_relative * self.u_throttle_d)

        self.throttle_acceleration = self.d.Intermediate((self.u_throttle_d * ((-1600 * self.v_mag/1410) + 1600)) >= 0.0)
        self.thrust_acceleration = self.d.Intermediate(self.u_thrust_d * 991.667)

        # Differental equations
        self.d.Equation(self.sx_d.dt()==self.tf_d * self.vx_d)
        self.d.Equation(self.sy_d.dt()==self.tf_d *self.vy_d)
        self.d.Equation(self.v_mag.dt()==self.tf_d * (self.throttle_acceleration + self.thrust_acceleration))
        # self.d.Equation(self.v_mag.dt()==self.tf_d * (self.u_thrust_d * 1060.0))
        # self.d.Equation(self.vx_d.dt()==self.tf_d *(self.u_throttle_d * ((-1600 * self.v_mag/1410) +1600) * self.d.cos(self.yaw)))
        # self.d.Equation(self.vy_d.dt()==self.tf_d *(self.u_throttle_d * ((-1600 * self.v_mag/1410) +1600) * self.d.sin(self.yaw)))
        self.d.Equation(self.yaw.dt()==self.tf_d *(self.u_turning_d * (self.curvature) * self.v_mag))

        self.d.Equation(self.x_desired.dt() == 0.0)
        self.d.Equation(self.y_desired.dt() == 0.0)

        # Objective functions
        self.d.Obj(self.final_d * 1e8 * (self.sx_d - self.x_desired)**2)
        self.d.Obj(self.final_d * 1e8 * (self.sy_d - self.y_desired)**2)
        # self.d.Obj( 1e8 * ((self.final_d * self.sx_d) - car_desired.x)**2)
        # self.d.Obj( 1e8 * ((self.final_d * self.sy_d) - car_desired.y)**2)
        # self.d.Obj(self.final_d * 1e4 * (self.vx_d - self.vx_desired)**2)
        # self.d.Obj(self.final_d * 1e4 * (self.vy_d - self.vy_desired)**2)
        self.d.Obj(self.tf_d * 1e4)

        self.d.solve()
        self.ts_d = np.multiply(self.d.time, self.tf_d.value[0])

        # solving complete reset flag
        self.solving = False

        return self.ts_d

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

        # Objective funciton to hit with a particular velocity
        # self.m.Obj(self.final*1e3*(self.vz/)**2)
        self.m.Obj(self.final*1e4*(self.vx + 1000)**2)
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
        print('ustar', self.u_pitch.value)
        # time.sleep(0.10)
        return self.u_thrust, self.u_pitch, self.ts, self.sx, self.sz, self.ball_sx, self.ball_sz, self.ball_vz, self.pitch


    def optimize3D(self, car, ball):
        #stuff where
        stuff = 0

    def MPC_optimize(self, car, ball):
        #NOTE: I should make some data structures to easily pass this data around as one variable instead of so many variables

        # variables intial conditions are placed here
        # CAR VARIABLES
        # NOTE: maximum velocites, need to be total velocity magnitude, not max on indididual axes, as you can max on both axes but actually be above the true max velocity of the game
            # Position and Velocity in 2d
        self.sx = self.m.Var(value=car.x, lb=-2500.0, ub=2500.0) #x position
        self.vx = self.m.Var(value=car.vx, lb=-1*2300, ub=2300.0) #x velocity
        self.sz = self.m.Var(value=car.z, lb = 0.0, ub = 4000.0) #z position
        self.vz = self.m.Var(value=car.vz,lb=-1*2300.0,ub=2300.0) #z velocity

            # Pitch rotation and angular velocity
        self.pitch = self.m.Var(value = car.pitch) #orientation pitch angle
        self.omega_pitch = self.m.Var(value=car.wy, lb=-5.5, ub=5.5) #angular velocity

        # BALL VARIABLES
        # NOTE: same issue with max velocity as car, will fix later
        self.ball_sx = self.m.Var(value=ball.x)
        self.ball_sz = self.m.Var(value=ball.z)
        self.ball_vx = self.m.Var(value=ball.vx)
        self.ball_vz = self.m.Var(value=ball.vz)



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
car.yaw = 0

car_desired = Car()
car_desired.x = -1000.0
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

opt.optimizeDriving(car, car_desired)
# pi = np.array([100, 100])
# opt.MPC_optimize(car, ball)

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
