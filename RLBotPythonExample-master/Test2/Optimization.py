import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import integrate as int
import scipy.linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
from gekko import GEKKO
import time
import threading
import traceback

class Optimizer():
    def __init__(self):

        # Create thread lock
        self.lock = threading.Lock()

        # Create thread parameters
        self.MPC_thread = threading.Thread(target=self.run, args=(), daemon=True)

        # Current Game state data
        self.currentBall = None
        self.currentCar = None

        # Optimal Control data updated by thread
        self.u_pitch_star = 0.0
        self.u_star = 0.0

        # Initialize optimization parameters
        self.initialize_optimization()


    def initialize_optimization(self):
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
        return self.u_thrust, self.u_pitch#, self.ts, self.sx, self.sz, self.ball_sx, self.ball_sz, self.ball_vz, self.pitch



    def run(self):
        print('yay thread is running')
        try:
            if(self.current_packet != None):
                while(1):
                    try:
                        # Lock variable to allow reading
                        self.lock.acquire()

                        # Copy data locally
                        ball = copy.deepycopy(self.currentBall)
                        car = copy.deepcopy(self.currentCar)

                        # Unlock variable to allow writing
                        self.lock.release()
                        break
                    except Exception as e:
                        print('Exception when locking to read from current data', e)

                # Run optimization function
                u, u_pitch = self.MPC_optimize(car, ball)

                while(1):
                    try:
                        # Lock to write to local data
                        self.lock.acquire()

                        # Copy data locally
                        self.u_star = u
                        self.u_pitch_star = u_pitch

                        # Unlock variable to allow reading
                        self.lock.release()
                        break

                    except Exception as e:
                        print('Exception when locking to write from current data', e)



                # Save local control vector and time that the control vector should start
        except Exception as e:
            print('Exception in optimization thread', e)
            traceback.print_exc()
