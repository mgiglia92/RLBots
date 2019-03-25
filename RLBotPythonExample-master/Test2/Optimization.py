import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import integrate as int
import scipy.linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
from gekko import GEKKO



class Optimizer():
    def __init__(self):
        self.m = GEKKO()
        nt = 101
        self.m.time = np.linspace(0, 5, nt)

        # options
        self.m.options.NODES = 100
        self.m.options.SOLVER = 3
        self.m.options.IMODE = 6 # MPC mode
        # m.options.IMODE = 9 #dynamic ode sequential
        self.m.options.MAX_ITER = 500
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
        self.u = self.m.MV(value=900,lb=0,ub=991.666)
        self.u.STATUS = 1
        self.u.DCOST = 1e-5


        # integral over time for u^2
        self.u2 = self.m.Var(value=0)
        self.m.Equation(self.u2.dt() == 0.5*self.u**2)


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
        self.m.Equation(self.v.dt()==(self.u - self.g))

        #Set constraints
        # specify endpoint conditions
        self.m.Obj(self.final*1e3*(self.s-s_tf)**2) # Soft constraints
        self.m.Obj(self.final*1e3*(self.v-v_tf)**2)

        # minimize thrust used
        self.m.Obj(self.u2*self.final)

        #solve
        self.m.solve()

        #

        return self.u, self.m.time
