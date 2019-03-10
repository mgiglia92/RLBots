import numpy as np
from scipy.optimize import minimize, Bounds
from scipy import integrate as int
import scipy.linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math

def objective():
    None

def future_state(a, v, s, dt):
    # In the air
    if(state.has_wheel_contact == False):
        g = np.array([0,0,-650])

        v1 = (a + g)*dt + v

        d1 = (a+g)*(dt**2)

        p1 = ((d1) / 2) + (v*dt) + s

        return p1, v1

# Code
t_f = 1.0
t = np.linspace(0., t_f, num = 60) # Time array for 1 second into the future with 0.01 increment
u = np.zeros(t.size) + 650
print(u)
g = -650
initial_position = 0
initial_velocity = 0
final_position = 100
final_velocity = 100

def car_dynamics(x):
    # Create time vector
    # t = np.linspace(0., t_f, num = 100) # Time array for 1 second into the future with 0.01 increment


    # Integrate over entire time to find v as a function of t
    a = x + g
    v = int.cumtrapz(a, t, initial = 0) + initial_velocity

    # Integrate v(t) to get s(t)
    s = int.cumtrapz(v, t, initial = 0) + initial_position

    return s, v

def fullconstraint(x):
    s, v = car_dynamics(x)

    print(s[0], v[0], s[-1], v[-1])
    return [initial_position - s[0],
            initial_velocity - v[0],
            final_position - s[-1],
            final_velocity - v[-1]]

def constraint1(x): # Final state constraints (Boundary conditions)
    s, v = car_dynamics(x)
    print('c1', s[0] - initial_position)
    return s[0] - initial_position

def constraint2(x): # Initial state constraints (initial conditions)
    s, v = car_dynamics(x)
    print('c2', v[0] - initial_velocity)
    return v[0] - initial_velocity

def constraint3(x):
    s, v = car_dynamics(x)
    print('c3', s[-1] - final_position)
    return s[-1] - final_position

def constraint4(x):
    s, v = car_dynamics(x)
    print('c4', v[-1] - final_velocity)
    return v[-1] - final_velocity

def constraint5(x):
    return x - 1000

def objective(x):
    u2 = np.square(x)
    return np.sum(u2)

cons = [{'type':'eq', 'fun':constraint1},
                {'type':'eq', 'fun':constraint2},
                {'type':'eq', 'fun':constraint3},
                {'type':'eq', 'fun':constraint4}]
                # {'type':'ineq', 'fun':constraint5}]

cons2 = {'type':'eq', 'fun':fullconstraint}
# Bounds
input_bound = Bounds(-1000,1000)

result = minimize(objective, u, bounds = input_bound, constraints = cons2, method = 'SLSQP', options={'eps':10, 'maxiter':100, 'ftol':1, 'disp':True})
print(result)

# [s, v] = car_dynamics(u)
# # Plotting the ODE
# plt.plot(t, s, 'g')
# plt.title('Multiple Parameters Test')
# plt.xlabel('Time')
# plt.ylabel('Magnitude')
# plt.show()
