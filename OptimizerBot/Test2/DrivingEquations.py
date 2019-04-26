import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import Plotting
import Predictions
from CoordinateSystems import CoordinateSystems
from BallController import BallController
from pyquaternion import Quaternion
from TrajectoryGeneration import Trajectory
from TrueProportionalNavigation import TPN
import controller as con
import State as s
import numpy as np


def get_turning_radius(car, controller):
    #Get velocity magnitude
    v = np.linalg.norm(car.velocity)
    r_min = None #turn radius 1/k
    k_min = None #1/turn radius

    # Setup interpolation values
    v_interp = np.array([0, 500, 1000, 1500, 1750, 2300])
    k_interp = np.array([0.0069, 0.00398, 0.00235, 0.001375, 0.0011, 0.00088])

    #Get radius using linear interpolation segments
    k_min = np.interp(v, v_interp, k_interp)
    
    # Get actual curvature by multiplying by sterring value
    k_actual = k_min * controller.steer
    return k_min, k_actual

def get_minimum_turning_radius_polynomial(velocity):
    #Get velocity magnitude
    v = np.linalg.norm(velocity)

    #Get radius using 3rd degree polynomial determined form google sheets R^2 = 0.997
    radius = 152 - (0.0501*v) + 0.000455*(v**2) - 0.000000107*(v**3)

    return radius
