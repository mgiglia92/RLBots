# from tkinter import Tk, Label, Button, StringVar, Entry, Listbox, Text, Scrollbar, Checkbutton, Canvas
import tkinter as tk
from tkinter import Tk, Label, Button, StringVar, Entry, Listbox, Text, Scrollbar, Checkbutton, Canvas, PhotoImage, NW
# from tkinter import *
from OptimizationDriving import Optimizer
import numpy as np
import math
from EnvironmentManipulator import EnvironmentManipulator
from Enumerations import CarStateEnumeration, BallStateEnumeration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import linecache

# RLBOT classes and services
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState
from rlbot.utils.game_state_util import CarState
from rlbot.utils.game_state_util import Physics
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.game_state_util import Rotator
from rlbot.utils.game_state_util import BallState


class GUI:
    def PrintException(self):
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

    def __init__(self, master, EM, sim_results):
        try:
            # Initialize plotting figures variables etc
            self.fig = plt.figure(1)
            self.fig2 = plt.figure(2)
            self.ax = self.fig.gca(projection='3d')
            # self.ax = plt.axes(projection='3d') #self.fig.add_subplot(111, projection='3d')

        except BaseException as e:
            self.PrintException()



#Initialize EnvironmentManipulator
        self.EM = EM
        self.EM.setValue(1.0)
# SImulation results
        self.sim_results = sim_results
#Import Optimizer Alogorithm Class
        self.opt = Optimizer()
#GUI INIT
        self.master = master
        master.title("Lotus Test Platform")

#INITIAL STATE OF TRAJECTORY WIDGETS
        self.label_initial = Label(master, text="Initial State of Trajectory")
        self.label_initial.grid(row=0, column=0, columnspan=2)

        self.label_initial_position = Label(master, text="Position")
        self.label_initial_position.grid(row=1, column=0, columnspan=2)
        self.label_initial_position.justify = "center"

        self.label_x = Label(master, text="X:")
        self.label_x.grid(row=2, column=0)

        self.label_y = Label(master, text="Y:")
        self.label_y.grid(row=3, column=0)

        self.entry_initial_x = Entry(master)#, validate = 'key', validatecommand = vcmd)
        self.entry_initial_x.grid(row=2, column=1)
        self.entry_initial_x.insert(0, "-2000")

        self.entry_initial_y = Entry(master)
        self.entry_initial_y.grid(row=3, column=1)
        self.entry_initial_y.insert(0, "-1000")

        self.label_initial_velocity = Label(master, text="Velocity")
        self.label_initial_velocity.grid(row=4, column=0, columnspan=2)
        self.label_initial_velocity.justify = "center"

        self.label_vmag = Label(master, text="V_Magnitude:")
        self.label_vmag.grid(row=5, column=0)

        # self.label_vy = Label(master, text="Y:")
        # self.label_vy.grid(row=6, column=0)

        self.entry_initial_vmag = Entry(master)#, validate = 'key', validatecommand = vcmd)
        self.entry_initial_vmag.grid(row=5, column=1)
        self.entry_initial_vmag.insert(0, "0")

        # self.entry_initial_vy = Entry(master)
        # self.entry_initial_vy.grid(row=6, column=1)
        # self.entry_initial_vy.insert(0, "500")

        self.label_initial_yaw = Label(master, text= "Yaw")
        self.label_initial_yaw.grid(row=7, column=0)

        self.entry_initial_yaw = Entry(master)
        self.entry_initial_yaw.grid(row=7, column=1)
        self.entry_initial_yaw.insert(0, str(np.pi/2))

        self.label_initial_omega = Label(master, text="Omega(yaw rate)")
        self.label_initial_omega.grid(row=8, column=0)

        self.entry_initial_omega = Entry(master)
        self.entry_initial_omega.grid(row=8, column =1)
        self.entry_initial_omega.insert(0, "0")

# FINAL STATE OF TRAJECTORY WUIDGETS
        self.label_final = Label(master, text="Final State of Trajectory")
        self.label_final.grid(row=0, column=2, columnspan=2)

        self.label_final_position = Label(master, text="Position")
        self.label_final_position.grid(row=1, column=2, columnspan=2)
        self.label_final_position.justify = "center"

        self.label_final_x = Label(master, text="X:")
        self.label_final_x.grid(row=2, column=2)

        self.label_final_y = Label(master, text="Y:")
        self.label_final_y.grid(row=3, column=2)

        self.entry_final_x = Entry(master)#, validate = 'key', validatecommand = vcmd)
        self.entry_final_x.grid(row=2, column=3)
        self.entry_final_x.insert(0, "0")

        self.entry_final_y = Entry(master)
        self.entry_final_y.grid(row=3, column=3)
        self.entry_final_y.insert(0, "0")

        self.label_final_velocity = Label(master, text="Velocity")
        self.label_final_velocity.grid(row=4, column=2, columnspan=2)
        self.label_final_velocity.justify = "center"

        self.label_final_vx = Label(master, text="X:")
        self.label_final_vx.grid(row=5, column=2)

        self.label_final_vy = Label(master, text="Y:")
        self.label_final_vy.grid(row=6, column=2)

        self.entry_final_vx = Entry(master)#, validate = 'key', validatecommand = vcmd)
        self.entry_final_vx.grid(row=5, column=3)
        self.entry_final_vx.insert(0, "0")

        self.entry_final_vy = Entry(master)
        self.entry_final_vy.grid(row=6, column=3)
        self.entry_final_vy.insert(0, "0")

        self.label_final_yaw = Label(master, text= "Yaw")
        self.label_final_yaw.grid(row=7, column=2)

        self.entry_final_yaw = Entry(master)
        self.entry_final_yaw.grid(row=7, column=3)
        self.entry_final_yaw.insert(0, "0")

#GENERATE TRAJECTORY BUTTONS
        self.label_trajectory_generation = Label(master, text="Trajectory Generation")
        self.label_trajectory_generation.grid(row=9, column=0, columnspan=2)
        self.label_trajectory_generation.justify = "center"

        self.button_generate_trajectory_minimum_time = Button(master, text="Min Time", command=lambda:self.getTrajectory())
        self.button_generate_trajectory_minimum_time.grid(row=10, column=0)

# Save sim data checkbox
        self.cb_save_simulation = Checkbutton(master, text="Save Sim Data")
        self.cb_save_simulation.grid(row=9, column=10)
        self.cb_save_simulation.select() # Make checkbutton checked when gui starts

#RUN TRAJECTORY BUTTON
        #RUN treatment button
        self.button_run = Button(master, text="RUN TRAJECTORY", command=lambda:self.runTrajectory())
        self.button_run.grid(row=10, column=10)

#PLOT TRAJECTORY BUTTON
        self.button_plot = Button(master, text="PLOT TRAJECTORY DATA", command=lambda:self.plotTrajectory())
        self.button_plot.grid(row=10, column=11)
#PLOT SIMULATION RESULTS BUTTON
        self.button_plot_sim_results = Button(master, text="PLOT SIM RESULTS", command=lambda:self.plotSimulationResults())
        self.button_plot_sim_results.grid(row=9, column = 11)
#REAL TIME OF TRAJECTORY
        self.label_t0_name = Label(master, text="t0")
        self.label_t0_name.grid(row=9, column=3, columnspan=2)

        self.label_t0 = Label(master, text="0.0")
        self.label_t0.grid(row=10, column=3)

        self.label_tnow_name = Label(master, text=":::::t_now:::::")
        self.label_tnow_name.grid(row=9, column=5, columnspan=4)

        self.label_tnow = Label(master, text="0.0")
        self.label_tnow.grid(row=10, column=5, columnspan=4)
        self.label_tnow.justify = "center"

#DISPLAY OPTIMATL TRAJECTORY DATA
        self.label_trajectory_data = Label(master, text="---------------------------------------------------------------------Trajectory Data---------------------------------------------------------------------")
        self.label_trajectory_data.grid(row=11, column = 0, columnspan = 15)
        self.label_trajectory_data.justfy = "center"

        self.text_trajectory_data = Text(master)
        self.text_trajectory_data.grid(row=12, column=0, columnspan=15, sticky='nsew', ipady=100)

        #Scroll bar for trajectory data
        # scrollb = Scrollbar(self.text_trajectory_data, command=self.text_trajectory_data.yview, orient='vertical')
        # scrollb.grid(row=11, column=11, sticky='nsew')
        # self.text_trajectory_data['yscrollcommand']=scrollb.set

# WAYPOINT PLACEMENT IMAGE AND MOVEABLE SHAPES
        self.canvas_waypoint=tk.Canvas(master, width=580, height=700, background='gray')
        self.canvas_waypoint.grid(row=1,column=15, columnspan = 5, rowspan = 15, sticky='nwe')
        self.field = PhotoImage(file='D:\Documents\RLBot\RLBotPythonExample-master\Lotus\RL_Field.png')
        self.canvas_waypoint.create_image(0, 0, image = self.field, anchor=NW)
# WAYPOINT CIRCLES
        self.waypoint_size = 25
        self.waypoint_position_x = [150, 300, 300, 150] # Position of the waypoint shapes in canvas
        self.waypoint_position_y= [150, 150, 300, 300]
        self.waypoint_shape = []
        self.waypoint_text = []

        #Create the 4 way points in their default positions
        for i in range(len(self.waypoint_position_x)):
            x = self.canvas_waypoint.create_oval(self.waypoint_position_x[i], self.waypoint_position_y[i], self.waypoint_position_x[i] + self.waypoint_size, self.waypoint_position_y[i] + self.waypoint_size, outline='black', fill='green')
            self.waypoint_shape.append(x)
            y = self.canvas_waypoint.create_text(self.waypoint_position_x[i] + 4, self.waypoint_position_y[i] + 4,anchor=NW, font=("Purisa", 10, 'bold'), text="W" + str(i+1), fill='white')
            self.waypoint_text.append(y)
        print(self.waypoint_shape)

#Bind left mousebutton click to update position of way point
        self.canvas_waypoint.bind("<ButtonRelease-1>", self.move_waypoint)

#Bind mouse entering canvas to give it keyboard focus
        self.canvas_waypoint.bind("<Enter>", lambda event: self.canvas_waypoint.focus_set())

#Bind keyboard buttons 1234 to change the "state" of the editing feature. Number you press changes which waypoint you will change
        self.canvas_waypoint.bind("0", self.set_editing_state)
        self.canvas_waypoint.bind("1", self.set_editing_state)
        self.canvas_waypoint.bind("2", self.set_editing_state)
        self.canvas_waypoint.bind("3", self.set_editing_state)
        self.canvas_waypoint.bind("4", self.set_editing_state)
        self.current_waypoint = 0 # Current waypoint that events will affect

    def move_waypoint(self, event):
        try:

            bbox = np.array(self.canvas_waypoint.coords(self.waypoint_shape[self.current_waypoint])) #Get bounding box of shape
            print(bbox)
            init = self.get_center_of_shape(bbox) #Get center positoin vector of shape from bounding box
            new = np.array([event.x, event.y]) # Get mouse position
            final = new - init # Find deltas
            self.canvas_waypoint.move(self.waypoint_shape[self.current_waypoint], final[0], final[1]) #Move shape by delta
            self.canvas_waypoint.move(self.waypoint_text[self.current_waypoint], final[0], final[1]) #Move shape by delta
        except BaseException as e:
            self.PrintException()

    def set_editing_state(self, event):
        self.current_waypoint = int(event.char) - 1
        print(self.current_waypoint)

    def get_center_of_shape(self, bbox): #bbox is bounding box (x1,y1,x2,y2)
        position = np.array([int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)])
        return position

    def compartmentalizeData(self):
        try:
            s_ti = [float(self.entry_initial_x.get()), float(self.entry_initial_y.get())]
            s_tf = [float(self.entry_final_x.get()), float(self.entry_final_y.get())]
            v_ti = float(self.entry_initial_vmag.get())
            v_tf = [float(self.entry_final_vx.get()), float(self.entry_final_vy.get())]
            r_ti = float(self.entry_initial_yaw.get())
            omega_ti = [float(self.entry_initial_omega.get())]

            return s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti

        except Exception as e:
            print(e)


    def getTrajectory(self):
        try:
            s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti = self.compartmentalizeData()

            self.opt.__init__() #Reset initial conditions and all variables to allow a clean simulation
            acceleration, yaw, t_star = self.opt.optimize2D(s_ti, s_tf, v_ti, v_tf, r_ti, omega_ti) #Run the driving optimization algorithm

            print("t_star", t_star, "acceleration", acceleration.value, "yaw", yaw.value)
            string = "\n--------------------------------------------------" + \
                str("\nt_star: ") + str(t_star)+"\nacceleration: "+ str(acceleration.value)+ "\nyaw: "+ str(yaw.value) + "\nturning: " + str(self.opt.u_turning_d.value)
            self.text_trajectory_data.insert('1.end', string)


            return t_star, acceleration, self.opt.u_turning_d.value

        except Exception as e:
            print(e)
            print('An entry is invalid')

    def plotSimulationResults(self):
        try:
            if 'self.ax' not in locals(): # create self.ax becuase it throws an error when in the __init__() function, b/c of threads?
                # self.ax = self.fig.add_subplot(111, projection='3d')
                self.ax = self.fig.gca(projection='3d')

            ts = self.sim_results.ts
            ts_a = np.array(self.sim_results.t_now[2:-1]) - self.sim_results.t_now[2]

            # Switch to right figure
            plt.figure(1)
            self.ax.clear()
            # Plot reference trajectory vs time
            Axes3D.plot(self.ax, self.sim_results.sx, self.sim_results.sy, ts, c='r', marker ='o')
            plt.ylabel('Position/Velocity y')
            plt.xlabel('Position/Velocity x')
            self.ax.set_zlabel('time')

            # Plot actual trajectory vs time
            Axes3D.plot(self.ax, self.sim_results.sx_a[2:-1], self.sim_results.sy_a[2:-1], ts_a, c='b', marker ='.')

            plt.figure(2)
            plt.clf()
            plt.subplot(3,1,1)
            plt.plot(ts, self.sim_results.vx, 'r-')
            # plt.ylabel('acceleration')

            plt.subplot(3,1,1)
            plt.plot(ts_a,  self.sim_results.vx_a[2:-1], 'b-')
            plt.ylabel('vx error')

            # Get v_mag and v_mag_a
            v_mag = np.sqrt(np.multiply(self.sim_results.vx, self.sim_results.vx) + np.multiply(self.sim_results.vy, self.sim_results.vy))
            v_mag_a = np.sqrt(np.multiply(self.sim_results.vx_a, self.sim_results.vx_a) + np.multiply(self.sim_results.vy_a, self.sim_results.vy_a))

            plt.subplot(3, 1, 3)
            plt.plot(ts, v_mag, 'b-')
            plt.ylabel('vmag')
            plt.subplot(3, 1, 3)
            plt.plot(ts_a, v_mag_a[2:-1], 'g-')
            plt.ylabel('vmag')


            plt.ion()
            plt.show()
            plt.pause(0.001)
        except BaseException as e:
            self.PrintException()

    def plotTrajectory(self):
        try:
            print("printing trajectory?")
            ts = self.opt.d.time * self.opt.tf.value[0]

            # Switch to right figure
            self.fig

            # plt.subplot(2, 1, 1)
            Axes3D.plot(self.ax, self.opt.sx.value, self.opt.sy.value, ts, c='r', marker ='o')
            plt.ylabel('Position/Velocity y')
            plt.xlabel('Position/Velocity x')
            self.ax.set_zlabel('time')

            # fig = plt.figure(3)
            # self.ax = fig.add_subplot(111, projection='3d')
            # plt.subplot(2, 1, 1)
            Axes3D.plot(self.ax, self.opt.vx.value, self.opt.vy.value, ts, c='b', marker ='.')

            self.fig2
            plt.clf()
            plt.subplot(3,1,1)
            plt.plot(ts, self.opt.a, 'r-')
            plt.ylabel('acceleration')

            plt.subplot(3,1,2)
            plt.plot(ts, np.multiply(self.opt.yaw, 180/math.pi), 'r-')
            plt.ylabel('turning input')

            plt.subplot(3, 1, 3)
            plt.plot(ts, self.opt.v_mag, 'b-')
            plt.ylabel('vmag')

            plt.ion()
            plt.show()
            plt.pause(0.001)

        except Exception as e:
            self.PrintException()

    def runTrajectory(self):
        #Set EnvironmentManipulator initial car and ball states
        b = self.createBallStateFromGUI() # BallStateEnumeration().ballStateHigh
        c = self.createCarStateFromGUI()
        self.EM.setInitialStates(c, b)
        self.EM.startTrajectory()

    def getTrajectoryData(self):
        # Trajectory Data Types
        ts = self.opt.d.time * self.opt.tf.value[0] # Get the time vector
        print("ts: ", ts)
        print(type(ts))
        sx = self.opt.sx
        sy = self.opt.sy
        vx = self.opt.vx
        vy = self.opt.vy
        yaw = self.opt.yaw
        omega = self.opt.omega
        curvature = self.opt.curvature

        return ts, sx, sy, vx, vy, yaw, omega, curvature

    def getInputData(self):
        return self.opt.a, self.opt.u_turning_d

    def createCarStateFromGUI(self):
        x = float(self.entry_initial_x.get())
        y = float(self.entry_initial_y.get())
        vmag = float(self.entry_initial_vmag.get())
        r = float(self.entry_initial_yaw.get())
        omega = float(self.entry_initial_omega.get())

        # get x y velocities
        vx = np.cos(r)*vmag
        vy = np.sin(r)*vmag

        carState = CarState(boost_amount=100,
                         physics=Physics(location = Vector3(x, y, 17.01),velocity=Vector3(vx, vy, 0), rotation = Rotator(pitch = 0, yaw = r, roll = 0), angular_velocity = Vector3(0,0,omega)))

        return carState

    def createBallStateFromGUI(self):
        x = float(self.entry_final_x.get())
        y = float(self.entry_final_y.get())
        ballState = BallState(Physics(location=Vector3(x, y, 1000), velocity = Vector3(0, 0, 0)))

        return ballState
