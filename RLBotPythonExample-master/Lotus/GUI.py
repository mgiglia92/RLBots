from tkinter import Tk, Label, Button, StringVar, Entry, Listbox, Text, Scrollbar
from OptimizationDriving import Optimizer
import numpy as np
import math
from EnvironmentManipulator import EnvironmentManipulator
from Enumerations import CarStateEnumeration, BallStateEnumeration

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
    def __init__(self, master, EM):
#Initialize EnvironmentManipulator
        self.EM = EM
        self.EM.setValue(1.0)

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

        self.label_vx = Label(master, text="X:")
        self.label_vx.grid(row=5, column=0)

        self.label_vy = Label(master, text="Y:")
        self.label_vy.grid(row=6, column=0)

        self.entry_initial_vx = Entry(master)#, validate = 'key', validatecommand = vcmd)
        self.entry_initial_vx.grid(row=5, column=1)
        self.entry_initial_vx.insert(0, "0")

        self.entry_initial_vy = Entry(master)
        self.entry_initial_vy.grid(row=6, column=1)
        self.entry_initial_vy.insert(0, "500")

        self.label_initial_yaw = Label(master, text= "Yaw")
        self.label_initial_yaw.grid(row=7, column=0)

        self.entry_initial_yaw = Entry(master)
        self.entry_initial_yaw.grid(row=7, column=1)
        self.entry_initial_yaw.insert(0, str(-1*math.pi/2))

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


#RUN TRAJECTORY BUTTON
        #RUN treatment button
        self.button_run = Button(master, text="RUN TRAJECTORY", command=lambda:self.runTrajectory())
        self.button_run.grid(row=10, column=10)

#REAL TIME OF TRAJECTORY
        self.label_t0_name = Label(master, text="t0")
        self.label_t0_name.grid(row=9, column=3, columnspan=2)

        self.label_t0 = Label(master, text="0.0")
        self.label_t0.grid(row=10, column=3)

        self.label_tnow_name = Label(master, text=":::::t_now:::::")
        self.label_tnow_name.grid(row=9, column=4, columnspan=2)

        self.label_tnow = Label(master, text="0.0")
        self.label_tnow.grid(row=10, column=4, columnspan=2)
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



    def compartmentalizeData(self):
        try:
            s_ti = [float(self.entry_initial_x.get()), float(self.entry_initial_y.get())]
            s_tf = [float(self.entry_final_x.get()), float(self.entry_final_y.get())]
            v_ti = [float(self.entry_initial_vx.get()), float(self.entry_initial_vy.get())]
            v_tf = [float(self.entry_final_vx.get()), float(self.entry_final_vy.get())]
            r_ti = [float(self.entry_initial_yaw.get())]
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


    def runTrajectory(self):
        #Set EnvironmentManipulator initial car and ball states
        b = BallStateEnumeration().ballStateHigh
        c = self.createCarStateFromGUI()
        self.EM.setInitialStates(c, b)
        self.EM.startTrajectory()

    def getTrajectoryData(self):
        # Trajectory Data Types
        ts = self.opt.ts
        sx = self.opt.sx
        sy = self.opt.sy
        vx = self.opt.vx
        vy = self.opt.vy
        yaw = self.opt.yaw
        omega = self.opt.omega

        return ts, sx, sy, vx, vy, yaw, omega

    def getInputData(self):
        return self.opt.a, self.opt.u_turning_d

    def createCarStateFromGUI(self):
        x = float(self.entry_initial_x.get())
        y = float(self.entry_initial_y.get())
        vx = float(self.entry_initial_vx.get())
        vy = float(self.entry_initial_vy.get())
        r = float(self.entry_initial_yaw.get())
        omega = float(self.entry_initial_omega.get())

        carState = CarState(boost_amount=100,
                         physics=Physics(location = Vector3(x, y, 17.01),velocity=Vector3(vx, vy, 0), rotation = Rotator(pitch = 0, yaw = r, roll = 0), angular_velocity = Vector3(0,0,omega)))

        return carState
