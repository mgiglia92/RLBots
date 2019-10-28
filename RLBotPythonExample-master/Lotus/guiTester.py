from tkinter import Tk, Label, Button, StringVar, Entry, Listbox
from GUI import GUI
from EnvironmentManipulator import EnvironmentManipulator
from SimulationResults import SimulationResults
EM = EnvironmentManipulator()
sm = SimulationResults()

root = Tk()
g = GUI(root, EM, sm)
g.master.mainloop() #GUI MAIN LOOP start
print(g.entry_final_x.get())
