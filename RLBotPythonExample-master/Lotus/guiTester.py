from tkinter import Tk, Label, Button, StringVar, Entry, Listbox
from GUI import GUI
from EnvironmentManipulator import EnvironmentManipulator

EM = EnvironmentManipulator()

root = Tk()
g = GUI(root, EM)
g.master.mainloop() #GUI MAIN LOOP start
print(g.entry_final_x.get())
