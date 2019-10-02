from tkinter import Tk, Label, Button, StringVar, Entry, Listbox
from GUI import GUI

root = Tk()
g = GUI(root)
g.master.mainloop() #GUI MAIN LOOP start
print(g.entry_final_x.value)
