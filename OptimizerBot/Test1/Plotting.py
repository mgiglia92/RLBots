import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp



def plotVelocities(actualVelocities, predictedVelocities, time):
    None
def plotAttitude(actualAttitude, predictedAttitude, time):
    None
def plotAngularVelocities(actualOmega, predictedOmega, time):
    None

class data:
    def __init__(self):
        self.d1 = np.array([0])
        self.d2 = np.array([0])
        self.d3 = np.array([0])
        self.d4 = np.array([0])
        self.flag = 0
    def add(self, d1, d2, d3, d4):
        if(self.flag == 0):
            self.d1 = np.array([d1])
            self.d2 = np.array([d2])
            self.d3 = np.array([d3])
            self.d4 = np.array([d4])
            self.flag = 1
        else:
            self.d1 = np.append(self.d1, d1)
            self.d2 = np.append(self.d2, d2)
            self.d3 = np.append(self.d3, d3)
            self.d4 = np.append(self.d4, d4)

def plotPositions(q1, q2):
    plt.ion()
    ln, = plt.plot([0], [0])
    plt.show()

    while True:
        print("DERP")
        obj = q.get()
        n = obj + 0
        ln.set_xdata(np.append(ln.get_xdata(), n))
        ln.set_ydata(np.append(ln.get_ydata(), n))
        ax.relim()
        ax.autoscale_view(True, True, True)
        fig.canvas.draw()
