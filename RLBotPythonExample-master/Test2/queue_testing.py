import queue
import time


class test():
    def __init__(self):

        self.data_queue = queue.Queue(maxsize = 0)
        print(self.data_queue)

        self.other_intialize()

    def other_intialize(self):
        self.m = 'wtf bruh'
