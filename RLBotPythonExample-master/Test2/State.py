class State():
    def __init__(self):
        self.current_state = None
        self.driving = 0
        self.flying = 1
        self.recovering = 2
        self.defending = 3
        self.optimization = 4
    def set_state(self, state):
        if state == 0:
            self.current_state = self.driving
        elif state == 1:
            self.current_state = self.flying
        elif state == 2:
            self.current_state = self.recovering
        elif state == 3:
            self.current_state = self.defending
        elif state == 4:
            self.current_state = self.optimization

    def get_state(self):
        return self.current_state
