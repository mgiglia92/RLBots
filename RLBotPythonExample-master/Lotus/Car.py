import numpy as np
class Car:
    def __init__(self):
        #Member variables initialized

        #mass
        self.mass = None
        #position
        self.x = None
        self.y = None
        self.z = None

        #velocity
        self.vx = None
        self.vy = None
        self.vz = None

        #Pitch Roll yaw
        self.pitch = None
        self.roll = None
        self.yaw = None

        #angular velocities
        self.wx = None
        self.wy = None
        self.wz = None
        self.position = np.array([0,0,0])
        self.velocity = np.array([0,0,0])
        self.angular_velocity = np.array([0,0,0])
        self.attitude = np.array([0,0,0])
        #MUST CHECK THSESE TO MAKE SURE THEY CORRELATE PROPERLY

        self.is_demolished = False
        self.has_wheel_contact = True
        self.is_super_sonic = False
        # self.jumped = False
        self.boost_left = '100'
        # self.double_jumped = 'False'

    def update(self, data):
        #Member variables initialized
        #position
        self.x = data.physics.location.x
        self.y = data.physics.location.y
        self.z = data.physics.location.z

        #velocity
        self.vx = data.physics.velocity.x
        self.vy = data.physics.velocity.y
        self.vz = data.physics.velocity.z

        #Pitch Roll yaw
        self.pitch = data.physics.rotation.pitch
        self.roll = data.physics.rotation.roll
        self.yaw = data.physics.rotation.yaw

        #angular velocities
        self.wx = data.physics.angular_velocity.x
        self.wy = data.physics.angular_velocity.y
        self.wz = data.physics.angular_velocity.z

        self.position = np.array([self.x,self.y,self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])
        self.angular_velocity = np.array([self.wx, self.wy, self.wz])
        self.attitude = np.array([self.roll, self.pitch, self.yaw])

        self.is_demolished = data.is_demolished
        self.has_wheel_contact = data.has_wheel_contact
        self.is_super_sonic = data.is_super_sonic
        # self.jumped = data.jumped
        self.boost_left = data.boost
        # self.double_jumped = data.double_jumped

    def printVals(self):
        print("x:", int(self.x), "y:", self.y, "z:", self.z, "wx:", self.wx, "wy:", self.wy, "wz:", self.wz)
