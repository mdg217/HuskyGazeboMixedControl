import rospy
from mpc_controller import *

mpc = MPC_controller(4, 0, [0,0,0])

#while not rospy.is_shutdown():    mpc.update()