import rospy
from mpc_controller import *
from klc_controller import *
from cost_cache import *

cache = CostCache()

klc = ControllerKLC([16, 16], 1)
x, y, time = klc.update()
cache.set_next_target(x, y)

mpc = ControllerMPC([0,0,0])

while not rospy.is_shutdown():    mpc.update()