import rospy
from geometry_msgs.msg import Twist
from tf import transformations as t
import numpy as np
from utility import *
from cost_cache import *
import do_mpc
from casadi import *
import numpy as np
from cost_cache import *
from model import *
from obstacle import *

u = np.load("/home/marco/catkin_ws/src/husky_mpc_datadriven/inputs_for_husky.npy")
print(np.shape(u))

rospy.init_node('husky_speed_controller', anonymous=True)

pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=1)

move_cmd = Twist()

# Set the rate for the ROS loop
rate = rospy.Rate(20)
i = 0

while not rospy.is_shutdown():

    move_cmd.linear.x = u[0, i] 
    move_cmd.angular.z = u[1, i]

    # Publish the motion command
    pub.publish(move_cmd)

    i += 1

    rate.sleep()