# Import ros package:
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf
# Import do_mpc package:
import do_mpc
from casadi import *
from Getpose import *

rospy.init_node('husky', anonymous=True)
pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=1)

import math
import numpy as np
# do-mpc implementation
model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))

v = model.set_variable(var_type='_u', var_name='v')
w = model.set_variable(var_type='_u', var_name='w')
    
model.set_rhs('x', v*np.cos(theta))
model.set_rhs('y', v*np.sin(theta))
model.set_rhs('theta', w)
model.setup()
mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    't_step': 0.1,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

xd = 2
yd = 4

"""if xd == 0 & yd > 0: thetad = np.deg2rad(90)
elif xd == 0 & yd <= 0: thetad = np.deg2rad(-90) 
elif yd == 0 & xd < 0: thetad = np.deg2rad(180)
else: thetad =  np.arctan(yd/xd)"""

mterm = (x-xd)**2 + 0.4*(y-yd)**2 #+ 0.01*(theta)**2   #lyapunov
lterm = (x-xd)**2 + 0.4*(y-yd)**2  + 1/2*v**2 + 1/2*w**2
    
mpc.set_objective(mterm=mterm, lterm=lterm)
# Lower bounds on states:
mpc.bounds['lower','_x', 'x'] = 0
mpc.bounds['lower','_x', 'y'] = 0

# Lower bounds on inputs:
mpc.bounds['lower','_u', 'v'] = -1
mpc.bounds['lower','_u', 'w'] = -1

# Lower bounds on inputs:
mpc.bounds['upper','_u', 'v'] = 1
mpc.bounds['upper','_u', 'w'] = 1

mpc.setup()
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = 0.1)
simulator.setup()
x0 = np.array([0, 0, 0]).reshape(-1, 1)
print(x0)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

move_cmd = Twist()
rate = rospy.Rate(10)
u0 = np.array([0, 0]).reshape(2,1)

data = [0, 0, 0]

def print_states(x,y,z):
    print("(x = " + str(x) + ", y = " + str(y) + ", theta = " + str(z) + ")")

def callback(msg):
    global data
    data = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation.z]

rospy.Subscriber('/husky_velocity_controller/odom', Odometry, callback)
pose = tf.TransformListener()

while not rospy.is_shutdown():

    try:
        (trans,rot) = pose.lookupTransform('/odom', '/base_link', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue

    orient = tf.transformations.euler_from_quaternion(rot)[2]
    print(orient)

    states = np.array([trans[0], trans[1], orient]).reshape(-1,1)
    print_states(trans[0], trans[1], orient)
    u0 = mpc.make_step(states)
    #print(u0)
    
    move_cmd.linear.x = u0[0]
    move_cmd.angular.z = u0[1]
    
    pub.publish(move_cmd)
    
    rate.sleep()