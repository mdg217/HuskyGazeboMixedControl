# Import ros package:
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf
from MPC_model import *
import numpy as np

# Function to print the robot's current state
def print_states(x, y, z):
    print("(x = " + str(x) + ", y = " + str(y) + ", theta = " + str(z) + ")")

# Initialize ROS node
rospy.init_node('husky', anonymous=True)
pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=1)

# Create a TransformListener to get the robot's pose
pose = tf.TransformListener()

# Create a Twist message for robot motion
move_cmd = Twist()

# Set the rate for the ROS loop
rate = rospy.Rate(10)

# Create an instance of the MPC_model class
xd = [5, 5, 5]
yd = [0, 5, 7]

mpc_model = MPC_model(xd[0], yd[0], init_state=[0, 0, 0]) #Reference Positioning
mpc = mpc_model.getModel()

# Initial control input
u0 = np.array([0, 0]).reshape(2, 1)

i = 1
# Main ROS loop
while not rospy.is_shutdown():

    try:
        # Get the transform between the '/odom' and '/base_link' frames
        (trans, rot) = pose.lookupTransform('/odom', '/base_link', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue

    # Extract the orientation angle from the quaternion
    orient = tf.transformations.euler_from_quaternion(rot)[2]

    # Get the robot's current states (position and orientation)
    states = np.array([trans[0], trans[1], orient]).reshape(-1, 1)
    print_states(trans[0], trans[1], orient)

    # Perform MPC step to get the control input
    u0 = mpc.make_step(states)

    # Set the linear and angular velocities for the robot's motion
    move_cmd.linear.x = u0[0]
    move_cmd.angular.z = u0[1]
    
    if u0[0] <= 0.01:
        print("Posizione finale raggiunta: " + str(print_states(trans[0], trans[1], orient)))
        input()
        # Get the updated cost function in the model
        mpc_model = MPC_model(xd[i], yd[i], init_state=[trans[0], trans[1], orient]) #Reference Positioning
        mpc = mpc_model.getModel()
        i+=1
        if i > len(xd):
            break

        # Initial control input
        u0 = np.array([0, 0]).reshape(2, 1)


    # Publish the motion command
    pub.publish(move_cmd)

    # Sleep according to the defined rate
    rate.sleep()
