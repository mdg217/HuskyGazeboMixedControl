# Import ros package:
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf2_ros
import tf
from tf import transformations as t
from MPC_model import *
import numpy as np
from gazebo_msgs.srv import GetLinkState 
from gazebo_msgs.msg import geometry_msgs
import time


def get_link_states(link_name, reference_frame):
    rospy.wait_for_service('/gazebo/get_link_state')
    try:
        get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        state = get_link_state(link_name, reference_frame)
        return state
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

# Function to print the robot's current state
def print_states(x, y, z):
    print("(x = " + str(x) + ", y = " + str(y) + ", theta = " + str(z) + ")")

# Initialize ROS node
rospy.init_node('husky', anonymous=True)
pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=1)

# Create a Twist message for robot motion
move_cmd = Twist()

# Set the rate for the ROS loop
rate = rospy.Rate(10)

T_odom_world = get_link_states('husky::base_link', 'world') #its the same as T_baselink_world
T_O_W = t.concatenate_matrices(t.translation_matrix([T_odom_world.link_state.pose.position.x, T_odom_world.link_state.pose.position.y, T_odom_world.link_state.pose.position.z]),
                               t.quaternion_matrix([T_odom_world.link_state.pose.orientation.x, T_odom_world.link_state.pose.orientation.y, T_odom_world.link_state.pose.orientation.z, T_odom_world.link_state.pose.orientation.w]))

print(T_O_W)

# Create an instance of the MPC_model class
xd = [14]
yd = [14]

mpc_model = MPC_model(xd[0], yd[0], init_state=[0, 0, 0]) #Reference Positioning
mpc = mpc_model.getModel()

# Initial control input
u0 = np.array([0, 0]).reshape(2, 1)

i = 1
# Main ROS loop
while not rospy.is_shutdown():
    new_pose = get_link_states('husky::base_link', 'world') 
    new_T_O_W = t.concatenate_matrices(t.translation_matrix([new_pose.link_state.pose.position.x, new_pose.link_state.pose.position.y, new_pose.link_state.pose.position.z]),
                                t.quaternion_matrix([new_pose.link_state.pose.orientation.x, new_pose.link_state.pose.orientation.y, new_pose.link_state.pose.orientation.z, new_pose.link_state.pose.orientation.w]))

    new_real_pose = np.dot(t.inverse_matrix(T_O_W),new_T_O_W)
    print(new_real_pose)

    trans = tf.transformations.translation_from_matrix(new_real_pose)
    rot = tf.transformations.quaternion_from_matrix(new_real_pose)
        
    # Get the robot's current states (position and orientation)
    states = numpy.array([trans[0], trans[1], rot[2]]).reshape(-1, 1)
    print_states(trans[0], trans[1], rot[2])

    # Perform MPC step to get the control input
    u0 = mpc.make_step(states)

    # Set the linear and angular velocities for the robot's motion
    move_cmd.linear.x = u0[0]
    move_cmd.angular.z = u0[1]
    print(u0[0])

    # Publish the motion command
    pub.publish(move_cmd)

    # Sleep according to the defined rate
    rate.sleep()