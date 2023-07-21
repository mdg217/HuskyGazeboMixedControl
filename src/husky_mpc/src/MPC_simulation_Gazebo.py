# Import ros package:
import rospy
from geometry_msgs.msg import Twist
import tf
import tf2_ros
from tf import transformations as t
from MPC_model import *
import numpy as np
from gazebo_msgs.srv import GetLinkState 
import geometry_msgs.msg
from utility import *
from ObstacleCircle import *
from CostCache import *


cache = CostCache()
visionfield_radius = 1.5

x_offset = 7
y_offset = 7
obs1 = ObstacleCircle((-4.60951+x_offset), (-3.97645+x_offset), 0.831860)


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

# Create an instance of the MPC_model class
xd = [5]
yd = [5]

mpc_model = MPC_model(xd[0], yd[0], init_state=[0, 0, 0]) #Reference Positioning
mpc = mpc_model.getModel()

# Initial control input
u0 = np.array([0, 0]).reshape(2, 1)

# Main ROS loop
time = 0

while not rospy.is_shutdown():
    new_pose = get_link_states('husky::base_link', 'world') 
    new_T_O_W = t.concatenate_matrices(t.translation_matrix([new_pose.link_state.pose.position.x, new_pose.link_state.pose.position.y, new_pose.link_state.pose.position.z]),
                t.quaternion_matrix([new_pose.link_state.pose.orientation.x, new_pose.link_state.pose.orientation.y, new_pose.link_state.pose.orientation.z, new_pose.link_state.pose.orientation.w]))

    new_real_pose = np.dot(t.inverse_matrix(T_O_W),new_T_O_W)
    trans = tf.transformations.translation_from_matrix(new_real_pose)
    rot = tf.transformations.quaternion_from_matrix(new_real_pose)

    if obs1.intersection(trans[0], trans[1], visionfield_radius):
        cache.set_cost(10000*max(0, obs1.r - obs1.distance(trans[0], trans[1])), time)

    # Get the robot's current states (position and orientation)
    states = numpy.array([trans[0], trans[1], rot[2]]).reshape(-1, 1)
    print_states(trans[0], trans[1], rot[2])

    publish_tf(trans, rot)

    states_noise = np.array(add_noise_to_states(states)).reshape(-1,1)

    # Perform MPC step to get the control input
    u0 = mpc.make_step(states_noise)

    # Set the linear and angular velocities for the robot's motion
    move_cmd.linear.x = u0[0]
    move_cmd.angular.z = u0[1]

    # Publish the motion command
    pub.publish(move_cmd)

    # Sleep according to the defined rate
    rate.sleep()
    time+=1
