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

def get_link_states(link_name, reference_frame):
    rospy.wait_for_service('/gazebo/get_link_state')
    try:
        get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        state = get_link_state(link_name, reference_frame)
        return state
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


def publish_tf(translation, rotation):
    br = tf2_ros.TransformBroadcaster()

    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "odom"
    t.child_frame_id = "base_link_gazebo"
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = 0.0
    t.transform.rotation.x = rotation[0]
    t.transform.rotation.y = rotation[1]
    t.transform.rotation.z = rotation[2]
    t.transform.rotation.w = rotation[3]

    br.sendTransform(t) 


def add_noise_to_states(states):
    noise_xy = 0*np.random.normal(0,1,1)
    noise_theta = 0*np.random.normal(0,1,1)

    return [states[0] + noise_xy, states[1] + noise_xy, states[2] + noise_theta]


def print_states(x, y, z):
    print("(x = " + str(x) + ", y = " + str(y) + ", theta = " + str(z) + ")")
