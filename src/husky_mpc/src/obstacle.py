import rospy
from geometry_msgs.msg import Twist
import tf
import tf2_ros
from tf import transformations as t
from MPC_model import *
import numpy as np
from gazebo_msgs.srv import GetLinkState 
import geometry_msgs.msg


class Obstacle:

    def __init__(self):

        #accedi da gazebo al nome passato per argomento e preleva la posa e le dimensioni
        

        #calcola e imposta i vari valori
        self.xc = 0
        self.yc = 0
        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0


    def get_model_boundingbox(model_name):
        rospy.wait_for_service('/gazebo/get_link_state')
        try:
            get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            state = get_link_state(link_name, reference_frame)
            return state
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)