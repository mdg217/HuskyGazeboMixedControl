import rospy
import numpy as np
from cost_cache import *
from utility import *
from gazebo_msgs.srv import GetModelState


class Obstacle:

    def __init__(self):

        #Get obstacles from parameter server
        self.cache = CostCache()
        self.obs_odom = []
        #self.num = rospy.get_param("/number_of_obstacles")

        """for i in range(self.num):
            param = "/obs" + str(i+1)
            self.obs_odom.append(get_obstacle_position_odom(self.cache.get_T(), rospy.get_param(param)[0:6]) + rospy.get_param(param)[6:])"""
        
        self.get_pose_from_gazebo()

        for obs in self.obs_odom:
            print(obs)


    def get_pose_from_gazebo(self):

        rospy.wait_for_service('/gazebo/get_model_state')
        
        for i in range(9):
            get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            obstacle = "obstacle" + str(i)
            param = get_model_state(obstacle , "")
            pose = [param.pose.position.x, param.pose.position.y, param.pose.position.z, param.pose.orientation.x, param.pose.orientation.y, param.pose.orientation.z, param.pose.orientation.w]
            self.obs_odom.append(get_obstacle_position_odom(self.cache.get_T(), pose) + [1, 1])
        

    def get_obs(self):
        return self.obs_odom

    def get_max_min(self):
        return []