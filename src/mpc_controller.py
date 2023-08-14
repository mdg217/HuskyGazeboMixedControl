import rospy
from geometry_msgs.msg import Twist
import tf
import tf2_ros
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

class MPC_controller:

    def tvp_fun(self, t_now):
        for k in range(21):
                x, y = self.cache.next_target(self.index)
                self.tvp_template['_tvp', k, 'xd'] = x
                self.tvp_template['_tvp', k, 'yd'] = y

        return self.tvp_template


    def __init__(self, init_state):

        self.cache = CostCache()
        self.index = 0

        # Get the trasformation between odom and world
        self.init_position = get_position()
        self.cache.set_T(self.init_position)

        # Initialize ROS node
        self.pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=1)

        # Create a Twist message for robot motion
        self.move_cmd = Twist()

        # Set the rate for the ROS loop
        self.rate = rospy.Rate(10)

        self.model = Model().get_model()

        #mpc controller INIT
        setup_mpc = {
            'n_horizon': 20,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
            'supress_ipopt_output': True
        }

        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**setup_mpc)

        self.set_cost_function()
        self.set_bounds()

        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)

        self.mpc.setup()
        
        # Set initial state for simulations
        x0 = np.array(init_state).reshape(-1, 1)
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()


    def update(self):

        # Get the robot's current states (position and orientation)
        actual_position = get_actual_position(self.init_position)
        states = numpy.array([actual_position[0], actual_position[1], actual_position[5]]).reshape(-1, 1)
        print(states)

        #Implement disturbance on the model
        states_noise = np.array(add_noise_to_states(states)).reshape(-1,1)

        # Perform MPC step to get the control input
        u = self.mpc.make_step(states)

        #Change reference if the previous target got
        if u[0] <= 0.5 and u[1] <= 0.5:
            self.index+=1

        # Set the linear and angular velocities for the robot's motion
        self.move_cmd.linear.x = u[0] 
        self.move_cmd.angular.z = u[1]

        # Publish the motion command
        self.pub.publish(self.move_cmd)

        # Sleep according to the defined rate
        self.rate.sleep()


    """ METHOD FOR THE __init__ and update Method (utility)"""

    def set_bounds(self):
        # Set lower bounds on states
        self.mpc.bounds['lower', '_x', 'x'] = 0
        self.mpc.bounds['lower', '_x', 'y'] = 0

        self.mpc.bounds['upper', '_x', 'x'] = 18
        self.mpc.bounds['upper', '_x', 'y'] = 18

        self.mpc.bounds['lower', '_x', 'theta'] = -np.pi
        self.mpc.bounds['upper', '_x', 'theta'] = np.pi

        # Set lower bounds on inputs
        self.mpc.bounds['lower', '_u', 'v'] = -1
        self.mpc.bounds['lower', '_u', 'w'] = -1

        # Set upper bounds on inputs
        self.mpc.bounds['upper', '_u', 'v'] = 1
        self.mpc.bounds['upper', '_u', 'w'] = 1


    def set_cost_function(self):       

        mterm = (self.model.x['x'] - self.model.tvp['xd'])**2 + (self.model.x['y'] - self.model.tvp['yd'])**2 
        lterm = mterm + 1/2*self.model.u['v']**2 + 1/2*self.model.u['w']**2 

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
