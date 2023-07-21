"""
The MPC_model class implements a Model Predictive Control (MPC) model. The class contains an init method 
to initialize the MPC model with state variables (x, y, theta) and control variables (v, w). A state equation 
is defined for each state variable, which determines how the state variable changes over time based on the 
control variables. An objective is also set for the MPC model, with a short-term cost function (mterm) and 
a long-term cost function (lterm). Lower and upper bounds are set for the state variables and control variables. 
A simulator is also initialized to perform model simulations. Finally, the class provides a getModel method 
to return the initialized MPC model.
"""

# Import the do_mpc package:
import do_mpc
from casadi import *
import numpy as np
from CostCache import *
from ObstacleCircle import *

class MPC_model():
    
    def tvp_fun(self, t_now):

        for k in range(self.n_horizon+1):
            obs = self.cache.get_cost()

            Rat = np.sqrt((self.x - self.xd)**2 + (self.y - self.yd)**2)
            Rrep = np.sqrt((self.x - obs[0])**2 + (self.y - obs[1])**2) - obs[2]
            print(Rrep)
            obs_a = (Rat/(Rrep**2 + 0.1))
            obs_b = (Rat/(Rrep**2 + 0.1))

            self.tvp_template['_tvp', k, 'obs'] = obs_a + obs_b

            print("Valore dello stato temp: " + str(self.model.tvp['obs']))
            print("Valore dato: " + str(self.tvp_template['_tvp', k, 'obs']))
        return self.tvp_template

    def __init__(self, xd, yd, init_state):
        # do-mpc implementation
        model_type = 'continuous'  # either 'discrete' or 'continuous'

        self.n_horizon = 20
        self.xd = xd
        self.yd = yd

        self.model = do_mpc.model.Model(model_type)
        self.cache = CostCache()

        # Define obstacle cost time-variing variable
        self.obs = self.model.set_variable(var_type='_tvp', var_name='obs')

        # Define state variables 
        self.x = self.model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        self.y = self.model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
        self.theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))

        # Define control variables
        self.v = self.model.set_variable(var_type='_u', var_name='v')
        self.w = self.model.set_variable(var_type='_u', var_name='w')

        # Define state equations
        self.model.set_rhs('x', self.v * np.cos(self.theta))
        self.model.set_rhs('y', self.v * np.sin(self.theta))
        self.model.set_rhs('theta', self.w)
        self.model.setup()

        setup_mpc = {
            'n_horizon': self.n_horizon,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
        }

        #self.mpc.supress_ipopt_output()
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**setup_mpc)

        # Set reference points for state variables

        E = [self.x - xd ,self.y - yd, 0]
        Q = [[1/(self.x - xd)**2 * self.model.tvp['obs'], 0, 0],[0, 1/(self.y - yd)**2 * self.model.tvp['obs'], 0],[0, 0, 0]]

        mterm = (E[0])**2 + (E[1])**2 * np.dot(np.dot(np.transpose(E), Q), E)
        lterm = mterm + 1/2*self.v**2 + 1/2*self.w**2 
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)

        # Set lower bounds on states
        #self.mpc.bounds['lower', '_x', 'x'] = 0
        #self.mpc.bounds['lower', '_x', 'y'] = 0
        self.mpc.bounds['lower', '_x', 'theta'] = -np.pi
        self.mpc.bounds['upper', '_x', 'theta'] = np.pi

        # Set lower bounds on inputs
        self.mpc.bounds['lower', '_u', 'v'] = -1
        self.mpc.bounds['lower', '_u', 'w'] = -1

        # Set upper bounds on inputs
        self.mpc.bounds['upper', '_u', 'v'] = 1
        self.mpc.bounds['upper', '_u', 'w'] = 1

        self.mpc.setup()
        
        # Set initial state for simulations
        x0 = np.array(init_state).reshape(-1, 1)
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        

    # Method to return the initialized MPC model
    def getModel(self):
        return self.mpc
