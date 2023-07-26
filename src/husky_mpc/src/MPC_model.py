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
from tf import transformations as t
import tf

class MPC_model():

    def __init__(self, xd, yd, init_state): 
        #model INIT
        self.cache = CostCache()

        Tobs1 = t.concatenate_matrices(t.translation_matrix([-9.675000, -0.754599, 0.5]),
                t.quaternion_matrix(t.quaternion_from_euler(0, 0, -1.570801)))
        
        obs1 = np.dot(t.inverse_matrix(self.cache.get_T()), Tobs1)

        x1 = tf.transformations.translation_from_matrix(obs1)[0]
        y1 = tf.transformations.translation_from_matrix(obs1)[1]

        print(x1)
        print(y1)

        Tobs2 = t.concatenate_matrices(t.translation_matrix([-5.674, -2.7733, 0.5]),
                t.quaternion_matrix(t.quaternion_from_euler(0, 0, 1.57)))
        
        obs2 = np.dot(t.inverse_matrix(self.cache.get_T()), Tobs2)

        x2 = tf.transformations.translation_from_matrix(obs2)[0]
        y2 = tf.transformations.translation_from_matrix(obs2)[1]

        print(x2)
        print(y2)

        # do-mpc implementation
        model_type = 'continuous'  # either 'discrete' or 'continuous'

        self.n_horizon = 20
        self.xd = xd
        self.yd = yd

        self.model = do_mpc.model.Model(model_type)

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

        #break class


        #mpc controller INIT
        setup_mpc = {
            'n_horizon': self.n_horizon,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
            'supress_ipopt_output': True
        }

        #self.mpc.supress_ipopt_output()
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**setup_mpc)

        # Set reference points for state variables

        #oterm1 = 50*np.exp(-((self.x-x1)**2)/(37.5) - ((self.y-y1)**2)/(1/50))
        oterm2 = 20*np.exp(-((self.x-x2)**2)/(1) - ((self.y-y2)**2)/(20))
        mterm = 0.1*(self.x - self.xd)**2 + 0.1*(self.y - self.yd)**2 + oterm2
        lterm = mterm + 1/2*self.v**2 + 1/2*self.w**2 
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

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

        self.mpc.setup()
        
        # Set initial state for simulations
        x0 = np.array(init_state).reshape(-1, 1)
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        

    # Method to return the initialized MPC model
    def getModel(self):
        return self.mpc
