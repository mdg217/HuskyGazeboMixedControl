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

class MPC_model():
    
    def init(self, xd, yd):
        # do-mpc implementation
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        # Define state variables
        x = self.model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        y = self.model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
        theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))

        # Define control variables
        v = self.model.set_variable(var_type='_u', var_name='v')
        w = self.model.set_variable(var_type='_u', var_name='w')

        # Define state equations
        self.model.set_rhs('x', v * np.cos(theta))
        self.model.set_rhs('y', v * np.sin(theta))
        self.model.set_rhs('theta', w)
        self.model.setup()

        setup_mpc = {
            'n_horizon': 20,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
        }
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**setup_mpc)

        # Set reference points for state variables
        self.xd = xd
        self.yd = yd

        # Define cost functions
        mterm = (x - self.xd)**2 + (y - self.yd)**2  # + 0.01 * (theta)**2  # lyapunov
        lterm = (x - self.xd)**2 + (y - self.yd)**2 + 1/2 * v**2 + 1/2 * w**2

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        
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

        # Initialize simulator for model simulations
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=0.1)
        simulator.setup()
        
        # Set initial state for simulations
        x0 = np.array([0, 0, 0]).reshape(-1, 1)
        print(x0)
        simulator.x0 = x0
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    # Method to return the initialized MPC model
    def getModel(self):
        return self.mpc
