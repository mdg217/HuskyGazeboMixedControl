import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log
from obstacle import *
import rospy
from Plants.uniform_plant import *
from Plants.linearized_plant import *
from Plants.trajectory_based_plant import *
from time import *
import rospy
from geometry_msgs.msg import Twist
from tf import transformations as t
import numpy as np
from utility import *
import do_mpc
from casadi import *
import numpy as np
from cost_cache import *
from model import *

"""
ControllerKLC: A class implementing a Kinematic Linearization Controller (KLC) for robot motion planning.

This class defines methods to initialize the controller and update its behavior based on robot states.
"""
class OnlineKLCMPC:


    """
    A method to calculate the time-varying parameters (TVP) for the MPC controller.
    
    :param self: The instance of the class.
    :return: The time-varying parameters for the controller.
    """
    def tvp_fun(self, time):

        for k in range(21):
                state = self.cache.get_next_state()
                self.tvp_template['_tvp', k, 'xd'] = state[0]
                self.tvp_template['_tvp', k, 'yd'] = state[1]

        return self.tvp_template
    


    """
    Update the controller's behavior based on the current state.
    
    :param self: The instance of the class.
    :return: Lists containing mean x position, mean y position, and time.
    """
    def __init__(self, goal, mode, init_state):

        self.cache = CostCache()

        rospy.init_node('husky', anonymous=True)

        # Get the trasformation between odom and world
        self.init_position = get_position()
        self.cache.set_T(self.init_position)

        self.obstacles = Obstacle()

        #Target definition
        self.goal = goal
        self.xd = goal[0]
        self.yd = goal[1]
        
        #Dimensions of the variables
        self.zdim = 2

        #Minimum values
        self.zmin = [0, 0] 

        #Discretization steps
        self.zstep = [0.5, 0.5]

        #Amount of discrete bins
        self.zdiscr = [36, 36]

        #Number of iterations for the simulations
        self.zsim = 1
        #Duration of the simulation
        self.duration = 100

        # Creazione del vettore 4D inizializzato con zeri

        self.passive_dynamics = np.zeros((self.zdiscr[0], self.zdiscr[0], self.zdiscr[0], self.zdiscr[0]))

        if mode == 0:
            self.passive_dynamics = uniform_plant().get_plant(self.zdiscr[0])
        elif mode == 1:
            self.passive_dynamics = linearized_plant().get_plant(2)
        elif mode == 2:
            self.passive_dynamics = trajectory_based_plant().get_plant(2, uniform_plant().get_plant(self.zdiscr[0]))       


        self.stateVect = np.zeros((self.zdiscr[0]**2, 2))

        for i in range(self.zdiscr[0]):
            #Enumerate the states from 1 to 36^2. Here we explicitly build the enumeration to later build the Q matrix and for convenience
            for j in range(self.zdiscr[0]):
                # Compute the angle and speed values for the current state
                x = (i)*self.zstep[0]
                y = (j)*self.zstep[0]
                # Calculate the index of the current state in the state vector
                ind = i*self.zdiscr[0] + j
                # Assign the angle and speed values to the state vector
                self.stateVect[ind] = [x, y] 

        self.Prob = np.zeros((self.zdiscr[0]**2, self.zdiscr[0]**2))

        for i in range(self.zdiscr[0]):
            for j in range(self.zdiscr[0]):
                pf = self.passive_dynamics[i,j]
                ind1 = i*self.zdiscr[0] + j
                self.Prob[ind1] = self.unravelPF(pf)

        self.z = np.array((np.shape(self.Prob))[0])

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

        self.cache.set_next_state([0, 0])

        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)

        self.mpc.setup()
        
        # Set initial state for simulations
        x0 = np.array(init_state).reshape(-1, 1)
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

        self.fullH = np.zeros((self.zsim,self.duration))
        self.fullHv = np.zeros((self.zsim,self.duration))
        self.diagMinusQ = np.zeros((self.zdiscr[0]**2, self.zdiscr[0]**2)) # q
        self.actions = 0.5*np.array([(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)])
        self.possible_states = []

    
        """
    Update the controller and apply the calculated control input to the robot.
    
    :param self: The instance of the class.
    """
    def update(self):

        stop_cond = 0
        hist = [[0,0]]*self.duration
        state = [0, 0] #Initialize the pendulum <------------
        self.cache.set_next_state(state)
        for i in range(self.duration): #For each step     

            # Get the robot's current states (position and orientation)
            actual_position = get_actual_position(self.init_position)
            actual_state = numpy.array([int(actual_position[0]*1000)/1000, int(actual_position[1]*1000)/1000, actual_position[5]]).reshape(-1, 1)
        

            for action in self.actions:
                is_good_state = [actual_state[0].item() + action[0], actual_state[1].item() + action[1]]
                print(is_good_state)

                if 0 <= is_good_state[0] < self.zdiscr[0]*self.zstep[0] and 0 <= is_good_state[1] < self.zdiscr[1]*self.zstep[1]:
                    state_index = 0
                    is_good_state = [self.n_discretize(is_good_state[0].item(), self.zdim, self.zmin, self.zstep), self.n_discretize(is_good_state[1].item(), self.zdim, self.zmin, self.zstep)]
                    print(is_good_state)

                    for s in self.stateVect:
                        if s[0] == is_good_state[0] and s[1] == is_good_state[1]:
                            #print(s)
                            self.possible_states.append(state_index)
                        state_index+=1
            
            print(actual_state)
            print(self.possible_states)
            print("-----------------------------------------")
                
            for k in self.possible_states:
                    #print("Il valore del vett in pos " + str(k) + " è: " + str(self.stateVect[k]))
                    #Build the diagonal matrix with the exponential of the opposite of the cost
                self.diagMinusQ[k,k] = np.exp(-self.cost(self.stateVect[k]))
                #print(self.diagMinusQ[k,k])

            self.z = self.powerMethod(self.diagMinusQ@self.Prob, self.zdiscr[0]**2)
            self.possible_states = []

            print("power method")

            hist[i]=[actual_state[0], actual_state[1]] #Log the state
            next_state = self.loop(hist[i]) #Sample the new state
            print("lo stato successivo è: " + str(actual_state))
            self.cache.set_next_state(next_state)
            

            while True:
                actual_position = get_actual_position(self.init_position)
                actual_state = numpy.array([actual_position[0], actual_position[1], actual_position[5]]).reshape(-1, 1)

                # Perform MPC step to get the control input
                print("sono nel while dell'mpc")
                u = self.mpc.make_step(actual_state)

                if abs(actual_state[0]-next_state[0])<=0.1 and abs(actual_state[1]-next_state[1])<=0.1:
                    break

                # Set the linear and angular velocities for the robot's motion
                self.move_cmd.linear.x = u[0] 
                self.move_cmd.angular.z = u[1]

                # Publish the motion command
                self.pub.publish(self.move_cmd)

                # Sleep according to the defined rate
                self.rate.sleep()


    """ METHOD FOR THE __init__ and update Method (utility)"""


    """
    Set the bounds for states and control inputs for the MPC controller.
    
    :param self: The instance of the class.
    """
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


    """
    Set the cost function for the MPC controller.
    
    :param self: The instance of the class.
    """
    def set_cost_function(self):       

        mterm = 2*(self.model.x['x'] - self.model.tvp['xd'])**2 + 2*(self.model.x['y'] - self.model.tvp['yd'])**2 
        lterm = mterm + 1/2*self.model.u['v']**2 + 1/2*self.model.u['w']**2 

        self.mpc.set_objective(mterm=mterm, lterm=lterm)


    """
    Discretize the continuous state variables.
    
    :param Z: The continuous state variables.
    :param Zdim: The dimensionality of the variables.
    :param Zmin: The minimum values for each dimension.
    :param Zstep: The discretization steps for each dimension.
    :return: The discretized indices.
    """

    def n_discretize(self, x, dim, zmin, zstep):
        ind = int((x-zmin)/zstep)
        return min +ind*zstep

    def discretize(self, Z, Zdim, Zmin, Zstep):
        res = [0]*Zdim #n-dimensional index
        for i in range(Zdim): #For each dimension
            elt = Z[i]#Extract the i-th element
            ind = int((elt - Zmin[i])//Zstep[i]) #Discretize
            res[i] = ind
        return(tuple(res)) #Return as tuple for array indexing
    
    
    """
    Calculate the cost of a given state.
    
    :param state: The state to calculate the cost for.
    :return: The calculated cost.
    """
    def cost(self, state):
        k = 20
        sx = 0.7
        sy = 0.7

        obsTerm = 0

        #ADD VISION OF THE ROBOT FOR THE OBSTACLES
        for obs in self.obstacles.get_obs():
            
            if(self.is_obstacle_in_fov(state[0], state[1], obs[0], obs[1]) == True):
                xterm = ((state[0] - obs[0]) / sx) ** 2
                yterm = ((state[1] - obs[1]) / sy) ** 2
                obsTerm += k * np.exp(-0.5 * (xterm + yterm))

        # Calculate the distance from the goal and introduce a regularization term for 2TypeSimulation
        distance_to_goal = np.sqrt((state[0] - self.xd) ** 2 + (state[1] - self.yd) ** 2)
        regularization_term = 0.1 * distance_to_goal  # Adjust the scaling factor as needed

        # Include the regularization term in the overall cost calculation 
        return 0.01*(state[0] - self.xd) ** 2 + 0.01*(state[1] - self.yd) ** 2 + obsTerm + regularization_term


    def is_obstacle_in_fov(self, rover_x, rover_y, obs_x, obs_y):
        
        fov_radius = 2.0  # Raggio del campo visivo del rover
        
        # Calcola la distanza tra il rover e l'ostacolo
        distance = np.sqrt((rover_x - obs_x)**2 + (rover_y - obs_y)**2)
        
        # Verifica se l'ostacolo è all'interno del campo visivo del rover
        if distance <= fov_radius:
            return True  # Ostacolo è nel campo visivo
        else:
            return False  # Ostacolo non è nel campo visivo
    
    
    """
    Unravel a 2D passive dynamics array into a 1D array.
    
    :param pf: The 2D passive dynamics array.
    :return: The unraveled 1D array.
    """
    def unravelPF(self, pf):
    
        res = np.zeros(self.zdiscr[0]**2)
        for i in range(self.zdiscr[0]):
            for j in range(self.zdiscr[0]):
                res[i*self.zdiscr[0]+j] = pf[i][j]
        return(res)
    
    
    """
    Perform the power method for eigenvalue estimation.
    
    :param mat: The matrix for eigenvalue estimation.
    :param dim: The dimensionality of the matrix.
    :return: The estimated eigenvector.
    """
    def powerMethod(self, mat, dim, epsilon=1e-6):
        vect = np.ones(dim)
        nrm = np.linalg.norm(vect)
        
        for i in range(self.zsim):
            prev_vect = vect.copy()  # Salva l'autovettore dell'iterazione precedente
            vect = mat.dot(vect)
            nrm = np.linalg.norm(vect)
            vect = vect / nrm
            
            """# Calcola la differenza tra l'autovettore attuale e quello precedente
            diff = np.linalg.norm(vect - prev_vect)
            
            # Verifica la condizione di arresto
            if diff < epsilon:
                break"""
        
        return vect

    
    """
    Perform the power method for eigenvalue estimation.
    
    :param mat: The matrix for eigenvalue estimation.
    :param dim: The dimensionality of the matrix.
    :return: The estimated eigenvector.
    """
    def loop(self, x):
    
        ind = self.discretize(x,  self.zdim, self.zmin, self.zstep) #Discretize the state
        print("indice : " +str(ind))
        pf = self.passive_dynamics[ind[0],ind[1]] #Get the pf corresponding to the passive dynamics
        print("pf: ", pf)
        pf_1D = self.unravelPF(pf) #Unravel it
        print("unpf: ", pf_1D)
        print("vettore di des: " +str(self.z))
        pf_weighted = pf_1D*self.z #Calculate the actual transition pf using z and the passive dynamics
        print("pf_w: " + str(pf_weighted))
        S = np.sum(pf_weighted) #Normalize
        print("S: "+ str(S))
        pf_weighted = pf_weighted/S #probabilities contain NaN ERRORE SPESSO USCITO FUORI forse perché non si riesce a minimizzare la funzione di costo a causa di qualche limite raggiunto
        print(pf_weighted)
        ind = np.random.choice(range(self.zdiscr[0]**2), p=pf_weighted) #Get the new (enumerated) state index using the calculated dynamics
        newState = self.stateVect[ind] #Get the new state from the state vector
        return(newState)
    
klc_controller = OnlineKLCMPC([16, 16], 0, [0, 0, 0])
klc_controller.update()