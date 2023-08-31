import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log
from obstacle import *
import rospy
from Plants.uniform_plant import *
from Plants.linearized_plant import *
from Plants.trajectory_based_plant import *

"""
ControllerKLC: A class implementing a Kinematic Linearization Controller (KLC) for robot motion planning.

This class defines methods to initialize the controller and update its behavior based on robot states.
"""
class ControllerKLCVision:

    """
    Update the controller's behavior based on the current state.
    
    :param self: The instance of the class.
    :return: Lists containing mean x position, mean y position, and time.
    """
    def __init__(self, goal, mode):

        self.cache = CostCache()

        rospy.init_node('husky', anonymous=True)

        # Get the trasformation between odom and world
        self.init_position = get_position()
        self.cache.set_T(self.init_position)

        self.obstacles = Obstacle()

        #Target definition
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
        self.zsim = 15

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

        print(np.shape(self.stateVect[0]))

        self.Prob = np.zeros((self.zdiscr[0]**2, self.zdiscr[0]**2))

        for i in range(self.zdiscr[0]):
            for j in range(self.zdiscr[0]):
                pf = self.passive_dynamics[i,j]
                ind1 = i*self.zdiscr[0] + j
                self.Prob[ind1] = self.unravelPF(pf)

        self.z = np.array((np.shape(self.Prob))[0])

        diagMinusQ = np.zeros((self.zdiscr[0]**2, self.zdiscr[0]**2)) # q

        for i in range(self.zdiscr[0]**2):
            #Build the diagonal matrix with the exponential of the opposite of the cost
            diagMinusQ[i,i] = np.exp(-self.cost(self.stateVect[i]))
                                     
        self.z = self.powerMethod(diagMinusQ@self.Prob, self.zdiscr[0]**2) 

    
    """
    Update the controller's behavior based on the current state.
    
    :param self: The instance of the class.
    :return: Lists containing mean x position, mean y position, and time.
    """
    def update(self):
        fullH = np.zeros((self.zsim,self.duration))
        fullHv = np.zeros((self.zsim,self.duration))
        nSteps = self.duration

        #Task:  obtain simulations for different initial conditions (say, 5 different initial conditions). For each of these, run 50 simulations.

        for j in range(self.zsim): #Perform 50 simulations
            #print("simulazione numero: " + str(j))
            hist = [[0,0]]*nSteps
            state = [0, 0] #Initialize the pendulum <------------
            for i in range(nSteps): #For each step
                hist[i]=state #Log the state
                state = self.dynamic_programming_next_state(state) #Sample the new state


            fullH[j] = [x[0] for x in hist]
            fullHv[j] = [x[1] for x in hist]

        meanx = [0]*self.duration #Get the means and stds for plotting
        stds = [0]*self.duration
        for i in range(self.duration):
            meanx[i] = np.mean(fullH[:,i])
            stds[i] = np.std(fullH[:,i])

        meany = [0]*self.duration #Get the means and stds for plotting
        stdsv = [0]*self.duration
        for i in range(self.duration):
            meany[i] = np.mean(fullHv[:,i])
            stdsv[i] = np.std(fullHv[:,i])

        time = np.array([time/10 for time in range(self.duration)])

        return [meanx, meany, time]

    
    # Utility methods for init and update methods


    """
    Discretize the continuous state variables.
    
    :param Z: The continuous state variables.
    :param Zdim: The dimensionality of the variables.
    :param Zmin: The minimum values for each dimension.
    :param Zstep: The discretization steps for each dimension.
    :return: The discretized indices.
    """
    def discretize(self, Z, Zdim, Zmin, Zstep):
        res = [0]*Zdim #n-dimensional index
        for i in range(Zdim): #For each dimension
            elt = Z[i] #Extract the i-th element
            ind = int((elt - Zmin[i])//Zstep[i]) #Discretize
            print("indice calcolato: " + str(ind))
            res[i] = ind
        return(tuple(res)) #Return as tuple for array indexing
    
    
    """
    Calculate the cost of a given state.
    
    :param state: The state to calculate the cost for.
    :return: The calculated cost.
    """
    def cost(self, state):
        k = 30
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
        return 0.5*(state[0] - self.xd) ** 2 + 0.5*(state[1] - self.yd) ** 2 + obsTerm + regularization_term
    
    
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
        pf = self.passive_dynamics[ind[0],ind[1]] #Get the pf corresponding to the passive dynamics
        pf_1D = self.unravelPF(pf) #Unravel it
        pf_weighted = pf_1D*self.z #Calculate the actual transition pf using z and the passive dynamics
        S = np.sum(pf_weighted) #Normalize
        pf_weighted = pf_weighted/S #probabilities contain NaN ERRORE SPESSO USCITO FUORI forse perché non si riesce a minimizzare la funzione di costo a causa di qualche limite raggiunto
        ind = np.random.choice(range(self.zdiscr[0]**2), p=pf_weighted) #Get the new (enumerated) state index using the calculated dynamics
        newState = self.stateVect[ind] #Get the new state from the state vector
        return(newState)
    

    def export_metrics(self, x, y, time):
        np.save("klc_vision_linear_results_from_planning", np.array([x, y, time]))

    
    def dynamic_programming_next_state(self, state):
        ind = self.discretize(state, self.zdim, self.zmin, self.zstep)  # Discretize the state
        current_state = self.passive_dynamics[ind[0], ind[1]]

        min_cost = float('inf')  # Initialize with a high value
        best_next_state = None

        # Define possible adjacent indices including diagonals
        adjacent_indices = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]

        for di, dj in adjacent_indices:
            new_i, new_j = ind[0] + di, ind[1] + dj

            if 0 <= new_i < self.zdiscr[0] and 0 <= new_j < self.zdiscr[0]:
                pf = self.passive_dynamics[new_i, new_j]
                transition_prob = self.unravelPF(pf)
                expected_cost = np.dot(transition_prob, self.z)  # Calculate the expected cost using V
                cost = self.cost(self.stateVect[new_i * self.zdiscr[0] + new_j]) + expected_cost

                if cost < min_cost:
                    min_cost = cost
                    best_next_state = self.stateVect[new_i * self.zdiscr[0] + new_j]

        return best_next_state



print("Prova del sistema KLC")

klc_controller = ControllerKLCVision([16, 16], 1)
x, y, time = klc_controller.update()
print(x[-1])
print(y[-1])

# Crea una griglia di subplot con 1 riga e 2 colonne
fig, axs = plt.subplots(1, 1, figsize=(10, 5))

# Plot del primo subplot
axs.plot(x, y, marker='o', linestyle='-', color='r')
axs.set_xlabel('X Position')
axs.set_ylabel('Y Position')
axs.set_title('Primo Plot')
for obs in klc_controller.obstacles.get_obs():
    axs.scatter(obs[0], obs[1], color='r', s=1000)

# Regola la spaziatura tra i subplot
plt.tight_layout()

# Mostra i subplot
plt.show()