import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log
import os
from obstacle import *
import time
import rospy
from Plants.uniform_plant import *
from Plants.linearized_plant import *
from Plants.trajectory_based_plant import *

"""
ControllerKLC: A class implementing a Kinematic Linearization Controller (KLC) for robot motion planning.
This class defines methods to initialize the controller and update its behavior based on robot states.
"""
class ControllerKLCWoodbury:

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
        self.duration = 80

        # Creazione del vettore 4D inizializzato con zeri
        self.passive_dynamics = trajectory_based_plant().get_plant(2, uniform_plant().get_plant(36))
        
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
        
        self.diagMinusQ = np.zeros((self.zdiscr[0]**2, self.zdiscr[0]**2)) # q

        for i in range(self.zdiscr[0]**2):
            #Build the diagonal matrix with the exponential of the opposite of the cost
            self.diagMinusQ[i,i] = np.exp(-self.cost(self.stateVect[i]))
                
    """
    Update the controller's behavior based on the current state.
    
    :param self: The instance of the class.
    :return: Lists containing mean x position, mean y position, and time.
    """
    def update(self):
        fullH = np.zeros((self.zsim,self.duration))
        fullHv = np.zeros((self.zsim,self.duration))
        nSteps = self.duration
        #diagMinusQ = np.zeros((self.zdiscr[0]**2, self.zdiscr[0]**2)) # q
        newdiagMinusQ = np.zeros((self.zdiscr[0]**2, self.zdiscr[0]**2)) # q
        actions = 0.5*np.array([(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)])
        possible_states = []
        oldz = None

        #Task:  obtain simulations for different initial conditions (say, 5 different initial conditions). For each of these, run 50 simulations.

        for j in range(self.zsim): #Perform simulations
            print("simulazione numero: " + str(j))
            hist = [[0,0]]*nSteps
            state = [0, 0] #Initialize the pendulum <------------
            for i in range(nSteps): #For each step

                #compute new possible states from statevect using the current state
                
                #Verificare prima quali stati sono disponibili a partire da state per andare in state + action nella 
                #compute di index of the next states
                for action in actions:
                    is_good_state = state + action
                    #print(is_good_state)
                    if 0 <= is_good_state[0] < self.zdiscr[0]*self.zstep[0] and 0 <= is_good_state[1] < self.zdiscr[1]*self.zstep[1]:
                        state_index = 0
                        for s in self.stateVect:
                            if s[0] == is_good_state[0] and s[1] == is_good_state[1]:
                                #print(s)
                                possible_states.append(state_index)
                                #print(possible_states)
                            state_index+=1

                for k in possible_states:
                    #Build the diagonal matrix with the exponential of the opposite of the cost
                    newdiagMinusQ[k,k] = np.exp(-self.cost(self.stateVect[k]))

                if i == 0:
                    self.z = self.powerMethod(self.diagMinusQ@self.Prob, self.zdiscr[0]**2)
                    oldz = self.z

                hist[i]=state #Log the state
                state = self.loop(state) #Sample the new state
                
                current_state = 0
                for s in self.stateVect:
                    if s[0] == state[0] and s[1] == state[1]:
                        break
                    current_state+=1

                self.z = self.woodburyMethod(self.Prob, self.diagMinusQ, self.diagMinusQ, possible_states, oldz)

                possible_states = []

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
                #print("ho trovato un nuovo ostacolo")
                xterm = ((state[0] - obs[0]) / sx) ** 2
                yterm = ((state[1] - obs[1]) / sy) ** 2
                obsTerm += k * np.exp(-0.5 * (xterm + yterm))


        # Include the regularization term in the overall cost calculation 
        return 0.05*(state[0] - self.xd) ** 2 + 0.05*(state[1] - self.yd) ** 2 + obsTerm 
    

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
    
    
    def woodburyMethod(self, P0, Q, Q0, indices, oldz):
        z = np.zeros((P0.shape[0], ))

        for i in indices:
            L0 = Q0-P0
            D0 = np.linalg.inv(L0)
            
            P = P0
            P[i,:] = 0
            P[i,i] = 1
            L = Q-P
            
            rows, cols = L.shape

            """for i in range(rows):
                for j in range(cols):
                    if L[i, j] != L0[i, j]:
                        print(f"Differenza nella posizione ({i}, {j}): {L[i, j]} (matrix1) vs {L0[i, j]} (matrix2)")
"""
            ds = np.where(np.sum(np.abs(L - L0), axis=1) != 0)[0]

            e = np.zeros((P.shape[0], 1))
            e[ds] = 1 # 1296

            d = L[ds,:] - L0[ds,:]
            if d.shape[0] == 0:
                return np.sum([z, oldz], axis=0)

            terms = np.diag(P) == 1
            p = P[:, terms]
            p[terms, :] = 0
            p[terms, :] = 0
            
            m0 = np.dot(D0, e)
            z0 = np.dot(D0, p)
        
            alpha = 1 / (1 + np.dot(d, m0))
            

            a1 = np.dot(d, z0)
            #print("il vettore a1: " +str(a1))
            a2 = np.dot(m0, a1)
            
            zc = z0 - np.dot(alpha.item(), a2)

            cost = np.exp(-Q[terms, terms])
            newz = np.zeros((P0.shape[0], ))

            if np.shape(zc)[1] == 1:
                newz = zc*cost
            else:
                for k in range(np.shape(zc)[0]):
                    for j in range(np.shape(cost)[0]):
                        newz[k] += zc[k, j]*cost[j]      

            newz[newz < 0] = 0
            newz = newz.flatten()
            newz = np.abs(newz)

            z = np.sum([z, newz], axis=0)
        
        #print(z)

        return z


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
        np.save("klc_gaussian_results_from_planning", np.array([x, y, time]))


print("Prova del sistema KLC")

klc_controller = ControllerKLCWoodbury([16, 16], 0)
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