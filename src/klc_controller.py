import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log
import os
from obstacle import *
import rospy

class KLC_controller:

    def __init__(self, goal):

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
        self.Zdim = 2

        #Minimum values
        self.Zmin = [0, 0] 

        #Discretization steps
        self.Zstep = [0.5, 0.5]

        #Amount of discrete bins
        self.Zdiscr = [36, 36]

        #Number of iterations for the simulations
        self.Zsim = 30

        #Duration of the simulation
        self.duration = 100

        # Creazione del vettore 4D inizializzato con zeri
        self.passive_dynamics = np.zeros((self.Zdiscr[0] , self.Zdiscr[0], self.Zdiscr[0], self.Zdiscr[0]))

        # Popolamento delle transizioni per gli stati adiacenti
        for row in range(self.Zdiscr[0]):
            for col in range(self.Zdiscr[0]):
                # Stato attuale
                current_state = (row, col)

                # Transizioni possibili: su, giù, sinistra, destra
                possible_transitions = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

                for dr, dc in possible_transitions:
                    next_row = row + dr
                    next_col = col + dc

                    # Verifica se la prossima posizione è all'interno della griglia
                    if 0 <= next_row < self.Zdiscr[0] and 0 <= next_col < self.Zdiscr[0]:
                        # Imposta la probabilità della transizione da current_state a next_state
                        self.passive_dynamics[row, col, next_row, next_col] = 1.0 / len(possible_transitions)

        # Impostazione delle celle di transizione dallo stato attuale a se stesso a 0
        for row in range(self.Zdiscr[0]):
            for col in range(self.Zdiscr[0]):
                self.passive_dynamics[row, col, row, col] = 0       


        self.stateVect = np.zeros((self.Zdiscr[0]**2, 2))

        for i in range(self.Zdiscr[0]):
            #Enumerate the states from 1 to 36^2. Here we explicitly build the enumeration to later build the Q matrix and for convenience
            for j in range(self.Zdiscr[0]):
                # Compute the angle and speed values for the current state
                x = (i)*self.Zstep[0]
                y = (j)*self.Zstep[0]
                # Calculate the index of the current state in the state vector
                ind = i*self.Zdiscr[0] + j
                # Assign the angle and speed values to the state vector
                self.stateVect[ind] = [x, y] 

        diagMinusQ = np.zeros((self.Zdiscr[0]**2, self.Zdiscr[0]**2))

        for i in range(self.Zdiscr[0]**2):
            #Build the diagonal matrix with the exponential of the opposite of the cost
            diagMinusQ[i,i] = np.exp(-self.cost(self.stateVect[i]))

        Prob = np.zeros((self.Zdiscr[0]**2, self.Zdiscr[0]**2))

        for i in range(self.Zdiscr[0]):
            for j in range(self.Zdiscr[0]):
                pf = self.passive_dynamics[i,j]
                ind1 = i*self.Zdiscr[0] + j
                Prob[ind1] = self.unravelPF(pf)

        self.z = self.powerMethod(diagMinusQ@Prob, self.Zdiscr[0]**2) 

        

    
    def update(self):
        fullH = np.zeros((self.Zsim,self.duration))
        fullHv = np.zeros((self.Zsim,self.duration))
        nSteps = self.duration

        #Task:  obtain simulations for different initial conditions (say, 5 different initial conditions). For each of these, run 50 simulations.

        for j in range(self.Zsim): #Perform 50 simulations
            hist = [[0,0]]*nSteps
            state = [0, 0] #Initialize the pendulum <------------
            for i in range(nSteps): #For each step
                hist[i]=state #Log the state
                state = self.loop(state) #Sample the new state
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

    def discretize(self, Z, Zdim, Zmin, Zstep):
        res = [0]*Zdim #n-dimensional index
        for i in range(Zdim): #For each dimension
            elt = Z[i] #Extract the i-th element
            ind = int((elt - Zmin[i])//Zstep[i]) #Discretize
            res[i] = ind
        return(tuple(res)) #Return as tuple for array indexing
    
    
    def cost(self, state):
        k = 50
        sx = 0.7
        sy = 0.7

        obsTerm = 0

        for obs in self.obstacles.get_obs():
            xterm = ((state[0] - obs[0]) / sx) ** 2
            yterm = ((state[1] - obs[1]) / sy) ** 2
            obsTerm += k * np.exp(-0.5 * (xterm + yterm))

        # Calculate the distance from the goal and introduce a regularization term
        distance_to_goal = np.sqrt((state[0] - self.xd) ** 2 + (state[1] - self.yd) ** 2)
        regularization_term = 0.01 * distance_to_goal  # Adjust the scaling factor as needed

        # Include the regularization term in the overall cost calculation
        return 0.1*(state[0] - self.xd) ** 2 + 0.1*(state[1] - self.yd) ** 2 + obsTerm + regularization_term

    
    def unravelPF(self, pf):
    
        res = np.zeros(self.Zdiscr[0]**2)
        for i in range(self.Zdiscr[0]):
            for j in range(self.Zdiscr[0]):
                res[i*self.Zdiscr[0]+j] = pf[i][j]
        return(res)
    
    def powerMethod(self, mat, dim):
    
        vect = np.ones(dim) #Initial guess
        nrm = np.linalg.norm(vect) #Get the norm (we won't use this one but is generally useful for building a stopping condition)
        for _ in range(self.Zsim): #Perform 50 iterations (heuristic stopping conditions)
            vect = mat.dot(vect) #Multiply the matrix and the vector
            nrm = np.linalg.norm(vect) #Normalize the result
            vect = vect/nrm
        return(vect)
    

    def loop(self, x):
    
        ind = self.discretize(x,  self.Zdim, self.Zmin, self.Zstep) #Discretize the state
        pf = self.passive_dynamics[ind[0],ind[1]] #Get the pf corresponding to the passive dynamics
        pf_1D = self.unravelPF(pf) #Unravel it
        pf_weighted = pf_1D*self.z #Calculate the actual transition pf using z and the passive dynamics
        S = np.sum(pf_weighted) #Normalize
        pf_weighted = pf_weighted/S
        ind = np.random.choice(range(self.Zdiscr[0]**2), p=pf_weighted) #Get the new (enumerated) state index using the calculated dynamics
        newState = self.stateVect[ind] #Get the new state from the state vector
        return(newState)

print("Prova del sistema KLC")

klc_controller = KLC_controller([16, 16])
x, y, time = klc_controller.update()

for x1, y1 in zip(x, y):
    print(str(x1) + ", " + str(y1))

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