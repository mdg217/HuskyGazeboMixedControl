import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log
from obstacle import *
import rospy
import time
from Plants.uniform_plant import *
from Plants.linearized_plant import *
from Plants.trajectory_based_plant import *

"""
ControllerKLC: A class implementing a Kinematic Linearization Controller (KLC) for robot motion planning.

This class defines methods to initialize the controller and update its behavior based on robot states.
"""
class ControllerKLC:

    """
    Update the controller's behavior based on the current state.
    
    :param self: The instance of the class.
    :return: Lists containing mean x position, mean y position, and time.
    """
    def __init__(self, goal, mode):

        self.cache = CostCache()

        self.mode = mode
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
        self.zstep = [0.25, 0.25]

        #Amount of discrete bins
        self.zdiscr = [40, 40]

        #Number of iterations for the simulations
        self.zsim = 15

        #Duration of the simulation
        self.duration = 30

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

        self.diagMinusQ = np.zeros((self.zdiscr[0]**2, self.zdiscr[0]**2)) # q

        

        for i in range(self.zdiscr[0]**2):
            #Build the diagonal matrix with the exponential of the opposite of the cost
            self.diagMinusQ[i,i] = np.exp(-self.cost(self.stateVect[i]))

        heatmap = np.zeros((self.zdiscr[0], self.zdiscr[1]))
        
        for i in range(self.zdiscr[0]):
            for j in range(self.zdiscr[1]):
                #Build the diagonal matrix with the exponential of the opposite of the cost
                state = self.stateVect[i*self.zdiscr[0] + j]
                print(self.cost(state))
                heatmap[j, i] = self.cost(state)
        

        # Crea il plot della heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label='State cost')
        
        # Imposta la dimensione dei caratteri per il titolo, l'etichetta x e l'etichetta y
        plt.title('State cost heatmap', fontsize=20)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.show()


        self.Prob = np.zeros((self.zdiscr[0]**2, self.zdiscr[0]**2))

        for i in range(self.zdiscr[0]):
            for j in range(self.zdiscr[0]):
                pf = self.passive_dynamics[i,j]
                ind1 = i*self.zdiscr[0] + j
                self.Prob[ind1] = self.unravelPF(pf)

        self.z = np.array((self.zdiscr[0]**2))

        #Compute the execution time
        start_time = time.time()
        zmp = self.powerMethod(self.diagMinusQ@self.Prob, self.zdiscr[0]**2) 
        #print(zmp)
        end_time = time.time()
        elapsed_time = end_time - start_time

        zmpheat = np.zeros((self.zdiscr[0], self.zdiscr[1]))
        
        for i in range(self.zdiscr[0]):
            for j in range(self.zdiscr[1]):
                #Build the diagonal matrix with the exponential of the opposite of the cost
                zmpheat[j, i] = zmp[i*self.zdiscr[0] + j]
        

        # Crea il plot della heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(zmpheat, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label='State cost')
        
        # Imposta la dimensione dei caratteri per il titolo, l'etichetta x e l'etichetta y
        plt.title('State cost heatmap', fontsize=20)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.show()



        print(f"Tempo di esecuzione: {elapsed_time} secondi")
        
        #Compute the execution time
        start_time = time.time()
        self.dynamic_programming() 
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Tempo di esecuzione: {elapsed_time} secondi")

        zdpheat = np.zeros((self.zdiscr[0], self.zdiscr[1]))
        
        for i in range(self.zdiscr[0]):
            for j in range(self.zdiscr[1]):
                #Build the diagonal matrix with the exponential of the opposite of the cost
                zdpheat[j, i] = self.z[i*self.zdiscr[0] + j]
        

        # Crea il plot della heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(zdpheat, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(label='State cost')
        
        # Imposta la dimensione dei caratteri per il titolo, l'etichetta x e l'etichetta y
        plt.title('State cost heatmap', fontsize=20)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.show()

    

    """
    set initial guess u=p (probably passive dynamics)
    set initial guess v
    repeat while not converged:
    for each state s:
        p(s'|s) (transition probabilities from s)
        u(s'|s) (transition probs from s)
        calculate the inside of brackets in (1) = l(s,u) + E_u(s'|s)[v(s')]
        find unext(s'|s) = min_u (what is above)
    """
    def dynamic_programming(self, max_iterations=100):
        u = self.passive_dynamics
        p = self.passive_dynamics
        v = np.zeros((self.zdiscr[0], self.zdiscr[1]))

        for _ in range(max_iterations):
            #new_V = np.copy(self.V)

            for x in range(self.zdiscr[0]):
                last_v = v
                for y in range(self.zdiscr[1]):
                    #compute the l(x,u) term
                    current_state = x*self.zdiscr[0] + y

                    q = self.cost(self.stateVect[current_state])

                    logvalue = np.log(u[x, y] / (p[x, y]))
                    logvalue = np.nan_to_num(np.log(u[x, y] / (p[x, y])), nan=0)

                    dkl = np.sum(u[x,y]*logvalue)

                    l = q + dkl
                    
                    #compute the v term
                    v[x,y] = l + np.sum(u[x,y]*v)

            for x in range(self.zdiscr[0]):
                for y in range(self.zdiscr[1]):
                    u[x, y] = (p[x, y]*np.exp(-v))/(np.sum(p[x, y]*np.exp(-v)))

        self.z = np.exp(-self.unravelPF(v))

        #print(self.z)


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
            hist = [[0,0]]*nSteps
            state = [0, 0] #Initialize the pendulum <------------
            for i in range(nSteps): #For each step
                #print("Step #" + str(i))
                hist[i]=state #Log the state
                state = self.loop(state) #Sample the new state
            fullH[j] = [x[0] for x in hist]
            fullHv[j] = [x[1] for x in hist]

        meanx = [0]*self.duration #Get the means and stds for plotting
        stdsx = [0]*self.duration
        for i in range(self.duration):
            meanx[i] = np.mean(fullH[:,i])
            stdsx[i] = np.std(fullH[:,i])

        meany = [0]*self.duration #Get the means and stds for plotting
        stdsy = [0]*self.duration
        for i in range(self.duration):
            meany[i] = np.mean(fullHv[:,i])
            stdsy[i] = np.std(fullHv[:,i])

        time = np.array([time for time in range(self.duration)])

        meanx = [0]*self.duration #Get the means and stds for plotting
        stdsx = [0]*self.duration
        for i in range(self.duration):
            meanx[i] = np.mean(fullH[:,i])
            stdsx[i] = np.std(fullH[:,i])

        print("result position:")
        print(meanx[-1])
        print(meany[-1])

        return [meanx, meany, time, stdsx, stdsy]

    
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
        k = 30
        sx = 0.7
        sy = 0.7
        q1 = 0.2
        q2 = 0.2
        obsTerm = 0

        for obs in self.obstacles.get_obs():
            xterm = ((state[0] - obs[0]) / sx) ** 2
            yterm = ((state[1] - obs[1]) / sy) ** 2
            obsTerm += k * np.exp(-0.5 * (xterm + yterm))

        # Include the regularization term in the overall cost calculation 
        return q1*(state[0] - self.xd) ** 2 + q2*(state[1] - self.yd) ** 2 + obsTerm
    
    
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
            vect = mat.dot(vect)
            nrm = np.linalg.norm(vect)
            vect = vect / nrm
            print(nrm)
            
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
        np.save("klc_planning_"+ str(self.mode), np.array([x, y, time]))


print("Prova del sistema KLC")

klc_controller = ControllerKLC([8, 8], 0)
print("Prova Dynamic Programming")
print("FINE DYNAMIC")

x, y, htime, sx, sy = klc_controller.update()
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