import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log
import os

class KLC_controller:

    def __init__(self, goal):

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

        # Creazione del vettore 4D inizializzato con zeri
        self.passive_dynamics = np.zeros((self.Zdiscr[0] , self.Zdiscr[0], self.Zdiscr[0], self.Zdiscr[0]))

        # Popolamento delle transizioni per gli stati adiacenti
        for row in range(self.Zdiscr[0]):
            for col in range(self.Zdiscr[0]):
                # Stato attuale
                current_state = (row, col)

                # Transizioni possibili: su, giù, sinistra, destra
                possible_transitions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

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


        stateVect = np.zeros((self.Zdiscr[0]**2, 2))

        for i in range(self.Zdiscr[0]):
            #Enumerate the states from 1 to 36^2. Here we explicitly build the enumeration to later build the Q matrix and for convenience
            for j in range(self.Zdiscr[0]):
                # Compute the angle and speed values for the current state
                x = (i)*self.Zstep[0]
                y = (j)*self.Zstep[0]
                # Calculate the index of the current state in the state vector
                ind = i*self.Zdiscr[0] + j
                # Assign the angle and speed values to the state vector
                stateVect[ind] = [x, y] 

        diagMinusQ = np.zeros((self.Zdiscr[0]**2, self.Zdiscr[0]**2))
        #Q matrix
        for i in range(self.Zdiscr[0]**2):
            #Build the diagonal matrix with the exponential of the opposite of the cost
            diagMinusQ[i,i] = np.exp(-self.cost(stateVect[i]))

        Prob = np.zeros((self.Zdiscr[0]**2, self.Zdiscr[0]**2))

        for i in range(self.Zdiscr[0]):
            for j in range(self.Zdiscr[0]):
                pf = self.passive_dynamics[i,j]
                ind1 = i*self.Zdiscr[0] + j
                Prob[ind1] = self.unravelPF(pf)

        self.z = self.powerMethod(diagMinusQ@Prob, self.Zdiscr[0]**2) 

    
    def update(self):
        fullH = np.zeros((self.Zsim,300))
        fullHv = np.zeros((self.Zsim,300))
        nSteps = 300

        #Task:  obtain simulations for different initial conditions (say, 5 different initial conditions). For each of these, run 50 simulations.

        for j in range(self.Zsim): #Perform 50 simulations
            hist = [[0,0]]*nSteps
            state = [0, 0] #Initialize the pendulum <------------
            for i in range(nSteps): #For each step
                hist[i]=state #Log the state
                state = self.loop(state) #Sample the new state
            fullH[j] = [x[0] for x in hist]
            fullHv[j] = [x[1] for x in hist]

        x = [0]*300 #Get the means and stds for plotting
        stds = [0]*300
        for i in range(300):
            x[i] = np.mean(fullH[:,i])
            stds[i] = np.std(fullH[:,i])

        from pylab import rcParams
        plt.rcParams.update({'font.size': 18})

        #PLOT X -> Angle
        x = np.array([x/10 for x in range(300)])
        y = np.array(x)
        ci = np.array(stds)

        fig, ax = plt.subplots()
        plt.xlim([0, 3])
        ax.plot(x,y)
        plt.xlabel("Time")
        plt.ylabel("x")
        plt.title("Position on x")
        ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)

        y = [0]*300 #Get the means and stds for plotting
        stdsv = [0]*300
        for i in range(300):
            y[i] = np.mean(fullHv[:,i])
            stdsv[i] = np.std(fullHv[:,i])

        #PLOT Vx -> Speed
        xv = np.array([x/10 for x in range(300)])
        yv = np.array(y)
        civ = np.array(stdsv)

        figv, axv = plt.subplots()
        plt.xlim([0, 3])
        plt.xlabel("Time")
        plt.ylabel("y")
        plt.title("Position on y")
        axv.plot(xv,yv)
        axv.fill_between(xv, (yv-civ), (yv+civ), color='b', alpha=.1)

        return [x, y]

    
    # Utility methods for init and update methods

    def discretize(self, Z, Zdim, Zmin, Zstep):
        res = [0]*Zdim #n-dimensional index
        for i in range(Zdim): #For each dimension
            elt = Z[i] #Extract the i-th element
            ind = int((elt - Zmin[i])//Zstep[i]) #Discretize
            res[i] = ind
        return(tuple(res)) #Return as tuple for array indexing
    
    
    def cost(self, state):
        
        #Aggiungere il codice per tutti gli ostacoli di ros
        xterm = ((state[0]-4)/0.7)*((state[0]-4)/0.7)
        yterm = ((state[1]-4)/0.7)*((state[1]-4)/0.7)

        return (state[0]-self.xd)*(state[0]-self.xd) + (state[1]-self.yd)*(state[1]-self.yd) + 100000*np.exp(-0.5* (xterm + yterm))
    
    
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

klc_controller = KLC_controller([10, 10])
x,y = klc_controller.update()