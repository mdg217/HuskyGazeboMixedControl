import numpy as np
from scipy.stats import multivariate_normal

class gaussian_linearized_plant():

    def get_plant(self, dim):

        self.sysData = np.load('/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/2TypeSimulation.npy')

        #Dimensions of the variables
        self.Zdim = dim

        #Minimum values
        self.Zmin = [0, 0] 

        #Discretization steps
        self.Zstep = [0.5, 0.5]

        #Amount of discrete bins
        self.Zdiscr = [37, 37]

        (full, Y) = self.getJointPMFs() #Get the joint pmfs with the parameters
        cond = self.getConditional(full, Y) #Get the conditional pmf

        #print(cond)

        return cond

    def discretize(self, Z):
        res = [0]*self.Zdim #n-dimensional index
        for i in range(self.Zdim): #For each dimension
            elt = Z[i] #Extract the i-th element
            ind = int((elt - self.Zmin[i])//self.Zstep[i]) #Discretize
            res[i] = ind
        return(tuple(res)) #Return as tuple for array indexing
    

    def getJointPMFs(self):

        fullJoint = np.zeros(self.Zdiscr*2) #p(Z,Y)
        Yjoint = np.zeros(self.Zdiscr) #p(Y)
        for Zhist in self.sysData: #For each trajectory in the dataset
            for i in range(len(Zhist)-1): #For each data point in the trajectory

                Z = Zhist[i+1] #Extract the realization of Z and Y
                Y = Zhist[i]

                Zind = self.discretize(Z) #Find the indexes
                Yind = self.discretize(Y)

                fullInd = Yind + Zind #Get the index of the joint variable Z,Y

                fullJoint[fullInd] = fullJoint[fullInd] + 1 #Update the values
                Yjoint[Yind] = Yjoint[Yind] + 1
        fullJoint = fullJoint/np.sum(fullJoint) #Normalizing
        Yjoint = Yjoint/np.sum(Yjoint)

        return(fullJoint, Yjoint)
    

    def getConditional(self, fullJoint, Yjoint):

        fullDiscr = 2*self.Zdiscr
        conditional = np.zeros(fullDiscr) #Initialize the pmf
        for (index, x) in np.ndenumerate(fullJoint): #For each index and each value in p(Z,Y) (we use this as it's robust w.r.t. the dimension)
            Yind = index[:self.Zdim] #Extract the index for Y
            if Yjoint[Yind] == 0: #Protect from dividing by zero
                conditional[index] = 0
            else:
                conditional[index] = fullJoint[index]/Yjoint[Yind] #Division
        return(conditional)
