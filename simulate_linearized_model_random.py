import numpy as np
import matplotlib.pyplot as plt

class simulate_model:

    def __init__(self):

        self.num_samples = 1000
        self.x = 0 + (16 - 0) * np.random.rand(self.num_samples)
        self.y = 0 + (16 - 0) * np.random.rand(self.num_samples)
        self.theta = -np.pi + (np.pi + np.pi) * np.random.rand(self.num_samples)
        self.v = np.random.rand(self.num_samples)
        self.w = np.random.rand(self.num_samples)
        self.T = 0.1
    
    def simulate(self):

        # Inizializza output vuoti
        simulated_output1 = []
        simulated_output2 = []
        simulated_output3 = []

        # Simula il sistema discreto
        for i in range(self.num_samples):
            xk = self.x[i]
            yk = self.y[i] + self.T*self.v[i]
            thetak = self.theta[i] + self.T*self.w[i]

            simulated_output1.append(xk)
            simulated_output2.append(yk)
            simulated_output3.append(thetak)

        simulated_output1 = np.array(simulated_output1)
        simulated_output2 = np.array(simulated_output2)
        simulated_output3 = np.array(simulated_output2)

        # Creazione di grafici per visualizzare i risultati
        plt.scatter(simulated_output1, simulated_output2, label='Simulated Data')
        plt.xlabel('Output 1')
        plt.ylabel('Output 2')
        plt.title('Simulated Discrete Linear System')
        plt.legend()
        plt.grid()
        plt.show()


sim = simulate_model()
sim.simulate()