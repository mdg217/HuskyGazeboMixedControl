import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, dlti, dlsim

class simulate_model:

    def __init__(self):
        
        self.T = np.array([0.1])

        self.A = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        
        self.B = self.T * np.array([[0.5, 0], [0.5, 0], [0, 1]])

        
        self.C = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        
        self.D = np.array([[0, 0], [0, 0], [0, 0]])

        self.system = dlti(self.A, self.B, self.C, self.D)
    
    def simulate(self):

        # Definisci il vettore di ingresso
        time_steps = 10000
        u = np.random.randn(time_steps, self.B.shape[1])

        x0 = np.maximum(np.random.randn(3), 0.0)
        print(x0)

        # Simula il sistema dinamico
        t, y, x = dlsim(self.system, u, x0=x0)

        # t: vettore dei tempi
        # y: vettore delle uscite
        # x: matrice dei vettori di stato

        x = np.maximum(x, 0.0)
        y = np.maximum(y, 0.0)

        x1 = y[:, 0]
        x2 = y[:, 1]
        x3 = y[:, 2]

        # Fai qualcosa con i risultati, ad esempio traccia i grafici delle uscite
        import matplotlib.pyplot as plt

        plt.plot(x1, x2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Discrete State-Space System Simulation')
        plt.grid(True)
        plt.show()


sim = simulate_model()
sim.simulate()