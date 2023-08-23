import numpy as np
import matplotlib.pyplot as plt

def extract_data_from_dataset(file_name):
    return np.load(file_name)

planning = extract_data_from_dataset("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/klc_gaussian_results_from_planning.npy")
simulation = extract_data_from_dataset("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/klc_gaussian_results_from_simulation.npy")

# Creazione del grafico
plt.plot(planning[0], planning[1], marker='o', label='Planning', zorder=1)
plt.plot(simulation[0], simulation[1], marker='x', label='Simulation', zorder=0)

# Aggiunta del punto in primo piano
plt.scatter(16, 16, color='green', marker='o', label='Final Target', s=100, zorder=2)

# Aggiunta di titoli e label agli assi
plt.title('Performance on the target')
plt.xlabel('x position')
plt.ylabel('y position')

# Aggiunta di una legenda
plt.legend()

# Mostra il grafico
plt.show()