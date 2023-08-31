import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def extract_data_from_dataset(file_name):
    return np.load(file_name)

def calculate_mean_position_error(true_path, estimated_path):
    assert true_path.shape == estimated_path.shape, "Paths must have the same shape"
    
    num_points = true_path.shape[1]
    distances = np.sqrt(np.sum((true_path - estimated_path)**2, axis=0))
    mean_error = np.mean(distances)
    
    return mean_error

def interpolate_waypoints(waypoints, new_length):
    # Calcola la distanza cumulativa tra i punti originali
    distances = np.sqrt(np.sum(np.diff(waypoints, axis=1)**2, axis=0))
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

    # Parametrizzazione arclength
    t = np.linspace(0, cumulative_distances[-1], new_length)

    # Interpolazione
    interpolator = interp1d(cumulative_distances, waypoints, kind='linear', axis=1)
    interpolated_waypoints = interpolator(t)

    return interpolated_waypoints

planning = extract_data_from_dataset("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/simulation_final/dynamic_vision_linear_results_from_planning.npy")
simulation = extract_data_from_dataset("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/simulation_final/dynamic_vision_linear_results_from_simulation.npy")
print("Time to get the target is: " + str(simulation[2][-1]))

# Determina il numero di punti desiderato per l'interpolazione
target_length = len(simulation[0])

# Interpolazione dei dati di "planning"
interpolated_waypoints = interpolate_waypoints(np.array([planning[0], planning[1]]), target_length)
for x1, x2 in zip(simulation[0], interpolated_waypoints[0]):
    print(str(x1) + "\t" + str(x2))
print("-----------------------------------------------")
for x1, x2 in zip(simulation[1], interpolated_waypoints[1]):
    print(str(x1) + "\t" + str(x2))

print(calculate_mean_position_error(np.array([simulation[0], simulation[1]]), interpolated_waypoints))

# Creazione del grafico
plt.plot(interpolated_waypoints[0], interpolated_waypoints[1], marker='o', label='Planning', zorder=0)
plt.plot(simulation[0], simulation[1], marker='o', label='Simulation', zorder=1)

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