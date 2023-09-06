import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation

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

planning = extract_data_from_dataset("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/klc_vision_linear_results_from_planning.npy")
simulation = extract_data_from_dataset("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/klc_results_from_real_simulation.npy")
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
fig, ax = plt.subplots(figsize=(10, 10))

# Set the initial size of the axis to 10x10 units
ax.set_xlim(0, 10)  # Set the x-axis limits from 0 to 10
ax.set_ylim(0, 10)  # Set the y-axis limits from 0 to 10
# Initialize empty line objects for the paths
planning_line, = ax.plot([],[] , marker='o', label='Planning', zorder=0)
simulation_line, = ax.plot([], [], marker='o', label='Simulation', zorder=1)

# Initialize a point for the final target
target_point = ax.scatter(7.9, 7.9, color='green', marker='o', label='Final Target', s=200, zorder=2)

# Add titles and labels to the plot
ax.set_title('Performance on the target')
ax.set_xlabel('x position')
ax.set_ylabel('y position')

# Add a legend
ax.legend()

# Function to initialize the plot
def init():
    planning_line.set_data([], [])
    simulation_line.set_data([], [])
    return planning_line, simulation_line

# Function to update the plot for each frame of the animation
def update(frame):
    x_planning = interpolated_waypoints[0, :frame+1]
    y_planning = interpolated_waypoints[1, :frame+1]
    x_simulation = simulation[0, :frame+1]
    y_simulation = simulation[1, :frame+1]

    planning_line.set_data(x_planning, y_planning)
    simulation_line.set_data(x_simulation, y_simulation)
    
    return planning_line, simulation_line

# Create an animation object
num_frames = target_length  # Number of frames (one for each point)
animation = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)

# Display the animation
plt.show()

