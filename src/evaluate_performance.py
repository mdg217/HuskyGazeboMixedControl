import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from obstacle import *
from klc_controller import *


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

mode = 0
online = "online_"
real = "real_"

print("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/last_result_for_thesis/dynamic_" + str(mode) + "/klc_"+ online + real + "planning_" + str(mode) + ".npy")

planning = extract_data_from_dataset("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/last_result_for_thesis/dynamic_" + str(mode) + "/klc_"+ online + real + "planning_" + str(mode) + ".npy")
simulation = extract_data_from_dataset("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/last_result_for_thesis/dynamic_" + str(mode) + "/klc_"+ online + real + "simulation_" + str(mode) + ".npy")
speeds = extract_data_from_dataset("/home/marco/catkin_ws/src/husky_mpc_datadriven/src/data/last_result_for_thesis/dynamic_" + str(mode) + "/inputs_for_husky_" +real +"klc_"+ online + str(mode) + ".npy")

print("Last value of position", simulation[:,-1])

print("Time to get the target is: " + str(simulation[2]))

# Determina il numero di punti desiderato per l'interpolazione
target_length = len(simulation[0])

# Interpolazione dei dati di "planning"
interpolated_waypoints = interpolate_waypoints(np.array([planning[0], planning[1]]), target_length)

print(calculate_mean_position_error(np.array([simulation[0], simulation[1]]), interpolated_waypoints))

# Creazione del grafico
fig, ax = plt.subplots(figsize=(10, 10))


klc_controller = ControllerKLC([8, 8], 0)
i = 0
for obs in klc_controller.obstacles.get_obs():
    if i == 0:
        ax.scatter(obs[0], obs[1], color='r', s=1000, label="Obstacle")
        i = 1
    else:
        ax.scatter(obs[0], obs[1], color='r', s=1000)


# Set the initial size of the axis to 10x10 units
ax.set_xlim(0, 10)  # Set the x-axis limits from 0 to 10
ax.set_ylim(0, 10)  # Set the y-axis limits from 0 to 10
ax.grid(True)
# Initialize empty line objects for the paths
"""planning_line, = ax.plot([],[] , marker='o', label='Planning', zorder=0)
simulation_line, = ax.plot([], [], marker='o', label='Simulation', zorder=1)"""
ax.plot(interpolated_waypoints[0], interpolated_waypoints[1], marker='o', label='Planning', zorder=1)
ax.plot(simulation[0], simulation[1], marker='o', label='Simulation', zorder=0)

# Initialize a point for the final target
target_point = ax.scatter(7.8, 7.8, color='green', marker='o', label='Final Target', s=300, zorder=2)

# Add titles and labels to the plot
# Add titles and labels to the plot with increased fontsize
ax.set_title('Performance on the target', fontsize=16)  # Adjust fontsize as needed
ax.set_xlabel('x position', fontsize=16)  # Adjust fontsize as needed
ax.set_ylabel('y position', fontsize=16)  # Adjust fontsize as needed

# Add a legend with increased fontsize
ax.legend(fontsize=16, loc='upper left')  # Adjust fontsize as needed
# Aggiunta di titoli e label agli assi

# Aggiunta di una legenda
# Mostra il grafico
plt.savefig('/home/marco/Desktop/klc_dinamica_passiva' + str(mode) + '/target_' + online + real + str(mode) +'.png' )

"""# Function to initialize the plot
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
animation = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, repeat=False)
# Display the animation
plt.show()"""


def calculate_mean_speeds(speeds):
    assert speeds.shape[0] == 2, "Speeds must have 2 rows"
    
    mean_speed_x = np.mean(speeds[0])
    mean_speed_y = np.mean(speeds[1])
    
    return mean_speed_x, mean_speed_y

# Calcola le medie delle velocità separatamente
mean_speed_x, mean_speed_y = calculate_mean_speeds(speeds)

# Stampa le medie delle velocità separatamente
print("Mean Speed (X): {:.2f}".format(mean_speed_x))
print("Mean Speed (Y): {:.2f}".format(mean_speed_y))

write_into_file_performance = "Final Position: \t" + str(simulation[2][-1]) + "\n"
write_into_file_performance += "Error Final Position: \t(" + str(8-simulation[0,-1]) + ", " + str(8-simulation[1,-1]) +")\n"
write_into_file_performance += "Mean Speed Value: \t" + str([mean_speed_x, mean_speed_y]) + "\n"
write_into_file_performance += "Mean Square Error on Position: \t" + str(calculate_mean_position_error(np.array([simulation[0], simulation[1]]), interpolated_waypoints)) + "\n"
write_into_file_performance += "Travel Time: \t" + str(simulation[2][-1])

percorso_file = '/home/marco/Desktop/klc_dinamica_passiva' + str(mode) + '/performance_' + online + real + str(mode) + '.txt'

# Apri il file in modalità scrittura ('w' sta per "write")
with open(percorso_file, 'w') as file:
    # Scrivi la stringa sul file
    file.write(write_into_file_performance)


# I dati delle velocità
#htime = np.arange(0, 22.1, 0.1)
htime = simulation[2]
#speed_x = speeds[0][0:np.shape(htime)[0]]  # Velocità sull'asse X
#speed_y = speeds[1][0:np.shape(htime)[0]]  # Velocità sull'asse Y
speed_x = speeds[0]
speed_y = speeds[1]


# Creazione del grafico con due subplot
plt.figure(figsize=(12, 6))  # Imposta le dimensioni della figura

# Subplot per la velocità sull'asse X
plt.subplot(2, 1, 1)  # 2 righe, 1 colonna, primo subplot
plt.plot(htime, speed_x, label='Velocità X', color='blue')
plt.xlabel('Execution time', fontsize=16)
plt.ylabel('Linear speed', fontsize=16)
plt.grid(True)

# Subplot per la velocità sull'asse Y
plt.subplot(2, 1, 2)  # 2 righe, 1 colonna, secondo subplot
plt.plot(htime, speed_y, label='Velocità Y', color='green')
plt.xlabel('Execution time', fontsize=16)
plt.ylabel('Twist speed', fontsize=16)
plt.grid(True)

# Regolare gli spazi tra i subplot per evitare sovrapposizioni
plt.tight_layout()

# Mostra il grafico
plt.savefig('/home/marco/Desktop/klc_dinamica_passiva' + str(mode) + '/speed_' + online + real + str(mode) +'.png' )