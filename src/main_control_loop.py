import rospy
from mpc_controller import *
from cost_cache import *
from klc_controller_online import *

cache = CostCache()

target = [8, 8]
klc = ControllerKLCOnline(target, 0)
x, y, time, sx, sy = klc.update()
#np.save("path_to_do_in_mpc", np.array([x,y]))
#np.save("mean_stdv_plot_data", np.array([x, sx, y, sy]))
klc.export_metrics(x, y, time)
cache.set_next_target(x, y)

mpc = ControllerMPC([0,0,0])

mpc_x_history = np.array([])
mpc_y_history = np.array([])
mpc_t_history = np.array([])
mpc_t = 0

while not rospy.is_shutdown():   
    mpc_x, mpc_y, stopping_cond = mpc.update(target)
    mpc_x_history = np.append(mpc_x_history, mpc_x)
    mpc_y_history = np.append(mpc_y_history, mpc_y)
    mpc_t += 0.1
    mpc_t_history = np.append(mpc_t_history, mpc_t)
    if stopping_cond == 1:
        break

print(x[-1])
print(y[-1])

# Crea una griglia di subplot con 1 riga e 2 colonne
fig, axs = plt.subplots(1, 1, figsize=(10, 5))

# Plot del primo subplot
axs.plot(mpc_x_history, mpc_y_history, marker='o', linestyle='-', color='r')
axs.set_xlabel('X Position')
axs.set_ylabel('Y Position')
axs.set_title('Primo Plot')
for obs in klc.obstacles.get_obs():
    axs.scatter(obs[0], obs[1], color='r', s=1000)

# Regola la spaziatura tra i subplot
plt.tight_layout()

# Mostra i subplot
plt.show()

print("salvataggio dei risultati nella simulazione!")
np.save("klc_results_from_real_simulation", np.array([mpc_x_history, mpc_y_history, mpc_t_history]))

u1, u2 = mpc.get_inputs()

np.save("inputs_for_husky", np.array([u1, u2]))