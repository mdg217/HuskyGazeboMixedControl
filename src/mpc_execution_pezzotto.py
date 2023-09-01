import rospy
from mpc_controller import *
from cost_cache import *
from klc_controller_online import *

cache = CostCache()

rospy.init_node('husky', anonymous=True)

target = [8, 8]

x, y = np.load("/home/marco/catkin_ws/src/husky_mpc_datadriven/path_to_do_in_mpc.npy")
print(x)
print(y)

cache.set_next_target(x, y)

print("prova mpc")
mpc = ControllerMPC([0,0,0])

mpc_x_history = np.array([])
mpc_y_history = np.array([])
mpc_t_history = np.array([])
mpc_t = 0

print("prova")
while not rospy.is_shutdown():   
    print("Sto nel while")
    mpc_x, mpc_y, stopping_cond = mpc.update(target)
    mpc_x_history = np.append(mpc_x_history, mpc_x)
    mpc_y_history = np.append(mpc_y_history, mpc_y)
    mpc_t += 0.1
    mpc_t_history = np.append(mpc_t_history, mpc_t)
    if stopping_cond == 1:
        break

print(x[-1])
print(y[-1])

print("salvataggio dei risultati nella simulazione!")
np.save("klc_results_from_real_simulation", np.array([mpc_x_history, mpc_y_history, mpc_t_history]))