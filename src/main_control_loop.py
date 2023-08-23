import rospy
from mpc_controller import *
from klc_controller import *
from cost_cache import *

cache = CostCache()

target = [16, 16]
klc = ControllerKLC(target, 1)
x, y, time = klc.update()
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

print("salvataggio dei risultati nella simulazione!")
np.save("klc_gaussian_results_from_simulation", np.array([mpc_x_history, mpc_y_history, mpc_t_history]))