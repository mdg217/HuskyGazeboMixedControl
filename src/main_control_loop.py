import rospy
from mpc_controller import *
from cost_cache import *
from klc_controller_online import *

cache = CostCache()

target = [8, 8]
klc = ControllerKLCOnline(target, 0)
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
    print("prova")
    if stopping_cond == 1:
        break

print("salvataggio dei risultati nella simulazione!")
np.save("klc_vision_linear_results_from_simulation", np.array([mpc_x_history, mpc_y_history, mpc_t_history]))