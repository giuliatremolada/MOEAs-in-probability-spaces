import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.factory import get_performance_indicator
from pymoo.problems.many import DTLZ2
from custom_callback_dtlz import MyCallback_dtlz
from moead_wst import MOEADWST

N_TRIALS = 5
hvs_moead, hvs_moead_wst = [], []

problem = DTLZ2(n_var=20, n_obj=3)

for trial in range(1, N_TRIALS+1):
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=6)
    c_moea_d_wst = MyCallback_dtlz()

    algorithm = MOEADWST(
        ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7

    )

    res1 = minimize(problem,
                algorithm,
                ('n_gen', 100),
                seed=trial,
                save_history=True,
                callback=c_moea_d_wst,
                verbose=True)
    hvs_moead_wst.append(c_moea_d_wst.data["HV"])

    c_moead = MyCallback_dtlz()

    algorithm = MOEAD(
        ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,

    )

    res2 = minimize(problem,
                algorithm,
                ('n_gen', 100),
                save_history=True,
                seed=trial,
                callback=c_moead,
                verbose=True)
    hvs_moead.append(c_moead.data["HV"])


hvs_mean_moead = np.asarray(hvs_moead).mean(axis=0)
hvs_mean_moead_wst = np.asarray(hvs_moead_wst).mean(axis=0)
hvs_std_moead = np.asarray(hvs_moead).std(axis=0)
hvs_std_moead_wst = np.asarray(hvs_moead_wst).std(axis=0)

means = [hvs_mean_moead_wst, hvs_mean_moead]
stds = [hvs_std_moead_wst, hvs_std_moead]

colors = ["orangered", "royalblue"]
line_labels = ["MOEA/D/WST", "MOEA/D"]
iters =  np.arange(0, 100)

plt.rc('font', size=12)  # controls default text sizes
plt.rc('axes', titlesize=14)  # fontsize of the axes title
plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.rc('legend', fontsize=12)  # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title

for idx, value in enumerate(line_labels):
    # Linea corrispondente alla media
    plt.plot(iters, means[idx], color=colors[idx], label=line_labels[idx])
    # STD
    lower = means[idx] - stds[idx]
    upper = means[idx] + stds[idx]
    lower[lower < 0] = 0
    # Deviazione standard attorno alla media
    plt.fill_between(iters, lower, upper, color=colors[idx], alpha=0.2)

plt.xlabel("Number of observations")
plt.ylabel("Hypervolume")
plt.legend(loc="lower right")
plt.show()
"""
hvcalc = get_performance_indicator("hv", ref_point=np.full(10, 1.5))
hvs_moead_wst, hvs_moead = [], []
for p1, p2 in zip(res1.history, res2.history):
    f1 = p1.pop.get("F")
    f2 = p2.pop.get("F")
    hvs_moead_wst.append(hvcalc.do(f1))
    hvs_moead.append(hvcalc.do(f2))


plt.plot(np.arange(0, len(hvs_moead_wst)), hvs_moead_wst, color="orangered", linewidth=3, label="MOEA/D-WST")
plt.plot(np.arange(0, len(hvs_moead)), hvs_moead, color="royalblue", linewidth=3, label="MOEA/D")
plt.legend()
plt.show()
"""

