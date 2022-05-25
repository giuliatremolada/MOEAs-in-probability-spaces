import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.factory import get_performance_indicator
from pymoo.problems.many import DTLZ2
from custom_callback_dtlz import MyCallback_dtlz
from moead_wst import MOEADWST

#hvs_moead, hvs_moead_wst = [], []

problem = DTLZ2(n_var=20, n_obj=10)

ref_dirs = get_reference_directions("das-dennis", 10, n_partitions=3)
#c_moea_d_wst = MyCallback_dtlz()

algorithm = MOEADWST(
    ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7

)

res1 = minimize(problem,
                algorithm,
                ('n_gen', 100),
                seed=1,
                save_history=True,
                # callback=c_moea_d_wst,
                verbose=True)
#hvs_moead_wst.append(c_moea_d_wst.data["HV"])

#c_moead = MyCallback_dtlz()

algorithm = MOEAD(
    ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7,

)

res2 = minimize(problem,
                algorithm,
                ('n_gen', 100),
                save_history=True,
                seed=1,
                #callback=c_moead,
                verbose=True)
#hvs_moead.append(c_moead.data["HV"])

# TODO How to add Wasserstein:
#  - Use in the selection of neighbours (moead.py, line 79)

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
