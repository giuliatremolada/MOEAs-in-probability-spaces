import json
import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_problem, get_termination, get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.interface import sample
from custom_selection_dtlz import WSTSelection

from custom_callback_dtlz import MyCallback_dtlz


N_GEN = 250
N_TRIAL = 5

problem_dtlz2 = get_problem("dtlz2", n_var=15, n_obj=7)


termination = get_termination("n_gen", N_GEN)

hvs_nsga2, hvs_nsga3, hvs_moea, hvs_moead = [], [], [], []

for SEED in range(1, N_TRIAL + 1):

    ### MOEA/D
    ref_dirs_moea_d = get_reference_directions("das-dennis", 7, n_partitions=2) # con obj=3 e con p=7 le reference direction sono 36, per 5 obbiettivi con p=3 le ref sono 35, per 7 obbiettivi con p=2 le ref sono 28
    algorithm_moea_d = MOEAD(ref_dirs=ref_dirs_moea_d,
                             n_offsprings=10,
                             seed=SEED)

    c_moea_d = MyCallback_dtlz()
    res_moea_d = minimize(problem_dtlz2, algorithm_moea_d, termination,
                            save_history=True,
                            verbose=True,
                            callback=c_moea_d,
                            seed=SEED)
    hvs_moead.append(c_moea_d.data["HV"])
    ### MOEA/WST

    algorithm_moea_wst = NSGA2(pop_size=40,
                           n_offsprings=10,
                           selection=WSTSelection(10),
                           eliminate_duplicates=True,
                           seed=SEED)

    c_moea_wst = MyCallback_dtlz()
    res_moea_wst = minimize(problem_dtlz2, algorithm_moea_wst, termination,
                        save_history=True,
                        verbose=True,
                        callback=c_moea_wst,
                        seed=SEED)


    hvs_moea.append(c_moea_wst.data["HV"])

    ### NSGA-II
    algorithm_nsga2 = NSGA2(pop_size=40,
                            n_offsprings=10,
                            eliminate_duplicates=True,
                            seed=SEED)

    c_nsga2 = MyCallback_dtlz()
    res_nsga2 = minimize(problem_dtlz2, algorithm_nsga2, termination, callback=c_nsga2, seed=SEED)
    hvs_nsga2.append(c_nsga2.data["HV"])

    ### NSGA-III
    ref_dirs = get_reference_directions("das-dennis", 7, n_partitions=2)

    algorithm_nsga3 = NSGA3(ref_dirs=ref_dirs,
                            pop_size=40,
                            n_offsprings=10,
                            eliminate_duplicates=True,
                            seed=SEED)

    c_nsga3 = MyCallback_dtlz()
    res_nsga3 = minimize(problem_dtlz2, algorithm_nsga3, termination, callback=c_nsga3, seed=SEED)
    hvs_nsga3.append(c_nsga3.data["HV"])


hypervolume = {"nsga2": hvs_nsga2, "nsga3": hvs_nsga3, "moea/wst": hvs_moea, "moea/d": hvs_moead}
out_file_hvs = open("Risultati_dtlz\\hypervolume.json", "w")
json.dump(hypervolume, out_file_hvs)