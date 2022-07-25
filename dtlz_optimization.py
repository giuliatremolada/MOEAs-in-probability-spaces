import json
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_problem, get_termination, get_reference_directions
from pymoo.optimize import minimize
from custom_selection_dtlz import WSTSelection
from moead_wst import MOEADWST

from custom_callback_dtlz import MyCallback_dtlz


N_GEN = 75
N_TRIAL = 2

problem_dtlz2 = get_problem("dtlz2", n_var=10, n_obj=10)


termination = get_termination("n_gen", N_GEN)

hvs_nsga2, hvs_nsga3, hvs_moea_wst, hvs_moead, hvs_moead_wst = [], [], [], [], []

for SEED in range(1, N_TRIAL + 1):

    ref_dirs_moea = get_reference_directions("das-dennis", 10, n_partitions=2) # con obj=3 e con p=7 le reference direction sono 36, per 5 obbiettivi con p=3 le ref sono 35, per 7 obbiettivi con p=2 le ref sono 28

    ## MOEAD/WST

    algorithm_moead_wst = MOEADWST(
        ref_dirs=ref_dirs_moea,
        n_offspring=10,
        n_neighbors=20,
        prob_neighbor_mating=0.7,
        seed=SEED
    )
    c_moead_wst = MyCallback_dtlz()

    res_moead_wst = minimize(problem_dtlz2,
                    algorithm_moead_wst,
                    termination,
                    seed=SEED,
                    save_history=True,
                    callback=c_moead_wst,
                    verbose=True)

    hvs_moead_wst.append(c_moead_wst.data["HV"])

    ### MOEA/D
    algorithm_moea_d = MOEAD(ref_dirs=ref_dirs_moea,
                             n_offsprings=10,
                             n_neighbors=20,
                             prob_neighbor_mating=0.7,
                             seed=SEED
                             )

    c_moea_d = MyCallback_dtlz()
    res_moea_d = minimize(problem_dtlz2, algorithm_moea_d, termination,
                            save_history=True,
                            verbose=True,
                            callback=c_moea_d,
                            seed=SEED)
    hvs_moead.append(c_moea_d.data["HV"])

    ### MOEA/WST

    algorithm_moea_wst = NSGA2(pop_size=55,
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


    hvs_moea_wst.append(c_moea_wst.data["HV"])

    ### NSGA-II
    algorithm_nsga2 = NSGA2(pop_size=55,
                            n_offsprings=10,
                            eliminate_duplicates=True,
                            seed=SEED)

    c_nsga2 = MyCallback_dtlz()
    res_nsga2 = minimize(problem_dtlz2, algorithm_nsga2, termination, callback=c_nsga2, seed=SEED)
    hvs_nsga2.append(c_nsga2.data["HV"])

    ### NSGA-III
    ref_dirs = get_reference_directions("das-dennis", 10, n_partitions=2)

    algorithm_nsga3 = NSGA3(ref_dirs=ref_dirs,
                            pop_size=55,
                            n_offsprings=10,
                            eliminate_duplicates=True,
                            seed=SEED)

    c_nsga3 = MyCallback_dtlz()
    res_nsga3 = minimize(problem_dtlz2, algorithm_nsga3, termination, callback=c_nsga3, seed=SEED)
    hvs_nsga3.append(c_nsga3.data["HV"])


hypervolume = {"nsga2": hvs_nsga2, "nsga3": hvs_nsga3, "moea/wst": hvs_moea_wst, "moea/d": hvs_moead, "moead/wst": hvs_moead_wst}
out_file_hvs = open("Risultati_dtlz\\hypervolume.json", "w")
json.dump(hypervolume, out_file_hvs)