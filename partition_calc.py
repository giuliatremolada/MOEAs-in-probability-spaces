import json

import pickle

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_termination, get_sampling, get_crossover, get_mutation,get_reference_directions
from pymoo.optimize import minimize

from objective_functions import coverage_pymoo, novelty_pymoo, accuracy_pymoo
from multiobjective_problem import OptimizationProblem
from custom_callback import MyCallback

with open('rating_matrix.pickle', 'rb') as f:
    rating_matrix = pickle.load(f)

problem_nsga = OptimizationProblem(rating_matrix = rating_matrix,
                              L = 15,
                              f1=coverage_pymoo, f2=novelty_pymoo, f3=accuracy_pymoo)
hvs_nsga3_all = []
for seed in range(1, 4):  # tre trial

    print(f"trial {seed}")
    hvs_nsga3 = []
    for p in range(1, 9): # p=8 è l'ultimo valore che da ref_dirs<40
        # create the reference directions to be used for the optimization
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=p)

        # create the algorithm object
        algorithm = NSGA3(ref_dirs=ref_dirs,
                          pop_size=40,
                          #n_offsprings=10,
                          sampling=get_sampling("int_random"),
                          crossover=get_crossover("int_sbx", prob=1, eta=3.0),
                          mutation=get_mutation("perm_inv"),
                          eliminate_duplicates=True,
                          seed = seed)

        termination = get_termination("n_gen", 250)
        # execute the optimization
        c = MyCallback()
        res = minimize(problem_nsga,
                       algorithm,
                       seed=seed,
                       callback=c,
                       termination=termination)

        hvs_nsga3.append(c.data["HV"])

    hvs_nsga3_all.append(hvs_nsga3)


hypervolume_partizioni = {"nsga3": hvs_nsga3_all}
out_file_partizioni = open("Risultati_pymoo\\partizioni.json", "w")
json.dump(hypervolume_partizioni, out_file_partizioni)
