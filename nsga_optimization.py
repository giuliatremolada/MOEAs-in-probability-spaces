import json
import pickle

import warnings
from pymoo.algorithms.moo.moead import MOEAD

warnings.filterwarnings("ignore")
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3

from pymoo.factory import get_termination, get_sampling, get_crossover, get_mutation, get_reference_directions
from pymoo.optimize import minimize


from objective_functions import coverage_pymoo, novelty_pymoo, accuracy_pymoo
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
from multiobjective_problem import OptimizationProblem
from custom_callback import MyCallback
from custom_selection import CustomSelection


N_GEN = 250
N_TRIAL = 5

### Read data

with open('rating_matrix.pickle', 'rb') as f:
    rating_matrix = pickle.load(f)


### EXPERIMENTS

### Problem

problem = OptimizationProblem(rating_matrix=rating_matrix,
                              L=15,
                              f1=coverage_pymoo, f2=novelty_pymoo, f3=accuracy_pymoo)

termination = get_termination("n_gen", N_GEN)

hvs_nsga3, hvs_moea, hvs_nsga2, hvs_moead = [], [], [], []

for SEED in range(1, N_TRIAL + 1):
    print(f"trial {SEED}")

    ### MOEAD

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=6)
    ref_dirs_moead = get_reference_directions("das-dennis", 3, n_partitions=7)

    algorithm_moead = MOEAD(ref_dirs=ref_dirs_moead, #la popolazione iniziale viene impostata pari al numero di reference direction (36)
                           n_offsprings=10,
                           sampling=get_sampling("int_random"),
                           crossover=get_crossover("int_sbx", prob=1, eta=3.0),
                           mutation=get_mutation("perm_inv"),
                           seed=SEED)
    c_moead = MyCallback()
    res_moead = minimize(problem, algorithm_moead, termination,
                        save_history=True,
                        verbose=True,
                        callback=c_moead,
                        seed=SEED)

    hvs_moead.append(c_moead.data["HV"])
    ### MOEA/WST

    algorithm_moea = NSGA2(pop_size=40,
                      n_offsprings=10,
                      sampling=get_sampling("int_random"),
                      crossover=get_crossover("int_sbx", prob=1, eta=3.0),
                      mutation=get_mutation("perm_inv"),
                      selection=CustomSelection(),
                      eliminate_duplicates=True,
                      seed=SEED)
    c_moea = MyCallback()
    res_moea = minimize(problem, algorithm_moea, termination,
                        save_history=True,
                        verbose=True,
                        callback=c_moea,
                        seed=SEED)

    hvs_moea.append(c_moea.data["HV"])

    ### NSGA-II

    algorithm_nsga2 = NSGA2(pop_size=40,
                           n_offsprings=10,
                           sampling=get_sampling("int_random"),
                           crossover=get_crossover("int_sbx", prob=1, eta=3.0),
                           mutation=get_mutation("perm_inv"),
                           eliminate_duplicates=True,
                           seed=SEED)
    c_nsga2 = MyCallback()
    res_nsga2 = minimize(problem, algorithm_nsga2, termination,
                        save_history=True,
                        verbose=True,
                        callback=c_nsga2,
                        seed=SEED)

    hvs_nsga2.append(c_nsga2.data["HV"])

    ### NSGA-III

    #ref_dirs = get_reference_directions("energy", 3, 28, seed=SEED)

    algorithm_nsga3 = NSGA3(ref_dirs = ref_dirs,
                      pop_size=40,
                      n_offsprings=10,
                      sampling=get_sampling("int_random"),
                      crossover=get_crossover("int_sbx", prob=1, eta=3.0),
                      mutation=get_mutation("perm_inv"),
                      eliminate_duplicates=True,
                      seed=SEED)

    c_nsga3 = MyCallback()
    res_nsga3 = minimize(problem, algorithm_nsga3, termination,
                        save_history=True,
                        verbose=True,
                        callback=c_nsga3,
                        seed=SEED)

    hvs_nsga3.append(c_nsga3.data["HV"])


hypervolume = {"nsga2": hvs_nsga2, "nsga3": hvs_nsga3, "moea/wst": hvs_moea, "moea/d": hvs_moead}
out_file_hvs = open("Risultati_pymoo\\hypervolume.json", "w")
json.dump(hypervolume, out_file_hvs)
