# import os
# os.system('conda install botorch -c pytorch -c gpytorch')
# os.system('pip install -U pymoo')


import numpy as np
import torch
from botorch.models.transforms import Standardize
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.interface import sample

import custom_callback
import pickle
import time
import json

from custom_selection import CustomSelection
from multiobjective_problem import OptimizationProblem, MultiObjectiveProblem
from custom_callback import MyCallback
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.factory import get_sampling, get_crossover, get_mutation
from objective_functions import accuracy, novelty, coverage, accuracy_pymoo, novelty_pymoo, coverage_pymoo
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from moead_wst import MOEADWST

tkwargs = {
    "dtype": torch.double
}

with open('rating_matrix.pickle', 'rb') as f:
    rating_matrix = pickle.load(f)

no_of_films = 15
problem_botorch = MultiObjectiveProblem(rating_matrix=rating_matrix,
                                        L=no_of_films,
                                        f1=coverage, f2=novelty, f3=accuracy).to(**tkwargs)

problem_nsga = OptimizationProblem(rating_matrix=rating_matrix,
                                   L=no_of_films,
                                   f1=coverage_pymoo, f2=novelty_pymoo, f3=accuracy_pymoo)

from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.optim.optimize import optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex, draw_sobol_samples

BATCH_SIZE = 4
N_BATCH = 20

standard_bounds = torch.zeros(2, problem_botorch.dim, **tkwargs)
standard_bounds[1] = 1


def generate_initial_data(n, trial):
    # generate training data
    """
    torch.manual_seed(trial + 1)
    sampling = get_sampling('int_random') #funziona ma solo con un trial
    train_x_nsga3 = sample(sampling, n, 943 * no_of_films, xl=1, xu= 1681)
    train_x_parego = torch.tensor(train_x_nsga, **tkwargs)
    """
    torch.manual_seed(trial) # di solito funziona con trial+5
    train_x_parego = torch.floor(torch.rand(n, 943 * no_of_films, **tkwargs) * 1682)
    train_x_ga = np.asarray(train_x_parego, dtype=int)

    train_obj = problem_botorch(train_x_parego)

    return train_x_parego, train_x_ga, train_obj


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def optimize_qparego_and_get_observation(model, train_obj, sampler):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qParEGO acquisition function, and returns a new candidate and observation."""
    acq_func_list = []
    for _ in range(BATCH_SIZE):
        weights = sample_simplex(problem_botorch.num_objectives, **tkwargs).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=train_obj))
        """
        acq_func = qExpectedImprovement(
            model=model,
            objective=objective,
            best_f=objective(train_obj).max().item(),
            sampler=sampler,
        )
        """
        acq_func = qUpperConfidenceBound(
            model=model,
            beta=100,
            objective=objective,
            sampler=sampler,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds= standard_bounds,
        num_restarts=20,
        raw_samples=1024,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},

    )
    # observe new values
    new_x = torch.floor(unnormalize(candidates.detach(), bounds=problem_botorch.bounds))
    new_obj = problem_botorch(new_x)
    return new_x, new_obj


termination = get_termination("n_gen", N_BATCH)

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
# from botorch.utils.multi_objective.hypervolume import Hypervolume

N_TRIALS = 5

parego_data = []

hvs_qparego_all, hvs_nsga2_all, hvs_nsga3_all, \
hvs_moea_wst_all = [], [], [], []

#hv_botorch = botorch.utils.multi_objective.hypervolume.Hypervolume(ref_point=problem_botorch.ref_point)

for trial in range(1, N_TRIALS + 1):
    hvs_qparego, iteration_time_qparego, iteration_time_nsga = [], [], []

    train_x_qparego, train_x_nsga, train_obj_qparego = generate_initial_data(n=10, trial=trial+2)
    mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)
    # compute pareto front
    pareto_mask = is_non_dominated(train_obj_qparego)
    pareto_y = train_obj_qparego[pareto_mask]
    pareto_y = pareto_y.tolist()
    pareto_y = np.asarray(pareto_y)
    # compute hypervolume
    volume = custom_callback.calcola_hypervolume(- pareto_y)
    hvs_qparego.append(volume)

    iteration_dict = {'train_obj': train_obj_qparego.tolist(), 'pareto front': (-pareto_y).tolist()}
    parego_data.append(iteration_dict)

    # nsga-II
    algorithm_nsga2 = NSGA2(
        pop_size=10,
        n_offsprings=BATCH_SIZE,
        sampling= train_x_nsga, #get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=1, eta=3.0),
        mutation=get_mutation("perm_inv"),
        eliminate_duplicates=True
    )
    c_nsga2 = MyCallback()
    res_nsga2 = minimize(problem_nsga, algorithm_nsga2, termination,
                   save_history=True,
                   callback=c_nsga2,
                   verbose=False)

    #nsga-III
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=2)
    algorithm_nsga3 = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=10,
        n_offsprings=BATCH_SIZE,
        sampling=train_x_nsga,  # get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=1, eta=3.0),
        mutation=get_mutation("perm_inv"),
        eliminate_duplicates=True
    )
    c_nsga3 = MyCallback()
    res_nsga3 = minimize(problem_nsga, algorithm_nsga3, termination,
                         save_history=True,
                         callback=c_nsga3,
                         verbose=False)

    # moea/wst
    algorithm_moea_wst = NSGA2(
        pop_size=10,
        n_offsprings=BATCH_SIZE,
        sampling= train_x_nsga, #get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=1, eta=3.0),
        mutation=get_mutation("perm_inv"),
        selection=CustomSelection(),
        eliminate_duplicates=True
    )
    c_moea_wst = MyCallback()
    res_moea_wst = minimize(problem_nsga, algorithm_moea_wst, termination,
                   save_history=True,
                   callback=c_moea_wst,
                   verbose=False)
    """
    ### MOEA/D
    algorithm_moea_d = MOEAD(ref_dirs=ref_dirs,
                             n_offsprings=BATCH_SIZE,
                             n_neighbors=3,
                             sampling=train_x_nsga,
                             crossover=get_crossover("int_sbx", prob=1, eta=3.0),
                             mutation=get_mutation("perm_inv"),
                             prob_neighbor_mating=0.7,
                             seed=trial+2)

    c_moea_d = MyCallback()
    res_moea_d = minimize(problem_nsga, algorithm_moea_d, termination,
                            save_history=True,
                            verbose=True,
                            callback=c_moea_d,
                            seed=trial+2)
    
    # MOEA/D/WST
    algorithm_moead_wst = MOEADWST(
        ref_dirs,
        n_offspring=BATCH_SIZE,
        n_neighbors=5,
        sampling=train_x_nsga,
        crossover=get_crossover("int_sbx", prob=1, eta=3.0),
        mutation=get_mutation("perm_inv"),
        prob_neighbor_mating=0.7
    )
    c_moead_wst = MyCallback()

    res_moead_wst = minimize(problem_nsga,
                             algorithm_moead_wst,
                             termination,
                             seed=trial+2,
                             save_history=True,
                             callback=c_moead_wst,
                             verbose=True)
    """
    for iteration in range(1, N_BATCH):
        print(f"CICLO {iteration} in trial {trial}")
        #qParego
        t0_qParego = time.time()

        fit_gpytorch_model(mll_qparego)

        qparego_sampler = SobolQMCNormalSampler(num_samples=128)
        new_x_qparego, new_obj_qparego = optimize_qparego_and_get_observation(
            model_qparego, train_obj_qparego, qparego_sampler
        )
        # update training points
        train_x_qparego = torch.cat([train_x_qparego, new_x_qparego])
        train_obj_qparego = torch.cat([train_obj_qparego, new_obj_qparego])

        # compute pareto front
        pareto_mask = is_non_dominated(train_obj_qparego)
        pareto_y = train_obj_qparego[pareto_mask]
        pareto_y = pareto_y.tolist()
        pareto_y = np.asarray(pareto_y)
        # compute hypervolume
        volume = custom_callback.calcola_hypervolume(- pareto_y)
        hvs_qparego.append(volume)

        iteration_dict = {'train_obj': train_obj_qparego.tolist(), 'pareto front': (-pareto_y).tolist()}
        parego_data.append(iteration_dict)

        mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)

        t1_qParego = time.time()
        iteration_time_qparego.append(t1_qParego - t0_qParego)

    hvs_qparego_all.append(hvs_qparego)
    hvs_nsga2_all.append(c_nsga2.data["HV"])
    hvs_nsga3_all.append(c_nsga3.data["HV"])
    hvs_moea_wst_all.append(c_moea_wst.data["HV"])
    #hvs_moead_all.append(c_moea_d.data["HV"])
    #hvs_moead_wst_all.append(c_moead_wst.data["HV"])


hypervolume = {"parego": hvs_qparego_all, "nsga2": hvs_nsga2_all, "nsga3": hvs_nsga3_all, "moea/wst": hvs_moea_wst_all}#, "moea/d": hvs_moead_all,
              # "moead/wst": hvs_moead_wst_all}
out_file_hvs = open("Risultati\\hypervolume.json", "w")
json.dump(hypervolume, out_file_hvs)

