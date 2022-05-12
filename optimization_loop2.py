#import os
#os.system('conda install botorch -c pytorch -c gpytorch')
#os.system('pip install -U pymoo')

import numpy as np
import torch
from botorch.models import ModelListGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import SumMarginalLogLikelihood
from pymoo.interface import sample

import multiobjective_problem as mp
import custom_callback
import pickle
import time
import json
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize


tkwargs = {
    "dtype": torch.double
}

with open('rating_matrix.pickle', 'rb') as f:
    rating_matrix = pickle.load(f)

no_of_films = 15
problem_botorch = mp.MultiObjectiveProblem(n_user = rating_matrix.shape[0],
                              L = no_of_films,
                              id_max_film = rating_matrix.shape[1] - 1,
                              f1=mp.coverage, f2=mp.novelty, f3=mp.accuracy).to(**tkwargs)

problem_nsga = mp.OptimizationProblem(n_user=rating_matrix.shape[0],
                              L = no_of_films,
                              id_max_film=rating_matrix.shape[1] - 1,
                              f1=mp.coverage_pymoo, f2=mp.novelty_pymoo, f3=mp.accuracy_pymoo)

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
    sampling = get_sampling('int_random') #funziona ma solo con un trial
    train_x_nsga = sample(sampling, n, 943 * no_of_films, xl=1, xu= 1681)
    train_x_parego = torch.tensor(train_x_nsga, **tkwargs)
    """

    train_x_parego= draw_sobol_samples( #funziona
        bounds=problem_botorch.bounds, n=n, q=1, seed= trial #torch.randint(1000000, (1,)).item()
    ).squeeze(-2)

    train_x_parego = torch.floor(train_x_parego)
    train_x_nsga = np.asarray(train_x_parego, dtype=int)
    """


    torch.manual_seed(trial)
    train_x_parego = torch.randint(1, 1681, (n, 943*no_of_films), **tkwargs)
    #train_x_parego = torch.floor(torch.rand(n, 943 * no_of_films, **tkwargs) * 1682)
    train_x_nsga = np.asarray(train_x_parego, dtype=int)
    print(train_x_nsga)
    print(train_x_parego)
    """
    train_obj = problem_botorch(train_x_parego)

    return train_x_parego, train_x_nsga, train_obj


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
        print(weights)
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y= train_obj))
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
            beta = 1,
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
from botorch.acquisition.monte_carlo import qUpperConfidenceBound, qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
#from botorch.utils.multi_objective.hypervolume import Hypervolume

N_TRIALS = 5

parego_data = []

hvs_qparego_all, hvs_qehvi_all, hvs_nsga_all, time_qparego_all, time_qehvi_all, time_nsga_all = [], [], [], [], [], []

#hv_botorch = botorch.utils.multi_objective.hypervolume.Hypervolume(ref_point=problem_botorch.ref_point)

for trial in range(1, N_TRIALS + 1):
    print(print(f"L = {no_of_films}"))
    hvs_qparego, iteration_time_qparego, iteration_time_nsga = [], [], []



    train_x_qparego, train_x_nsga, train_obj_qparego = generate_initial_data(n=10, trial=trial)
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

    # nsga-III
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=2)
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=10,
        n_offsprings=BATCH_SIZE,
        sampling= train_x_nsga, #get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=1, eta=3.0),
        mutation=get_mutation("perm_inv"),
        eliminate_duplicates=True,
        seed = trial
    )
    c = custom_callback.MyCallback()
    res = minimize(problem_nsga, algorithm, termination,
                   save_history=True,
                   seed=trial,
                   callback=c,
                   verbose=False)

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
    hvs_nsga_all.append(c.data["HV"])
    time_qparego_all.append(iteration_time_qparego)

    time_error = c.data["time"][0]
    for i in range(0, N_BATCH):
        c.data["time"][i] = c.data["time"][i] - time_error
    time_nsga_all.append(c.data["time"])



hypervolume = {"parego": hvs_qparego_all, "nsga": hvs_nsga_all}
time = {"parego time": time_qparego_all, "nsga time": time_nsga_all}
out_file_hvs = open("Risultati2\\hypervolume.json", "w")
out_file_time = open("Risultati2\\time.json", "w")
out_file_parego = open("Risultati2\\parego_data.json", "w")
out_file_nsga = open("Risultati2\\nsga_data.json", "w")
json.dump(hypervolume, out_file_hvs)
json.dump(time, out_file_time)
json.dump(parego_data, out_file_parego)
json.dump(c.data["nsga data"], out_file_nsga)
