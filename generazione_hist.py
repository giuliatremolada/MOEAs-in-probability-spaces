import pickle

import numpy as np
import torch

from multiobjective_problem_hist import MultiObjectiveProblem
from objective_functions import accuracy_distr, novelty_distr, coverage_distr, sum_distr

with open('rating_matrix.pickle', 'rb') as f:
    rating_matrix = pickle.load(f)

tkwargs = {
    "dtype": torch.double
}

problem_parego = MultiObjectiveProblem(rating_matrix = rating_matrix, num_bin=10,
                              sum_distr= sum_distr, L=15)


initial_data_parego = torch.floor(torch.rand(10, 943 * 15, **tkwargs) * 1682)

train_x_distr = []
for x in initial_data_parego:

    x = np.reshape(x, (943, 15))
    cov = torch.tensor(coverage_distr(x))
    nov = torch.tensor(novelty_distr(x))
    acc = torch.tensor(accuracy_distr(x, rating_matrix))
    funz = torch.stack([cov, nov, acc])
    train_x_distr.append(funz)

train_x_distr = torch.stack(train_x_distr)

train_x_hist, train_x_bins = [], []
for x in train_x_distr:
    hist_cov = np.histogram(x[0]) # con range = (0,15) si fissano gli estremi
    hist_nov = np.histogram(x[1])
    hist_acc = np.histogram(x[2])
    min_cov_bin = min(hist_cov[1])
    max_cov_bin = max(hist_cov[1])
    min_nov_bin = min(hist_nov[1])
    max_nov_bin = max(hist_nov[1])
    min_acc_bin = min(hist_acc[1])
    max_acc_bin = max(hist_acc[1])
    cov = torch.cat([torch.tensor(hist_cov[0]), torch.tensor([min_cov_bin, max_cov_bin])])
    nov = torch.cat([torch.tensor(hist_nov[0]), torch.tensor([min_nov_bin, max_nov_bin])])
    acc = torch.cat([torch.tensor(hist_acc[0]), torch.tensor([min_acc_bin, max_acc_bin])])
    train_x_hist.append(torch.stack([cov, nov, acc]))

train_x_hist = torch.stack(train_x_hist)
print(train_x_hist)
train_obj_parego = problem_parego(train_x_hist)
print(train_obj_parego)
