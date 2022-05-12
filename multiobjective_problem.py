import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem
from pymoo.core.problem import ElementwiseProblem

class MultiObjectiveProblem(BaseTestProblem):

    def __init__(self, rating_matrix, L, f1, f2, f3):

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.rating_matrix = rating_matrix
        self.n_user = int(self.rating_matrix.shape[0])
        n_var = int(self.n_user * L)
        self.L = int(L)
        self.id_max_film = int(self.rating_matrix.shape[1] - 1)
        self.dim = n_var
        self.num_objectives = 3
        self._bounds = [(0, self.id_max_film)]
        self._bounds.extend([(0, self.id_max_film) for _ in range(n_var - 1)])
        self.ref_point = torch.tensor([0, 0, 0])

        super().__init__()

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = []
        for x in X:
            x = np.reshape(x, (self.n_user, self.L))
            f.append([self.f1(x), self.f2(x), self.f3(x, self.rating_matrix)])
        return torch.tensor(f)


class OptimizationProblem(ElementwiseProblem):

    def __init__(self, rating_matrix, L, f1, f2, f3):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.rating_matrix = rating_matrix
        self.n_user = int(self.rating_matrix.shape[0])
        n_var = int(self.n_user * L)
        self.L = int(L)
        self.id_max_film = int(self.rating_matrix.shape[1]-1)
        super().__init__(n_var=n_var,
                         n_obj=3,
                         n_constr=0,
                         xl=np.full(n_var, 0),
                         xu=np.full(n_var, self.id_max_film))
        self.hist_nov = []
        self.hist_cov = []
        self.hist_acc = []

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array(x)
        x = np.reshape(x, (self.n_user, self.L))
        out['F'] = [self.f1(x), self.f2(x), self.f3(x, self.rating_matrix)]


