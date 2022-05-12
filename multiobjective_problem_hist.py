import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem

class MultiObjectiveProblem(BaseTestProblem):

    def __init__(self, rating_matrix, num_bin, sum_distr, L):
        self.num_bin = num_bin
        self.L = L
        self.sum_distr = sum_distr
        self.rating_matrix = rating_matrix
        self.n_user = int(self.rating_matrix.shape[0])
        self.num_objectives = 3
        n_var = (num_bin + 2) * self.num_objectives
        self.dim = n_var
        self.max_cov = self.L
        self.max_nov = 10
        self.max_acc = 5 * L # nell'accuratezza normale sarebbe 5*dimensioni matrice, qui ho persato di condiderare solo una riga
        self._bounds = [(0, self.n_user)]
        self._bounds.extend([(0, self.n_user) for _ in range(num_bin - 1)])
        self._bounds.extend([(0, self.max_cov) for _ in range(2)])
        self._bounds.extend([(0, self.n_user) for _ in range(num_bin)])
        self._bounds.extend([(0, self.max_nov) for _ in range(2)])
        self._bounds.extend([(0, self.n_user) for _ in range(num_bin)])
        self._bounds.extend([(0, self.max_acc) for _ in range(2)])
        self.ref_point = torch.tensor([0, 0, 0])

        super().__init__()

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = []
        for x in X:
            cov=x[0]
            nov=x[1]
            acc=x[2]
            cov_result, nov_result, acc_result = [], [], []

            for metric, result in zip(
                    (cov, nov, acc),
                    (cov_result, nov_result, acc_result),
            ):
                min_bin = metric[len(metric) - 2]

                difference = (metric[len(metric) - 1] - metric[len(metric) - 2]) / 10
                for i in range(len(metric) - 2):
                    max_bin = min_bin + difference
                    result.append(metric[i] * ((min_bin + max_bin) / 2))
                    min_bin = max_bin


            f.append([self.sum_distr(cov_result), self.sum_distr(nov_result), self.sum_distr(acc_result)])

        return torch.tensor(f)




