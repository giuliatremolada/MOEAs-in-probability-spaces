import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.selection import Selection
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from scipy.stats import wasserstein_distance


def cwstdist(P, Q):
    mat = np.zeros((P.shape[0], Q.shape[0]))
    for i, r1 in enumerate(P):
        for j, r2 in enumerate(Q):
            mat[i, j] = wasserstein_distance(r1, r2)
    return mat

# =========================================================================================================
# Neighborhood Selection
# =========================================================================================================


class NeighborhoodSelection(Selection):

    def __init__(self, neighbors, prob=1.0) -> None:
        super().__init__()
        self.neighbors = neighbors
        self.prob = prob

    def _do(self, pop, n_select, n_parents, k=None, **kwargs):
        if k is None:
            k = np.random.permutation(len(pop))[:n_select]
        assert len(k) == n_select

        N = self.neighbors
        P = np.full((n_select, n_parents), -1)

        for i, j in enumerate(k):
            if np.random.random() < self.prob:
                P[i] = np.random.choice(N[j], n_parents, replace=False)
            else:
                P[i] = np.random.permutation(len(pop))[:n_parents]

        return P


# =========================================================================================================
# Implementation
# =========================================================================================================

class MOEADWST(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 sampling=FloatRandomSampling(),
                 crossover=SimulatedBinaryCrossover(prob=1.0, eta=20),
                 mutation=PolynomialMutation(prob=None, eta=20),
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """
        Parameters
        ----------
        ref_dirs
        n_neighbors
        decomposition
        prob_neighbor_mating
        display
        kwargs
        """

        self.ref_dirs = ref_dirs
        self.n_neighbors = min(len(ref_dirs), n_neighbors)
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomp = decomposition
        # t1 = cdist(self.ref_dirs, self.ref_dirs)
        # t2 = cwstdist(self.ref_dirs, self.ref_dirs)
        # neighbours includes the entry by itself intentionally for the survival method
        # self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]
        self.neighbors = np.argsort(cwstdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')
        self.neighbors = self.neighbors[:, self.neighbors.shape[1] - self.n_neighbors:]

        self.selection = NeighborhoodSelection(self.neighbors, prob=prob_neighbor_mating)

        super().__init__(pop_size=len(ref_dirs), sampling=sampling, crossover=crossover, mutation=mutation,
                         eliminate_duplicates=NoDuplicateElimination(), display=display,
                         advance_after_initialization=False, **kwargs)

        # the mating is just performed once here - population does not need to be filled up
        self.mating.n_max_iterations = 1

    def _setup(self, problem, **kwargs):
        assert not problem.has_constraints(), "This implementation of MOEAD does not support any constraints."

        if isinstance(self.decomp, str):
            # for one or two objectives use tchebi otherwise pbi
            if self.decomp == 'auto':
                if self.problem.n_obj <= 2:
                    from pymoo.decomposition.tchebicheff import Tchebicheff
                    self.decomp = Tchebicheff()
                else:
                    from pymoo.decomposition.pbi import PBI
                    self.decomp = PBI()

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self.ideal = np.min(self.pop.get("F"), axis=0)

    def _infill(self):
        # MOEA\D inherits from genetic algorithm but does not implement the infill/advance interface
        pass

    def _advance(self, **kwargs):
        pop = self.pop

        # iterate for each member of the population in random order
        for i in np.random.permutation(len(pop)):
            # get the parents using the neighborhood selection
            P = self.selection.do(pop, 1, self.mating.crossover.n_parents, k=[i])

            # perform a mating using the default operators - if more than one offspring just pick the first
            off = self.mating.do(self.problem, pop, 1, parents=P)[0]

            # evaluate the offspring
            self.evaluator.eval(self.problem, off, algorithm=self)

            # update the ideal point
            self.ideal = np.min(np.vstack([self.ideal, off.F]), axis=0)

            # now actually do the replacement of the individual is better
            self._replace(i, off)

    def _replace(self, i, off):
        pop = self.pop

        # calculate the decomposed values for each neighbor
        N = self.neighbors[i]
        FV = self.decomp.do(pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal)
        off_FV = self.decomp.do(off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal)

        # this makes the algorithm support constraints - not originally proposed though and not tested enough
        # if self.problem.has_constraints():
        #     CV, off_CV = pop[N].get("CV")[:, 0], np.full(len(off_FV), off.CV)
        #     fmax = max(FV.max(), off_FV.max())
        #     FV, off_FV = parameter_less(FV, CV, fmax=fmax), parameter_less(off_FV, off_CV, fmax=fmax)

        # get the absolute index in F where offspring is better than the current F (decomposed space)
        I = np.where(off_FV < FV)[0]
        pop[N[I]] = off