import math
import numpy as np
import ot
import random
from itertools import permutations

from pymoo.core.selection import Selection
from pymoo.util.misc import random_permuations
from scipy.stats import wasserstein_distance

from objective_functions import accuracy_distr_pymoo, coverage_distr_pymoo, novelty_distr_pymoo


def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Pareto front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def binary_tournament(pop, P, problem):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")
    if P.shape[2] != 2:
        raise ValueError("Only implemented for two parents case!")

    S = []
    for i in range(P.shape[0]):
        parents1, parents2 = P[i, 0], P[i, 1]
        x_parents = [[
            np.reshape(pop[parents1[0]].get("X"), (problem.n_user, problem.L)),
            np.reshape(pop[parents1[1]].get("X"), (problem.n_user, problem.L))
        ], [
            np.reshape(pop[parents2[0]].get("X"), (problem.n_user, problem.L)),
            np.reshape(pop[parents2[1]].get("X"), (problem.n_user, problem.L))
        ]]

        # Get histograms of the pairs of parents
        hists = []
        xbins, ybins = None, None
        edges = []
        for pairs in x_parents:
            pair_hist = []
            for individual in pairs:
                # hist, xbins, ybins = np.histogram2d(coverage_distr(individual), novelty_distr(individual),
                #                                     bins=[[22, 23, 24, 25],
                #                                           np.arange(5, 8, 0.2)])
                cov, nov = coverage_distr_pymoo(individual), novelty_distr_pymoo(individual)
                acc = accuracy_distr_pymoo(individual, problem.rating_matrix)
                distr = np.stack((cov, nov, acc), axis=1)
                H, edges = np.histogramdd(distr, bins=[[12, 13, 14, 15], np.arange(5, 8, 0.2), np.arange(50, 110, 4)]) # era 22 23 24 25
                pair_hist.append(H)
            hists.append(pair_hist)

        # xbins = xbins[:-1]
        # ybins = ybins[:-1]
        # support = np.array([[x, y] for x in xbins for y in ybins])
        support = np.array([[x, y, z] for x in edges[0][:-1] for y in edges[1][:-1] for z in edges[2][:-1]])
        # Compute 2d-Wasserstein
        M = ot.dist(support, support)
        M /= M.max()

        parents1_ws = ot.emd2(hists[0][0].flatten(), hists[0][1].flatten(), M)
        parents2_ws = ot.emd2(hists[1][0].flatten(), hists[1][1].flatten(), M)

        # Compare the two pair of parents and get the best (the one with higher distance)
        winner = compare(parents1, parents1_ws, parents2, parents2_ws, method='larger_is_better',
                         return_random_if_equal=True)
        S.append(winner)
    return np.array(S)


def compare(parents1, parents1_val, parents2, parents2_val, method, return_random_if_equal=False):
    if method == 'larger_is_better':
        if parents1_val > parents2_val:
            return parents1
        elif parents1_val < parents2_val:
            return parents2
        else:
            if return_random_if_equal:
                if np.random.choice([0, 1]) == 0:
                    return parents1
                else:
                    return parents2
            else:
                return None
    elif method == 'smaller_is_better':
        if parents1_val < parents2_val:
            return parents1
        elif parents1_val > parents2_val:
            return parents2
        else:
            if return_random_if_equal:
                if np.random.choice([0, 1]) == 0:
                    return parents1
                else:
                    return parents2
            else:
                return None
    else:
        raise Exception("Unknown method.")


class CustomSelection(Selection):

    def _do(self, pop, n_select, n_parents, **kwargs):
        # Number of random individuals needed
        n_random = n_select * 2 * n_parents
        # Get pareto indeces
        F = np.array([individual.F for individual in pop])
        PF_indeces = identify_pareto(F)
        # Get random permutations
        P = list(permutations(PF_indeces, 2))
        # Sample from pareto front
        if len(P) >= n_random / 2:
            random.shuffle(P)
            P = P[:int(n_random / 2)]
            # Convert list of tuple in list of list
            P = [list(ele) for ele in P]
        else:
            # Sample from the entire population
            # Number of permutations needed
            n_perms = math.ceil(n_random / len(pop))
            # Get random permutations
            P = random_permuations(n_perms, len(pop))[:n_random]
        # Reshape the permutations
        P = np.reshape(P, (n_select, n_parents, 2))
        # Start the tournament
        S = binary_tournament(pop, P, kwargs["algorithm"].problem)
        return S

