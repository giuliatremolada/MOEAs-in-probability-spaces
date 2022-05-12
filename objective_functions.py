import math
import numpy as np


def accuracy_pymoo(top_recc, rating_matrix):
    cols = rating_matrix.columns.to_numpy() - 1
    total_score = 0
    for idx_user, user_list in enumerate(top_recc):
        total_score += sum(rating_matrix.iloc[idx_user, cols[user_list]].values)
    max_accuracy = 5 * top_recc.shape[0] * top_recc.shape[1]
    return -(total_score/max_accuracy)

def coverage_pymoo(top_recc):
  N = 1682
  Nu = np.unique(top_recc).shape[0]
  return -(Nu / N)

def novelty_pymoo(top_recc):
  M = top_recc.shape[0]
  L = top_recc.shape[1]
  unique, counts = np.unique(top_recc, return_counts=True)
  item_occurencies = dict(zip(unique, counts))
  nov = 0
  for i in range(M):
    for j in range(L):
      Ni = math.log2(M / item_occurencies[top_recc[i, j]])
      nov += Ni / L
  return -(nov / M)

def accuracy(top_recc, rating_matrix):
    cols = rating_matrix.columns.to_numpy() -1
    total_score = 0
    for idx_user, user_list in enumerate(top_recc.int()):
        total_score += sum(rating_matrix.iloc[idx_user, cols[user_list]].values)
    max_accuracy = 5*top_recc.shape[0]*top_recc.shape[1]
    return total_score/max_accuracy
    # max 5*dimensione matrice


def coverage(top_recc):
    N = 1682
    Nu = np.unique(top_recc).shape[0]
    return Nu / N


def novelty(top_recc):
    M = top_recc.shape[0]
    L = top_recc.shape[1]
    unique, counts = np.unique(top_recc, return_counts=True)
    item_occurencies = dict(zip(unique, counts))
    nov = 0
    for i in range(M):
        for j in range(L):
            Ni = math.log2(M / item_occurencies[top_recc[i, j].tolist()])
            nov += Ni / L
    return nov / M


def coverage_distr(top_recc):
    N = top_recc.shape[0] * top_recc.shape[1]
    distr = []
    for i in range(top_recc.shape[0]):
        distr.append(np.unique(top_recc[i]).shape[0])
    return distr


def novelty_distr(top_recc):
    # M is the total number of users
    M = top_recc.shape[0]
    L = top_recc.shape[1]
    unique, counts = np.unique(top_recc, return_counts=True)
    item_occurencies = dict(zip(unique, counts))
    distr = []
    for i in range(M):
        user_nov = 0
        for j in range(L):
            # Compute The self information of item j (Nj)
            Ni = math.log2(M / item_occurencies[top_recc[i, j].tolist()])
            user_nov += Ni / L
        distr.append(user_nov)
    return distr


def accuracy_distr(top_recc, rating_matrix):
    cols = rating_matrix.columns.to_numpy()-1
    distr = []
    for idx_user, user_list in enumerate(top_recc.int()):
        distr.append(sum(rating_matrix.iloc[idx_user, cols[user_list]].values))
    return distr

def coverage_distr_pymoo(top_recc):
    N = top_recc.shape[0] * top_recc.shape[1]
    distr = []
    for i in range(top_recc.shape[0]):
        distr.append(np.unique(top_recc[i]).shape[0])
    return distr


def novelty_distr_pymoo(top_recc):
    # M is the total number of users
    M = top_recc.shape[0]
    L = top_recc.shape[1]
    unique, counts = np.unique(top_recc, return_counts=True)
    item_occurencies = dict(zip(unique, counts))
    distr = []
    for i in range(M):
        user_nov = 0
        for j in range(L):
            # Compute The self information of item j (Nj)
            Ni = math.log2(M / item_occurencies[top_recc[i, j]])
            user_nov += Ni / L
        distr.append(user_nov)
    return distr


def accuracy_distr_pymoo(top_recc, rating_matrix):
    cols = rating_matrix.columns.to_numpy()-1
    distr = []
    for idx_user, user_list in enumerate(top_recc):
        distr.append(sum(rating_matrix.iloc[idx_user, cols[user_list]].values))
    return distr

def sum_distr(distr):
    sum=0
    for i in distr:
        sum += i
    return sum
