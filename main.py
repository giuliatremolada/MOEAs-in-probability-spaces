import numpy as np
import json
from matplotlib import pyplot as plt

f_hvs = open("Risultati_dtlz\\hypervolume.json",)
hypervolume = json.load(f_hvs)
f_time = open("Risultati\\time.json",)
time = json.load(f_time)

mean_time_qparego = np.asarray(time['parego time']).mean(axis=0)
mean_time_nsga = np.asarray(time['nsga time']).mean(axis=0)

time_qparego = [0]
count = 0
for i in mean_time_qparego:
    count += i
    time_qparego.append(count)

time_nsga = []
count = 0
for i in mean_time_nsga:
    count += i
    time_nsga.append(count)

iters = np.arange(250) * 10

time = [time_qparego, time_nsga]

hvs_mean_nsga2 = np.asarray(hypervolume['nsga2']).mean(axis=0)
hvs_mean_nsga3 = np.asarray(hypervolume['nsga3']).mean(axis=0)
hvs_mean_moea = np.asarray(hypervolume['moea/wst']).mean(axis=0)
hvs_mean_moead = np.asarray(hypervolume['moea/d']).mean(axis=0)
hvs_std_nsga2 = np.asarray(hypervolume['nsga2']).std(axis=0)
hvs_std_nsga3 = np.asarray(hypervolume['nsga3']).std(axis=0)
hvs_std_moea = np.asarray(hypervolume['moea/wst']).std(axis=0)
hvs_std_moead = np.asarray(hypervolume['moea/d']).std(axis=0)

means = [hvs_mean_nsga2, hvs_mean_nsga3, hvs_mean_moea, hvs_mean_moead] # List of lists (medie su tutti i trial)
stds = [hvs_std_nsga2, hvs_std_nsga3, hvs_std_moea, hvs_std_moead] # List of lists (deviazione standard su tutti i trial)

colors = ["orange", "blue", "green", "grey"] # List of n colors (string)
line_labels = ["Nsga-II", "Nsga-III", "Moea/wst", "Moea/d"] # List of n labels (string)

plt.rc('font', size=12)  # controls default text sizes
plt.rc('axes', titlesize=14)  # fontsize of the axes title
plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.rc('legend', fontsize=12)  # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title

"""
for idx, value in enumerate(line_labels):
    # Linea corrispondente alla media
    plt.plot(time[idx], means[idx], color=colors[idx], label=line_labels[idx])
    # STD
    lower = means[idx] - stds[idx]
    upper = means[idx] + stds[idx]
    lower[lower < 0] = 0
    # Deviazione standard attorno alla media
    plt.fill_between(time[idx], lower, upper, color=colors[idx], alpha=0.2)
"""

for idx, value in enumerate(line_labels):
    # Linea corrispondente alla media
    plt.plot(iters, means[idx], color=colors[idx], label=line_labels[idx])
    # STD
    lower = means[idx] - stds[idx]
    upper = means[idx] + stds[idx]
    lower[lower < 0] = 0
    # Deviazione standard attorno alla media
    plt.fill_between(iters, lower, upper, color=colors[idx], alpha=0.2)

#plt.xlabel("Time(s)")
plt.xlabel("Number of observations")
plt.ylabel("Hypervolume")
plt.legend(loc="lower right")


#plt.savefig("C:/Users/39348/PycharmProjects/Multiobjective_Optimization/Risultati/time_plot.png")
plt.savefig("C:/Users/39348/PycharmProjects/Multiobjective_Optimization/Risultati_pymoo/iterations_plot.png")
plt.show()

