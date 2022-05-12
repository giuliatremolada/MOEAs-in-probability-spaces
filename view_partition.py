import json
from matplotlib import pyplot as plt
import numpy as np

f_part = open("Risultati_pymoo\\partizioni.json",)
partizioni = json.load(f_part)

hvs_mean = np.asarray(partizioni['nsga3']).mean(axis=0)

iters = np.arange(250) * 10
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for partition, hv in enumerate(hvs_mean):
    ax.errorbar(
        iters, hv,
        label=partition + 1, linewidth=1.5,
    )

ax.set(xlabel='number of observations', ylabel='Hypervolume')
ax.legend(loc="lower left")
plt.show()