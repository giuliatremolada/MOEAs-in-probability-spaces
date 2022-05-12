import numpy as np
import time

from pymoo.indicators.hv import Hypervolume
from pymoo.core.callback import Callback

def calcola_hypervolume(res_f):
    ref_point = np.array([0.0, 0.0, 0.0])
    # create the performance indicator object with reference point
    metric = Hypervolume(ref_point = ref_point,
                         norm_ref_point = True,
                         zero_to_one = True,
                         ideal = np.array([-1, -10, -1]),
                         nadir = np.array([0,0,0])
                         )

    # calculate for each generation the HV metric
    hv = metric.do(res_f)
    return hv


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["HV"] = []
        self.data["time"] = []
        # self.data["nsga data"] = []

    def notify(self, algorithm, **kwargs):
        self.data["HV"].append(calcola_hypervolume(algorithm.opt.get("F")))
        self.data["time"].append(time.time())
        # iteration_dict = {'pareto front': algorithm.opt.get("F").tolist()}
        # self.data["nsga data"].append(iteration_dict)