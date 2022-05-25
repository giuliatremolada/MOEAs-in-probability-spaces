import numpy as np

from pymoo.indicators.hv import Hypervolume
from pymoo.core.callback import Callback


def calcola_hypervolume(res_f):
    ref_point = np.array([0.0, 0.0, 0.0])
    # create the performance indicator object with reference point
    metric = Hypervolume(ref_point = ref_point,
                         norm_ref_point = True,
                         zero_to_one = True,
                         ideal = np.array([-1, -10, -1]),
                         nadir = np.array([0, 0, 0])
                         )

    # calculate for each generation the HV metric
    hv = metric.do(res_f)
    return hv


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["HV"] = []

    def notify(self, algorithm, **kwargs):
        self.data["HV"].append(calcola_hypervolume(algorithm.opt.get("F")))
