import numpy as np

from pymoo.indicators.hv import Hypervolume
from pymoo.core.callback import Callback


def calcola_hypervolume_dtlz(res_f):
    ref_point = np.array([5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5])

    # create the performance indicator object with reference point
    metric = Hypervolume(ref_point=ref_point,
                         norm_ref_point = True,
                         zero_to_one = True,
                         ideal = np.array([0, 0, 0, 0, 0, 0, 0]),
                         nadir = np.array([5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5])
                         )

    # calculate for each generation the HV metric
    hv = metric.do(res_f)
    return hv


class MyCallback_dtlz(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["HV"] = []


    def notify(self, algorithm, **kwargs):
        self.data["HV"].append(calcola_hypervolume_dtlz(algorithm.opt.get("F")))
        print(algorithm.opt.get("F"))