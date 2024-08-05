import logging
import math
from math import pi
import numpy as np
from collections import OrderedDict

import plotly.graph_objects as go

from staliro.core import worst_eval, worst_run
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model, staliro
from staliro.options import SignalOptions

from autotrans import AutotransModel
from staliroBoInterface import Behavior, BO
from bo.gprInterface import InternalGPR
from bo.bayesianOptimization import InternalBO

import pickle

INIT_BUDGET = 50
MAX_BUDGET = 100

AT2_phi = "G[0, 20] (speed <= 120)"
specification = RTAMTDense(AT2_phi, {"speed":0, "rpm": 1})

gpr_model = InternalGPR()
bo_model = InternalBO()
optimizer = BO(10, gpr_model, bo_model, "lhs_sampling", Behavior.FALSIFICATION)

signals = [
            SignalOptions(control_points = [(0, 100)]*7, signal_times=np.linspace(0.,50.,7)),
            SignalOptions(control_points = [(0, 325)]*3, signal_times=np.linspace(0.,50.,3)),
        ]

options = Options(runs=1, iterations= MAX_BUDGET, interval=(0, 50),  signals=signals)

f16_model = AutotransModel()


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    result = staliro(f16_model, specification, optimizer, options)
    worst_sample = worst_eval(worst_run(result)).sample
    worst_result = simulate_model(f16_model, options, worst_sample)

    print(worst_sample)
    with open("AT_monitor_results.pkl", "wb") as f:
        pickle.dump(result, f)
    
    # print(result)