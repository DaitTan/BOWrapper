import logging
import math
from math import pi
import numpy as np
from collections import OrderedDict
import plotly.graph_objects as go

from f16Model import F16Model

from staliro.core import worst_eval, worst_run
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model, staliro


from staliroBoInterface import Behavior, BO
from bo.gprInterface import InternalGPR
from bo.bayesianOptimization import InternalBO

import pickle
F16_PARAM_MAP = OrderedDict({
    'air_speed': {
        'enabled': False,
        'default': 540
    },
    'angle_of_attack': {
        'enabled': False,
        'default': np.deg2rad(2.1215)
    },
    'angle_of_sideslip': {
        'enabled': False,
        'default': 0
    },
    'roll': {
        'enabled': True,
        'default': None,
        'range': (pi / 4) + np.array((-pi / 20, pi / 30)),
    },
    'pitch': {
        'enabled': True,
        'default': None,
        'range': (-pi / 2) * 0.8 + np.array((0, pi / 20)),
    },
    'yaw': {
        'enabled': True,
        'default': None,
        'range': (-pi / 4) + np.array((-pi / 8, pi / 8)),
    },
    'roll_rate': {
        'enabled': False,
        'default': 0
    },
    'pitch_rate': {
        'enabled': False,
        'default': 0
    },
    'yaw_rate': {
        'enabled': False,
        'default': 0
    },
    'northward_displacement': {
        'enabled': False,
        'default': 0
    },
    'eastward_displacement': {
        'enabled': False,
        'default': 0
    },
    'altitude': {
        'enabled': False,
        # 'default': 2338.4
        'default': 2335
    },
    'engine_power_lag': {
        'enabled': False,
        'default': 9
    }
})



# Define the specification
phi = "always(alt>0.0)"
specification = RTAMTDense(phi, {"alt": 0})

# Define the optimizer
gpr_model = InternalGPR()
bo_model = InternalBO()
optimizer = BO(100, gpr_model, bo_model, "lhs_sampling", Behavior.MINIMIZATION)

# Initial Search Conditions
initial_conditions = [
    (math.pi / 4) + np.array((-math.pi / 20, math.pi / 30)),
    (-math.pi / 2) * 0.8 + np.array((0, math.pi / 20)),
    (-math.pi / 4) + np.array((-math.pi / 8, math.pi / 8)),
]

options = Options(runs=10, iterations=1000, interval=(0, 15), static_parameters=initial_conditions, seed = 12345)


f16_model =  F16Model(F16_PARAM_MAP, 35, "stevens", 'euler')

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    result = staliro(f16_model, specification, optimizer, options)
    # best_sample = worst_eval(worst_run(result)).sample
    # best_result = simulate_model(f16_model, options, best_sample)
    # # print(best_result.trace.states)
    # # print(np.array(best_result.trace.states).shape)
    # figure = go.Figure()
    # figure.add_trace(
    #     go.Scatter(
    #         name="Altitude",
    #         x=np.array(best_result.trace.times),
    #         y=np.array(best_result.trace.states)[:,2],
    #         mode="lines",
    #         line_color = "blue",
    #     )
    # )
    # figure.add_trace(
    #     go.Scatter(
    #         name="Speed",
    #         x=np.array(best_result.trace.times),
    #         y=np.array(best_result.trace.states)[:,5],
    #         mode="lines",
    #         line_color = "green",
    #     )
    # )
    # figure.update_layout(title=f"Example 1: {[round(a,5) for a in best_sample.values]}", xaxis_title="time (s)")
    # figure.write_image("fig1.pdf")
    with open("HighFidelity_stevens_euler_35.pkl", "wb") as f:
        pickle.dump(result, f)