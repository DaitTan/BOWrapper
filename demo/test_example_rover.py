import logging
import math
from math import pi
import numpy as np
from collections import OrderedDict
import plotly.graph_objects as go

from RoverModel import RoverModel

from staliro.core import worst_eval, worst_run
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model, staliro


from staliroBoInterface import Behavior, BO
from bo.gprInterface import InternalGPR
from bo.bayesianOptimization import InternalBO

import pickle

# Define the specification
phi = "always(x<=1 and y<=1)"
specification = RTAMTDense(phi, {"x": 0, "y": 1})

# Define the optimizer
gpr_model = InternalGPR()
bo_model = InternalBO()
optimizer = BO(20, gpr_model, bo_model, "lhs_sampling", Behavior.FALSIFICATION)

# Initial Search Conditions
initial_conditions = [
    [0.1, 0.3], #ti bounds
    [0.1,0.3], #gr bounds,
    [0.1,0.3]
]

options = Options(runs=1, iterations=100, interval=(0, 15), static_parameters=initial_conditions)

f16_model = RoverModel()


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    result = staliro(f16_model, specification, optimizer, options)
    best_sample = worst_eval(worst_run(result)).sample
    best_result = simulate_model(f16_model, options, best_sample)
    # print(best_result.trace.states)
    # print(np.array(best_result.trace.states).shape)
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
    with open("RoverModel.pkl", "wb") as f:
        pickle.dump(result, f)