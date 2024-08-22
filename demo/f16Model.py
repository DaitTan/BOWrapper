from staliro.core.interval import Interval
from staliro.core.model import Model, ModelInputs, Trace, ExtraResult
import numpy as np
from numpy.typing import NDArray

from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot

from time import perf_counter

F16DataT = NDArray[np.float_]
F16ResultT = ExtraResult[F16DataT, list]


class F16Model(Model[F16ResultT, None]):
    def __init__(self, static_params_map, step_size, model, integrator) -> None:
        self.F16_PARAM_MAP = static_params_map
        self.step_size = step_size
        self.model = model
        self.integrator = integrator

    def get_static_params(self):
        static_params = []
        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                static_params.append(config['range'])
        return static_params


    def _compute_initial_conditions(self, X):
        conditions = []
        index = 0

        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                conditions.append(X[index])
                index = index + 1
            else:
                conditions.append(config['default'])

        return conditions

    def simulate(
        self, inputs: ModelInputs, intrvl: Interval
    ) -> F16ResultT:
        
        init_cond = self._compute_initial_conditions(inputs.static)
        
        step = 1 / self.step_size
        autopilot = GcasAutopilot(init_mode="roll", stdout=False, gain_str="old")

        start_time = perf_counter()
        result = run_f16_sim(init_cond, intrvl.upper, autopilot, step, extended_states=True, model_str=self.model, integrator_str=self.integrator)
        end_time = perf_counter() - start_time

        trajectories = result["states"][:, 11:12].T.astype(np.float64)
        
        timestamps = np.array(result["times"], dtype=(np.float32))
        outTrace = Trace(timestamps, trajectories)
        inTrace = inputs.static
        return F16ResultT(outTrace, [inTrace, self.step_size, self.model, self.integrator, end_time])