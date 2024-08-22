from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple
import logging

import numpy as np
from staliro.core import Interval, Optimizer, ObjectiveFn, Sample

from bo.bayesianOptimization import BOSampling, Behavior
from bo.utils import compute_robustness
from bo.sampling import uniform_sampling, lhs_sampling

Bounds = Sequence[Interval]
BOResult = tuple[Any,Any]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class BO(Optimizer[float, BOResult]):
    """The BO  optimizer runs the BO code."""

    init_sampling_budget: int
    gpr_model: Any
    bo_model: Any
    init_sampling_method: str
    behavior: Behavior
    

    def optimize(self, func: ObjectiveFn[float], bounds: Bounds, budget: int, seed: int) -> BOResult:
        if self.init_sampling_budget > budget:
            raise ValueError("Init Sampling budget cannot be greater than Maximmum Budget")
        region_support = np.array((tuple(bound.astuple() for bound in bounds),))[0]
        
        def test_function(sample) -> float:
            return func.eval_sample(Sample(sample))
        
        rng = np.random.default_rng(seed)
        bo = BOSampling(self.bo_model)

        if self.init_sampling_method == "uniform_sampling":
            in_samples_1 = uniform_sampling(self.init_sampling_budget, region_support, len(region_support), rng)
        elif self.init_sampling_method == "lhs_sampling":
            in_samples_1 = lhs_sampling(self.init_sampling_budget, region_support, len(region_support), rng)
        
        out_samples_1 = []
        for iter, single_sample in enumerate(in_samples_1):
            
            sample_cost = compute_robustness(np.array([single_sample]), test_function)
            out_samples_1.append(sample_cost)
            logger.debug(f"Cost {sample_cost}")
            if self.behavior == Behavior.FALSIFICATION:
            # print(f"Cost {sample_cost}")
                if sample_cost <= 0:
                    logger.debug(f"Falsified during Initial Search")
                    return BOResult([in_samples_1[iter, :], np.array(out_samples_1)])

        out_samples_1 = np.array(out_samples_1).squeeze()

        return BOResult(bo.sample(test_function, budget-self.init_sampling_budget, in_samples_1, out_samples_1, region_support, self.gpr_model, self.behavior, rng))