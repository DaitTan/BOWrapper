import pickle
from typing import Callable, Tuple
from numpy.typing import NDArray
import numpy as np
import pathlib
from tqdm import tqdm

from .gprInterface import GPR
from .bayesianOptimization import BOSampling
from .utils import Fn, compute_robustness
from .sampling import uniform_sampling, lhs_sampling

class PerformBO:
    def __init__(
            self, 
            test_function: Callable,
            init_budget: int,
            max_budget: int,
            region_support: NDArray,
            seed,
            init_sampling_type = "lhs_sampling",
        ):
            """Internal BO Model

            Args:
                test_function: Function of System Under Test.
                num_samples: Number of samples to generate from BO.
                x_train: Samples from Training set.
                y_train: Evaluated values of samples from Trainig set.
                region_support: Min and Max of all dimensions
                gpr_model: Gaussian Process Regressor Model developed using Factory
                rng: RNG object from numpy

            Raises:
                TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
                TypeError: If y_train is not (n,) numpy array
                TypeError: If there is a mismatch between x_train and y_train

            Returns:
                x_complete
                y_complete
                x_new
                y_new
            """

            self.tf_wrapper = Fn(test_function)
            self.init_budget = init_budget
            self.max_budget = max_budget
            self.region_support = region_support
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.init_sampling_type = init_sampling_type
            

    def __call__(self, bo_model, gpr_model):
        tf_dim = self.region_support.shape[0]
        bo_routine = BOSampling(bo_model)
        

        if self.init_sampling_type == "lhs_sampling":
            x_train = lhs_sampling(self.init_budget, self.region_support, tf_dim, self.rng)
        elif self.init_sampling_type == "uniform_sampling":
            x_train = uniform_sampling(self.init_budget, self.region_support, tf_dim, self.rng)
        else:
            raise ValueError(f"{self.init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")

        y_train = compute_robustness(x_train, self.tf_wrapper)

        bo_routine.sample(self.tf_wrapper, self.max_budget - self.init_budget, x_train, y_train, self.region_support, gpr_model, self.rng)

        return self.tf_wrapper.point_history