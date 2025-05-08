import pickle
import unittest

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from numpy.typing import NDArray

from bo.core import BOSampling, InternalBO
from bo.gpr import InternalGPR
from bo.sampling import uniform_sampling
from bo.utils import Fn, compute_robustness

bo = BOSampling(InternalBO())

def internal_function(X: NDArray[np.float_]) -> float:
    return float(X[0] ** 2 + X[1]**2 + X[2]**2)

rng = default_rng(12345)

region_support = np.array([[-1, 1], [-1, 1], [-1, 1]])
tf_dimension = 3
func1 = Fn(internal_function)
in_samples_1 = uniform_sampling(19, region_support, tf_dimension, rng)
out_samples_1 = compute_robustness(in_samples_1, func1)

gpr_model = InternalGPR()

num_samples = 10
result = bo.sample(func1, tf_dimension, 20, 20, region_support, gpr_model, rng, in_samples_1, out_samples_1)

print(result.initial_points.x_points.shape)
print(result.initial_points.y_points.shape)
print(result.initial_points_sampled.x_points.shape)
print(result.initial_points_sampled.y_points.shape)
print(result.sampled_points.x_points.shape)
print(result.sampled_points.y_points.shape)
