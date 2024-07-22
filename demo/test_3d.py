import pickle
import numpy as np
from numpy.random import default_rng
import pickle
import unittest

from bo.gprInterface import InternalGPR
from bo.utils import Fn, compute_robustness
from bo.sampling import uniform_sampling
from bo.gprInterface import internalGPR
from bo.bayesianOptimization import BOSampling, InternalBO
from matplotlib import pyplot as plt



bo = BOSampling(InternalBO())

def internal_function(X):
    return X[0] ** 2

rng = default_rng(12345)

region_support = np.array([[-1, 1]])
tf_dimension = 1
func1 = Fn(internal_function)
in_samples_1 = uniform_sampling(20, region_support, tf_dimension, rng)
out_samples_1 = compute_robustness(in_samples_1, func1)

gpr_model = InternalGPR()

num_samples = 10
x_complete, y_complete = bo.sample(func1, num_samples, in_samples_1, out_samples_1, region_support, gpr_model, rng)

print(x_complete)
print(y_complete)