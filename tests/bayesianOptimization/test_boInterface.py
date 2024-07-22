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


class Test_bointerface(unittest.TestCase):
    def test1_boInterface(self):
        bo = BOSampling(InternalBO())

        def internal_function(X):
            return X[0] ** 2 + X[1]**2 + X[2]**2

        rng = default_rng(12345)

        region_support = np.array([[-1, 1], [-1, 1], [-1, 1]])

        func1 = Fn(internal_function)
        in_samples_1 = uniform_sampling(20, region_support, 3, rng)
        out_samples_1 = compute_robustness(in_samples_1, func1)

        in_samples_2 = uniform_sampling(30, region_support, 3, rng)
        out_samples_2 = compute_robustness(in_samples_2, func1)

        
        
        gpr_model = InternalGPR()
        
            # gr_x_complete, gr_y_complete, gr_x_new, gr_y_new = pickle.load(f)
        self.assertRaises(TypeError, bo.sample, func1, 50, np.array([in_samples_1]), out_samples_1, region_support, gpr_model, rng)
        self.assertRaises(TypeError, bo.sample, func1, 50, in_samples_1, np.array([out_samples_1]).T, region_support, gpr_model, rng)
        self.assertRaises(TypeError, bo.sample, func1, 50, in_samples_1, out_samples_2, region_support, gpr_model, rng)
        x_complete, y_complete = bo.sample(func1, 50, in_samples_1, out_samples_1, region_support, gpr_model, rng)

        
        with open("tests\\bayesianOptimization\\goldResources\\test_1_bo.pickle", "rb") as f:
            # pickle.dump([x_complete, y_complete], f)
            gr_x_complete, gr_y_complete = pickle.load(f)
        

        np.testing.assert_array_equal(x_complete, gr_x_complete)
        np.testing.assert_array_equal(y_complete, gr_y_complete)
        
if __name__== "__main__":
    unittest.main()