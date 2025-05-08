from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
import tqdm
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

from .gpr import GPR, GPRSkeleton
from .sampling import uniform_sampling
from .utils import Fn, compute_robustness


class BO_Interface(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """ Initialize BO Method for use in Part-X

        Args:
            bo_model: Bayesian Optimization Class developed with partxv2.byesianOptimization.BO_Interface factory.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, test_function: Fn,
        num_samples: int,
        x_train: NDArray,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: GPRSkeleton,
        rng: np.random.Generator) -> NDArray:
        """Sampling using User Defined BO.

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

        """
        raise NotImplementedError
    
class BOSampling:
    def __init__(self, bo_model: BO_Interface) -> None:
        """ Initialize BO Method for use in Part-X

        Args:
            bo_model: Bayesian Optimization Class developed with partxv2.byesianOptimization.BO_Interface factory.
        """
        self.bo_model = bo_model

    def sample(
        self,
        test_function: Fn,
        num_samples: int,
        x_train: NDArray,
        y_train: NDArray,
        region_support: NDArray,
        gpr_model: GPRSkeleton,
        rng: np.random.Generator,
    ) -> tuple: 
        """Wrapper around user defined BO Model.

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

        dim = region_support.shape[0]
        if len(x_train.shape) != 2 or x_train.shape[1] != dim:
            raise TypeError(f"Received samples set input: Expected (n, dim) array, received {x_train.shape} instead.")
        if len(y_train.shape) != 1:
            raise TypeError(f"Received evaluations set input: Expected (n,) array, received {y_train.shape} instead.")
        if x_train.shape[0] != y_train.shape[0]:
            raise TypeError(f"x_train, y_train set mismatch. x_train has shape {x_train.shape} and y_train has shape {y_train.shape}")


        x_new = []
        y_new = []
        best_pt = np.min(y_train)
        for _ in tqdm.tqdm(range(num_samples)):
            # print(len(x_new))
            # print(n_tries)
            point = self.bo_model.sample(
                x_train, y_train, region_support, gpr_model, rng
            )
            x_new.append(point)
            pred_sample_y = compute_robustness(np.array([point]), test_function)
            x_train = np.vstack((x_train, np.array([point])))
            y_train = np.hstack((y_train, pred_sample_y))
            # best_pt = min(best_pt, pred_sample_y)
            # print(best_pt)
        # x_new = np.array(x_new)
        # y_new = np.array(y_new)
        # print(x_new)
        # print(x_new.shape)
        
        assert len(x_train.shape) == 2, f"Returned merged samples set input: Expected (n, dim) array, returned {x_train.shape} instead."
        assert len(y_train.shape) == 1, f"Returned merged evaluations set input: Expected (n, ) array, returned {y_train.shape} instead."

        return x_train, y_train


# local_oracle = None


class InternalBO(BO_Interface):
    def __init__(self) -> None:
        pass

    def sample(
         self,
         x_train: NDArray,
         y_train: NDArray,
         region_support: NDArray,
         gpr_model:  GPRSkeleton,
         rng: np.random.Generator,
      ) -> NDArray:

        """Internal BO Model

        Args:
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.
            region_support: Min and Max of all dimensions
            gpr_model: Gaussian Process Regressor Model developed using Factory
            oracle_info: Oracle defining the constraints.
            rng: RNG object from numpy

        Raises:
            TypeError: If x_train is not 2 dimensional numpy array or does not match dimensions
            TypeError: If y_train is not (n,) numpy array
            TypeError: If there is a mismatch between x_train and y_train

        Returns:
            x_new
         """
        
        model = GPR(gpr_model)
        model.fit(x_train, y_train)

        pred_sample_x = self._opt_acquisition(y_train, model, region_support, rng)


        return pred_sample_x

    def _opt_acquisition(self, y_train: NDArray, gpr_model: GPR, region_support: NDArray, rng: np.random.Generator) -> NDArray:
        """Get the sample points

        Args:
            X: sample points
            y: corresponding robustness values
            model: the GP models
            sbo: sample points to construct the robustness values
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)
            region_support: The bounds of the region within which the sampling is to be done.
                                        Region Bounds is M x N x O where;
                                            M = number of regions;
                                            N = test_function_dimension (Dimensionality of the test function);
                                            O = Lower and Upper bound. Should be of length 2;

        Returns:
            The new sample points by BO
        """

        tf_dim = region_support.shape[0]
        lower_bound_theta = np.ndarray.flatten(region_support[:, 0])
        upper_bound_theta = np.ndarray.flatten(region_support[:, 1])

        random_samples = uniform_sampling(5000, region_support, tf_dim, rng)
        
        curr_best = np.min(y_train)
        # constraints_out = np.array([oracle_info(x).val for x in random_samples])
        # [print(oracle_info(x).val, oracle_info(x).sat) for x in random_samples]
        # if oracle_info.oracle_function is not None:
        #     constraint_model.fit(random_samples, constraints_out)

        # bnds = Bounds(lower_bound_theta, upper_bound_theta)
        fun = lambda x_: -1 * self._acquisition(y_train, x_, gpr_model)

        
        
        min_bo_val = -1 * self._acquisition(
            y_train, random_samples, gpr_model, "multiple")

        min_bo = np.array(random_samples[np.argmin(min_bo_val), :])
        # import matplotlib.pyplot as plt
        # plt.plot(random_samples, min_bo_val, ".")
        # plt.show()
        min_bo_val = np.min(min_bo_val)

        for _ in range(9):
            new_params = minimize(
                fun,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                x0=min_bo,
            )

            if not new_params.success:
                continue

            if min_bo is None or fun(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = fun(min_bo)
        new_params = minimize(
            fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        min_bo = new_params.x
        # penalty = oracle_info(np.array(min_bo)).val

        return np.array(min_bo)

    def _surrogate(self, gpr_model: GPR, x_train: NDArray):
        """_surrogate Model function

        Args:
            model: Gaussian process model
            X: Input points

        Returns:
            Predicted values of points using gaussian process model
        """

        return gpr_model.predict(x_train)

    def _acquisition(self, y_train: NDArray, sample: NDArray, gpr_model: GPR, sample_type:str ="single") -> NDArray|float:
        """Acquisition Model: Expected Improvement

        Args:
            y_train: corresponding robustness values
            sample: Sample(s) whose EI is to be calculated
            gpr_model: GPR model
            sample_type: Single sample or list of model. Defaults to "single". other options is "multiple".

        Returns:
            EI of samples
        """
        curr_best = np.min(y_train)

        
        if sample_type == "multiple":
            mu, std = self._surrogate(gpr_model, sample)
            # mu_con, std_con = constraint_model.predict(sample)
            ei_list = []
            for mu_iter, std_iter, samp in zip(mu, std, sample):
                # if oracle_info(samp).sat:
                pred_var = std_iter

                if pred_var > 0:
                    # con_term = norm.cdf(0, mu_con_iter, std_con_iter)
                    var_1 = curr_best - mu_iter
                    var_2 = var_1 / pred_var

                    ei = (var_1 * norm.cdf(var_2)) + (
                        pred_var * norm.pdf(var_2)
                    ) 
                else:
                    ei = 0.0
                # else:
                #     ei = -999
                ei_list.append(ei)
            
        elif sample_type == "single":
                # if oracle_info(sample).sat:
            mu, std = self._surrogate(gpr_model, sample.reshape(1, -1))
            # mu_con, std_con = constraint_model.predict(np.array([sample]))
            pred_var = std[0]
            if pred_var > 0:
                # con_term = norm.cdf(0,mu_con[0], std_con[0])
                var_1 = curr_best - mu[0]
                var_2 = var_1 / pred_var

                ei = (var_1 * norm.cdf(var_2)) + (
                    pred_var * norm.pdf(var_2)
                )
            else:
                ei = 0.0
            # else:
            #     ei = -999
            # return ei

        if sample_type == "multiple":
            return_ei = np.array(ei_list)
        elif sample_type == "single":
            return_ei = ei

        return return_ei
