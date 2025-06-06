from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import tqdm
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

from .gpr import GPR, GPRSkeleton
from .sampling import uniform_sampling
from .utils import Fn, compute_robustness


class BOInterface(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """ Initialize BO Method for use in Part-X

        Args:
            bo_model: Bayesian Optimization Class developed with partxv2.byesianOptimization.BO_Interface factory.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self,
        x_train: NDArray[np.float_],
        y_train: NDArray[np.float_],
        region_support: NDArray[np.float_],
        gpr_model: GPRSkeleton,
        rng: np.random.Generator,
        curr_best: NDArray[np.float_] | None) -> NDArray[np.float_]:
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
    
@dataclass(frozen=True)
class InitialPoints:
    x_points: NDArray[np.float_]
    y_points: NDArray[np.float_]

@dataclass(frozen=True)
class InitialPointsSampled:
    x_points: NDArray[np.float_]
    y_points: NDArray[np.float_]


@dataclass(frozen=True)
class SampledPoints:
    x_points: NDArray[np.float_]
    y_points: NDArray[np.float_]

@dataclass
class Result:
    initial_points: InitialPoints
    initial_points_sampled: InitialPointsSampled  # Fixed typo in field name
    sampled_points: SampledPoints

class BOSampling:
    def __init__(self, bo_model: BOInterface) -> None:
        self.bo_model = bo_model

    def sample(
        self,
        test_function: Fn,
        dim: int,
        num_init_samples: int,
        num_bo_samples: int, 
        region_support: NDArray[np.float_],
        gpr_model: GPRSkeleton,
        rng: np.random.Generator,
        x_train: NDArray[np.float_] | None = None,
        y_train: NDArray[np.float_] | None = None,
        curr_best: NDArray[np.float_] | None = None
    ) -> Result:
        dim = region_support.shape[0]
        
        
        if num_bo_samples <= 0:
            raise ValueError("num_bo_samples cannot be zero")

        if num_init_samples <= 0:
            raise ValueError("num_init_samples cannot be zero")
        
        # Initialize containers for different point types
        initial_x = np.empty((0, dim)) if x_train is None else x_train.copy()
        initial_y = np.empty(0) if y_train is None else y_train.copy()
        initial_sampled_x = np.empty((0, dim))
        initial_sampled_y = np.empty(0)
        
        # Handle existing data case
        if x_train is not None and y_train is not None:
            # Validate input shapes
            if len(x_train.shape) != 2 or x_train.shape[1] != dim:
                raise TypeError(f"Expected (n, {dim}) array, got {x_train.shape}")
            if len(y_train.shape) != 1:
                raise TypeError(f"Expected (n,) array, got {y_train.shape}")
            if x_train.shape[0] != y_train.shape[0]:
                raise TypeError("x_train and y_train size mismatch")

            # Calculate needed initial samples
            points_to_sample = max(num_init_samples - x_train.shape[0], 0)
            if points_to_sample > 0:
                initial_sampled_x = uniform_sampling(points_to_sample, region_support, dim, rng)
                initial_sampled_y = compute_robustness(initial_sampled_x, test_function)
                
                # Update training data (for BO process)
                x_train = np.vstack((x_train, initial_sampled_x))
                y_train = np.append(y_train, initial_sampled_y)

            
        else:
            # No existing data - sample all initial points
            initial_sampled_x = uniform_sampling(num_init_samples, region_support, dim, rng)
            initial_sampled_y = compute_robustness(initial_sampled_x, test_function)
            x_train, y_train = initial_sampled_x.copy(), initial_sampled_y.copy()
            

        # Perform BO sampling
        bo_x = np.empty((num_bo_samples, dim))
        bo_y = np.empty(num_bo_samples)
        for i in range(num_bo_samples):
            point = self.bo_model.sample(x_train, y_train, region_support, gpr_model, rng, curr_best)
            y_val = compute_robustness(np.array([point]), test_function)
            # print(f"{point} -> {y_val}")
            x_train = np.vstack((x_train, point))
            y_train = np.append(y_train, y_val)
            bo_x[i] = point
            bo_y[i] = y_val

        return Result(
            initial_points=InitialPoints(initial_x, initial_y),
            initial_points_sampled=InitialPointsSampled(initial_sampled_x, initial_sampled_y),
            sampled_points=SampledPoints(bo_x, bo_y)
        )



class InternalBO(BOInterface):
    def __init__(self) -> None:
        pass

    def sample(
        self,
        x_train: NDArray[np.float_],
        y_train: NDArray[np.float_],
        region_support: NDArray[np.float_],
        gpr_model:  GPRSkeleton,
        rng: np.random.Generator,
        curr_best: NDArray[np.float_]|None
      ) -> NDArray[np.float_]:

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
        self.curr_best = min(np.min(y_train), curr_best) if curr_best is not None else np.min(y_train)
        model = GPR(gpr_model)
        model.fit(x_train, y_train)
        return self._opt_acquisition(y_train, model, region_support, rng)

    def _opt_acquisition(
        self, 
        y_train: NDArray[np.float_], 
        gpr_model: GPR, 
        region_support: NDArray[np.float_], 
        rng: np.random.Generator
    ) -> NDArray[np.float_]:
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

        fun = lambda x_: -1 * self._acquisition(y_train, x_, gpr_model)

        
        
        min_bo_val = -1 * self._acquisition(
            y_train, random_samples, gpr_model, "multiple")

        min_bo = np.array(random_samples[np.argmin(min_bo_val), :])
        
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

        return np.array(min_bo)

    def _surrogate(self, gpr_model: GPR, x_train: NDArray) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
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
        

        
        if sample_type == "multiple":
            mu, std = self._surrogate(gpr_model, sample)
            # mu_con, std_con = constraint_model.predict(sample)
            ei_list = []
            for mu_iter, std_iter, samp in zip(mu, std, sample):
                # if oracle_info(samp).sat:
                pred_var = std_iter

                if pred_var > 0:
                    # con_term = norm.cdf(0, mu_con_iter, std_con_iter)
                    var_1 = self.curr_best - mu_iter
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
                var_1 = self.curr_best - mu[0]
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
