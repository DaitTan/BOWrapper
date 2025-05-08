import numpy as np
from numpy.typing import NDArray

from .function import Fn


def compute_robustness(samples_in: NDArray[np.float_], test_function: Fn) -> NDArray[np.float_]:
    """Compute the fitness (robustness) of the given sample.

    Args:
        samples_in: Samples points for which the fitness is to be computed.
        test_function: Test Function insitialized with Fn
    Returns:
        Fitness (robustness) of the given sample(s)
    """
    samples_out = np.empty(samples_in.shape[0])
    
    if samples_in.shape[0] == 1:
        samples_out[0] = test_function(samples_in[0])
        return samples_out
    
    for iterate, sample in enumerate(samples_in):
        samples_out[iterate] = test_function(sample)
    return samples_out
