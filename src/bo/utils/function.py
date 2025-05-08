import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class Fn:
    def __init__(self, func: Callable[[NDArray[np.float_]], float]):
        self.func = func
        self.count: int = 0
        self.point_history: list[tuple[int, NDArray[np.float_],float]] = []
        self.simultation_time: list[float] = []

    def __call__(self, input_vec: NDArray[np.float_]) -> float:
        self.count += 1
        sim_time_start = time.perf_counter()
        rob_val = self.func(input_vec)
        time_elapsed = time.perf_counter() - sim_time_start
        self.simultation_time.append(time_elapsed)
        self.point_history.append((self.count, input_vec, rob_val))
        return rob_val
