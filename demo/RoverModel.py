from staliro.core.interval import Interval
from staliro.core.model import Model, ModelInputs, Trace, ExtraResult
import numpy as np
from numpy.typing import NDArray
try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True

import matplotlib.pyplot as plt

RoverDataT = NDArray[np.float_]
RoverResultT = ExtraResult[RoverDataT, RoverDataT]


class RoverModel(Model[RoverResultT, None]):
    def __init__(self) -> None:
        self.engine = matlab.engine.start_matlab()
        s = self.engine.genpath('matlab_rover')
        self.engine.addpath(s, nargout=0)

    def simulate(
        self, inputs: ModelInputs, intrvl: Interval
    ) -> RoverResultT:
        
        
        
        timestamps, data = self.engine.simulator(inputs.static[0], inputs.static[1], inputs.static[2], nargout=2)
        
        data_py = np.array(data)
        timestamps_py = np.array(timestamps).flatten()
        
        outTrace = Trace(timestamps_py, data_py)
        inTrace = inputs.static
        return RoverResultT(outTrace, inTrace)