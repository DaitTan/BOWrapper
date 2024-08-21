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

DroneDataT = NDArray[np.float_]
DroneResultT = ExtraResult[DroneDataT, DroneDataT]


class DroneModel(Model[DroneResultT, None]):
    def __init__(self, MODE) -> None:
        self.engine = matlab.engine.start_matlab()
        self.engine.cd(r'/home/local/ASURITE/tkhandai/Research_Work/BOWrapper/demo', nargout=0)
        self.MODE = MODE

    def simulate(
        self, inputs: ModelInputs, intrvl: Interval
    ) -> DroneResultT:
        
        # inputs.static
        if self.MODE == 1:
            timestamps, data = self.engine.IntegrationPsiTaliro2(inputs.static[0], inputs.static[1], inputs.static[2], nargout=2)
        elif self.MODE == 2:
            timestamps, data = self.engine.IntegrationPsiTaliroEMA(inputs.static[0], inputs.static[1], inputs.static[2], nargout=2)
        elif self.MODE == 3:
            timestamps, data = self.engine.IntegrationPsiTaliroEMA_HA(inputs.static[0], inputs.static[1], inputs.static[2], nargout=2)
        print(inputs.static)
        data_py = np.array(data).T
        timestamps_py = np.array(timestamps).flatten()
        timestamps_py = timestamps_py[:data_py.shape[0]]
        

        
        
        # plt.plot(timestamps_py, data_py[:,2], "-", label = "Alt")
        # plt.plot(timestamps_py, data_py[:,5], "-", label = "Speed")
        # plt.legend()
        # plt.savefig("trace.pdf")
        # print(min(data_py))

        # timestamps = np.array(result["times"], dtype=(np.float32))
        outTrace = Trace(timestamps_py, data_py)
        inTrace = inputs.static
        return DroneResultT(outTrace, inTrace)