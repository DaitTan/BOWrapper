

import json
from pathlib import Path
import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng

from bo.bo import BOSampling, InternalBO
from bo.gpr import InternalGPR
from bo.utils import Fn, compute_robustness


def shubert_func(x: np.ndarray)->float:
    if x.shape[0]!=2:
        raise ValueError("Shubert only for dimension=2.")
    s1= sum(i*np.cos((i+1)*x[0]+i) for i in range(1,6))
    s2= sum(i*np.cos((i+1)*x[1]+i) for i in range(1,6))
    return s1*s2

rng = default_rng(12345)

region_support = np.array([[-10, 10], [-10, 10]])
tf_dimension = 2
func1 = Fn(shubert_func)
# in_samples_1 = uniform_sampling(20, region_support, tf_dimension, rng)
# json_data = '''
# [[[-9.64066346617858, -4.671412745111034], [-5.311183261060944, -9.686788135889305], [-8.666602042818553, -1.1603707518528168], [-7.635648965243451, 6.057789921577424], [-6.5096632543450355, 3.1062732784432523]], [[-3.6503251493872413, 2.863660168510954], [-0.1895713599224358, 1.0102386772720244], [-4.193895775363645, 9.478467769503538], [-2.323776444521558, -6.936375734943561], [-1.7203225020934276, -3.858598731349665]], [[4.781072714519915, 3.430736937233828], [2.191595479417666, -8.01409249859589], [3.057138325219711, 1.6702853705459866], [1.350969216284217, -5.386467877826323], [0.006403941117943179, 9.32178913527645]], [[8.002884021075086, -6.6269470601508385], [5.858950102325462, 3.617427054214943], [6.950954709570763, 0.7073469795459779], [7.213109376656437, -5.691827461618364], [9.523352486156917, 6.574239141134754]]]
# '''

for REPL_NUM in range(1,11):
    folder = "./initial_points"
    benchmark = "shubert_2dim_8agents_80iter_2horizon_5NC_rep"
    # file_str = "/shubert_2dim_4agents_80iter_2horizon_5NC_rep"
    with open(Path.cwd().joinpath(folder).joinpath(f"{benchmark}{REPL_NUM}.json"), 'r') as file:
        data = json.load(file)

    # data = json.loads(json_data)
    numpy_array = np.array(data)
    in_samples_1 = numpy_array.reshape(-1, tf_dimension)
    out_samples_1 = compute_robustness(in_samples_1, func1)

    gpr_model = InternalGPR()


    bo = BOSampling(InternalBO())
    num_samples = 640
    x_complete, y_complete = bo.sample(func1, num_samples, in_samples_1, out_samples_1, region_support, gpr_model, rng)
    with open(Path.cwd().joinpath("./results").joinpath(f"{benchmark}{REPL_NUM}.pkl"), "wb") as f:
        pickle.dump((x_complete, y_complete),f)
    
# with open("data_shubert.pkl", "rb") as f:
#     x_complete, y_complete = pickle.load(f)
# # Compute running minimum
# n = y_complete.shape[0]
# running_min = np.minimum.accumulate(y_complete).flatten()
# x = np.arange(n)  # X-axis values

# # Plot the running minimum as a black line
# plt.plot(x, running_min, color='black', linewidth=1, label='Running Minimum')

# # Add red markers at indices 0 and num_samples-1
# red_indices = list(range(0, in_samples_1.shape[0]))
# plt.scatter(
#     x[red_indices], running_min[red_indices],
#     color='red', marker='o', s=10, zorder=5, label='InitialPoints'
# )

# # Add blue markers from num_samples to end
# blue_indices = x[in_samples_1.shape[0]:]
# plt.scatter(
#     blue_indices, running_min[blue_indices],
#     color='blue', marker='o', s=10, zorder=5, label='Sampled BO POints'
# )

# # Customize the plot
# plt.xlabel('Step')
# plt.ylabel('Value')
# plt.title(f'Running Minimum with Markers (Split at Step {num_samples})')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.savefig("test.pdf")

# print(running_min[-1])