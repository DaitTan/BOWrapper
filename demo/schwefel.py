

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng

from bo.core import BOSampling, InternalBO
from bo.gpr import InternalGPR
from bo.utils import Fn, compute_robustness
from scipy.stats import t  

parser = argparse.ArgumentParser()

parser.add_argument("--func", help="display a square of a given number", type = str)
parser.add_argument("--dim", help="display a square of a given number", type = int)
parser.add_argument("--agents", help="display a square of a given number", type = int)
parser.add_argument("--iterations", help="display a square of a given number", type = int)
parser.add_argument("--horizon", help="display a square of a given number", type = int)
parser.add_argument("--n_children", help="display a square of a given number", type = int)
parser.add_argument("--tot_samples", help="display a square of a given number", type = int)
parser.add_argument('--init_test', action='store_true', help='A boolean switch')

args = parser.parse_args()

# def rastrigin_func(x: np.ndarray) -> float:
#     A = 10.0
#     d = x.shape[0]
#     return A*d + np.sum(x**2 - A*np.cos(2*np.pi*x))

# def schwefel_func(x: np.ndarray) -> float:
#     d = x.shape[0]
#     return 418.9829*d - np.sum(x*np.sin(np.sqrt(np.abs(x))))

def langermann_func(x: np.ndarray) -> float:
    m = 4
    c = np.array([1,2,5,2])
    A_fixed = np.array([
        [3,5,2,1, 4,6,2,3,5,7],
        [2,4,1,3, 2,5,4,2,6,3],
        [5,3,2,4, 6,1,3,5,2,4],
        [1,2,3,4, 5,6,7,8,9,10]
    ])
    d = x.shape[0]
    if d <= 10:
        A = A_fixed[:, :d]
    else:
        reps = int(np.ceil(d/10))
        A_tile = np.tile(A_fixed, (1, reps))
        A = A_tile[:, :d]
    val = 0.0
    for i in range(m):
        norm_sq = np.sum((x - A[i])**2)
        val += c[i] * np.exp(-norm_sq/np.pi) * np.cos(np.pi*norm_sq)
    return val

def shubert_func(x: np.ndarray)->float:
    if x.shape[0]!=2:
        raise ValueError("Shubert only for dimension=2.")
    s1= sum(i*np.cos((i+1)*x[0]+i) for i in range(1,6))
    s2= sum(i*np.cos((i+1)*x[1]+i) for i in range(1,6))
    return s1*s2

tf_dimension = args.dim
# region_support = np.array([[-2.5,3] for _ in range(args.dim)])
region_support = np.array([[0,10] for _ in range(args.dim)])
# region_support = np.array([[-500,500] for _ in range(args.dim)])
func1 = Fn(langermann_func)

rng = default_rng(12345)

all_running_mins = []

for REPL_NUM in range(1,11):
    folder = "./initial_points"
    benchmark = f"{args.func}_{args.dim}dim_{args.agents}agents_{args.iterations}iter_{args.horizon}horizon_{args.n_children}NC_rep"

    with open(Path.cwd().joinpath(folder).joinpath(f"{benchmark}{REPL_NUM}.json"), 'r') as file:
        data = json.load(file)

    numpy_array = np.array(data)
    in_samples_1 = numpy_array.reshape(-1, tf_dimension)
    out_samples_1 = compute_robustness(in_samples_1, func1)
    num_samples = args.tot_samples-in_samples_1.shape[0]
    if args.init_test:
        print(f"Received JSON Input: {numpy_array.shape}")
        print(f"Initial Samples Size: {in_samples_1.shape}")
        print(f"Initial Samples Fn array size: {out_samples_1.shape}")
        print(f"Initial Samples = {in_samples_1.shape[0]}\nNum BO Iterations = {num_samples}\ntotal_budget = {in_samples_1.shape[0]+num_samples}")
        print(f"Dimensionality is {args.dim}")
        print(f"Region Support is {region_support}")
        print(f"Did you check for region support?")
        print("***********************************")
        continue

    gpr_model = InternalGPR()


    bo = BOSampling(InternalBO())
    
    x_complete, y_complete = bo.sample(func1, num_samples, in_samples_1, out_samples_1, region_support, gpr_model, rng)
    with open(Path.cwd().joinpath("./results").joinpath(f"{benchmark}{REPL_NUM}.pkl"), "wb") as f:
        pickle.dump((x_complete, y_complete),f)
    
    # with open(Path.cwd().joinpath("./results").joinpath(f"{benchmark}{REPL_NUM}.pkl"), "rb") as f:
    #     x_complete, y_complete = pickle.load(f)
    # # Compute running minimum
    
    # running_min = np.minimum.accumulate(y_complete).flatten()
    # print(running_min[-1])
    # all_running_mins.append(running_min)
    
# all_running_mins = np.array(all_running_mins)   

# print(all_running_mins.shape)
 
# n = y_complete.shape[0]
# x = np.arange(n)  # X-axis values
# means = all_running_mins.mean(axis=0)
# stds = all_running_mins.std(axis=0)
# standard_errors = stds / np.sqrt(all_running_mins.shape[0])

# t_critical = t.ppf(0.975, df=all_running_mins.shape[0]-1)
# confidence_intervals = t_critical * standard_errors


#     # Plot the running minimum as a black line
# plt.plot(x, means, color='black', linewidth=1, label='Running Minimum')

# plt.fill_between(x, 
#                  means - confidence_intervals, 
#                  means + confidence_intervals, 
#                  color='blue', alpha=0.2, 
#                  label='95% Confidence Interval')

# # Add red markers at indices 0 and num_samples-1
# # red_indices = list(range(0, in_samples_1.shape[0]))
# # plt.scatter(
# #     x[red_indices], running_min[red_indices],
# #     color='red', marker='o', s=10, zorder=5, label='InitialPoints'
# # )

# # # Add blue markers from num_samples to end
# # blue_indices = x[in_samples_1.shape[0]:]
# # plt.scatter(
# #     blue_indices, running_min[blue_indices],
# #     color='blue', marker='o', s=10, zorder=5, label='Sampled BO POints'
# # )

# # Customize the plot
# plt.xlabel('Samples')
# plt.ylabel('Cost Values')
# plt.title(f'{benchmark}')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.savefig("test.pdf")

# print(running_min[-1])
# print(agg)