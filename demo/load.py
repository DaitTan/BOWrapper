import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import t


parser = argparse.ArgumentParser()

parser.add_argument("--func", help="display a square of a given number", type = str)
parser.add_argument("--dim", help="display a square of a given number", type = int)
parser.add_argument("--agents", help="display a square of a given number", type = int)
parser.add_argument("--iterations", help="display a square of a given number", type = int)
parser.add_argument("--horizon", help="display a square of a given number", type = int)
parser.add_argument("--n_children", help="display a square of a given number", type = int)

args = parser.parse_args()
bm_name = f"{args.func}_{args.dim}dim_{args.agents}agents_{args.iterations}iter_{args.horizon}horizon_{args.n_children}NC"
benchmarks = [f"{bm_name}_rep{REPL_NUM}" for REPL_NUM in range(1,11)]


def load_initial_data(file_path: Path) -> NDArray[np.float_]:
    with Path.open(file_path, 'r') as f:
        data = json.load(f)
    numpy_array = np.array(data)
    return numpy_array.reshape(-1, 2)
# Load result files to calculate number of initial points

in_samples = load_initial_data(Path.cwd().joinpath("initial_points").joinpath(f"{benchmarks[0]}.json"))
num_initial_points = in_samples.shape[0]

# Load result files
initial_files = [Path.cwd().joinpath("results").joinpath(f"{benchmark}.pkl") for benchmark in benchmarks]

all_running_mins = []
for file_path in initial_files:
    with Path.open(file_path, "rb") as f:
        x_complete, y_complete = pickle.load(f)
        
    running_min = np.minimum.accumulate(y_complete).flatten()
    all_running_mins.append(running_min)
    

all_running_mins = np.array(all_running_mins)   

n = y_complete.shape[0]
x = np.arange(n)  # X-axis values
means = all_running_mins.mean(axis=0)
stds = all_running_mins.std(axis=0)
standard_errors = stds / np.sqrt(all_running_mins.shape[0])

t_critical = t.ppf(0.975, df=all_running_mins.shape[0]-1)
confidence_intervals = t_critical * standard_errors


    # Plot the running minimum as a black line
plt.plot(x, means, color='black', linewidth=1, label='Running Minimum')

plt.fill_between(x[:num_initial_points], 
                 means[:num_initial_points] - confidence_intervals[:num_initial_points], 
                 means[:num_initial_points] + confidence_intervals[:num_initial_points], 
                 color='red', alpha=0.2, 
                 label='95% Confidence Interval - Initial Points')

plt.fill_between(x[num_initial_points:], 
                 means[num_initial_points:] - confidence_intervals[num_initial_points:], 
                 means[num_initial_points:] + confidence_intervals[num_initial_points:], 
                 color='blue', alpha=0.2, 
                 label='95% Confidence Interval - BO Sampled Points')

# Customize the plot
plt.xlabel('Samples')
plt.ylabel('Cost Values')
plt.title(f'{bm_name}')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("test.pdf")
