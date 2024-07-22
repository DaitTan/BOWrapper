import pickle
import pathlib
import os
import time
import csv
from pathos.multiprocessing import ProcessingPool as Pool

from partxv2.coreAlgorithm import PartXOptions, run_single_replication
from partxv2.results import generate_statistics

def run_partx(BENCHMARK_NAME, test_function, num_macro_reps, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, 
                init_sampling_type, cs_sampling_type, 
                q_estim_sampling, mc_integral_sampling_type, 
                results_sampling_type, 
                results_at_confidence, results_folder_name, num_cores):
    
    
    # create a directory for storing result files
    base_path = pathlib.Path()
    result_directory = base_path.joinpath(results_folder_name)
    result_directory.mkdir(exist_ok=True)
    benchmark_result_directory = result_directory.joinpath(BENCHMARK_NAME)
    benchmark_result_directory.mkdir(exist_ok=True)
    
    benchmark_result_pickle_files = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_result_generating_files")
    benchmark_result_pickle_files.mkdir(exist_ok=True)

    results_csv = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_results_csv")
    results_csv.mkdir(exist_ok=True)
    
    
    # create partx options
    options = PartXOptions(BENCHMARK_NAME, init_reg_sup, tf_dim,
                max_budget, init_budget, bo_budget, cs_budget, 
                alpha, R, M, delta, fv_quantiles_for_gp,
                branching_factor, uniform_partitioning, start_seed, 
                gpr_model, bo_model, 
                init_sampling_type, cs_sampling_type, 
                q_estim_sampling, mc_integral_sampling_type, 
                results_sampling_type)
    
    

    with open(benchmark_result_pickle_files.joinpath(options.BENCHMARK_NAME + "_options.pkl"), "wb") as f:
        pickle.dump(options,f)
    


    # Start running

    inputs = []
    if num_cores == 0:
        raise Exception("Number of cores to use cannot be 0")
    elif num_cores == 1:
        print("Running without parallalization")
        results = []
        for replication_number in range(num_macro_reps):
            data = [replication_number, options, test_function, benchmark_result_directory]
            inputs.append(data)
            res = run_single_replication(data)
            results.append(res)
    elif num_cores != 1:
        num_cores_available = min((os.cpu_count() - 1), num_cores)
        if num_cores == num_cores_available:
            print("Running with {} cores".format(num_cores_available))
        elif num_cores > num_cores_available:
            print("Cannot run with {} cores. Instead running with {} cores.".format(num_cores, num_cores_available))
        elif num_cores < num_cores_available:
            print("Max cores uitilised can be {}. Instead running with {} cores.".format((os.cpu_count() - 1), num_cores_available))
        for replication_number in range(num_macro_reps):
            data = [replication_number, options, test_function, benchmark_result_directory]
            inputs.append(data)
        with Pool(num_cores_available) as pool:
            results = list(pool.map(run_single_replication, inputs))
        pool.close()
    result_dictionary = generate_statistics(options.BENCHMARK_NAME, num_macro_reps, options.fv_quantiles_for_gp, results_at_confidence,results_folder_name)

    today = time.strftime("%m/%d/%Y")
    file_date = today.replace("/","_")
    values = []
    with open(results_csv.joinpath(options.BENCHMARK_NAME+"_"+file_date+ "_results.csv"), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in result_dictionary.items():
            writer.writerow([key, value])
            values.append(value)
    print("Done")
    # result = Result(*values)

    return 1
