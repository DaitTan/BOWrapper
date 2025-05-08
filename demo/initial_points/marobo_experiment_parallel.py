import os
import numpy as np
import logging
import argparse
import time
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

from scipy.stats import qmc
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import json

os.makedirs("outputs", exist_ok=True)

# ---------- Default Hyperparameters -----------
EI_THRESHOLD = 0.0    
NO_SPLIT_PROB = 0.1   
LOCAL_ACQ_PER_ITER = 2
HORIZON = 3
NC = 10
N_RO = 1

SEARCH_MIN, SEARCH_MAX = -5.0, 5.0
GLOBAL_MIN_LOC = None
GLOBAL_MIN_VAL = None
TEST_FUNC: Callable = None
TEST_FUNC_NAME = ""
DIMENSION = 2

def setup_logger(func: str, dim: int, m: int, iterations: int, h: int, nc: int, rep: int) -> logging.Logger:
    filename = f"{func}_{dim}dim_{m}agents_{iterations}iter_{h}horizon_{nc}NC_rep{rep}.log"
    path = os.path.join("outputs", filename)
    logger = logging.getLogger("ma_logger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(path, mode="w")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def rastrigin_func(x: np.ndarray) -> float:
    A = 10.0
    d = x.shape[0]
    return A*d + np.sum(x**2 - A*np.cos(2*np.pi*x))

def schwefel_func(x: np.ndarray) -> float:
    d = x.shape[0]
    return 418.9829*d - np.sum(x*np.sin(np.sqrt(np.abs(x))))

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

def shubert_func(x: np.ndarray) -> float:
    if x.shape[0] != 2:
        raise ValueError("Shubert only for dimension=2.")
    s1 = sum(i*np.cos((i+1)*x[0] + i) for i in range(1,6))
    s2 = sum(i*np.cos((i+1)*x[1] + i) for i in range(1,6))
    return s1 * s2

def set_test_function_choice(func_name: str, dimension: int, logger: logging.Logger):
    global TEST_FUNC_NAME, DIMENSION, SEARCH_MIN, SEARCH_MAX, GLOBAL_MIN_LOC, GLOBAL_MIN_VAL, TEST_FUNC
    TEST_FUNC_NAME = func_name.lower()
    DIMENSION = dimension
    if TEST_FUNC_NAME == "shubert":
        if dimension != 2:
            raise ValueError("Shubert only for dimension=2.")
        SEARCH_MIN, SEARCH_MAX = -10.0, 10.0
        GLOBAL_MIN_LOC = np.array([-7.0835, 4.8580])
        GLOBAL_MIN_VAL = -186.7309
        TEST_FUNC = shubert_func
    elif TEST_FUNC_NAME == "rastrigin":
        if dimension == 2:
            SEARCH_MIN, SEARCH_MAX = -5.0, 5.0
        else:
            SEARCH_MIN, SEARCH_MAX = -2.5, 3.0
        GLOBAL_MIN_LOC = np.zeros(dimension)
        GLOBAL_MIN_VAL = 0.0
        TEST_FUNC = rastrigin_func
    elif TEST_FUNC_NAME == "schwefel":
        SEARCH_MIN, SEARCH_MAX = -500.0, 500.0
        GLOBAL_MIN_LOC = np.full(dimension, 420.9687)
        GLOBAL_MIN_VAL = 0.0
        TEST_FUNC = schwefel_func
    elif TEST_FUNC_NAME == "langermann":
        SEARCH_MIN, SEARCH_MAX = 0.0, 10.0
        GLOBAL_MIN_LOC = np.full(dimension, 2.0)
        GLOBAL_MIN_VAL = -5.16
        TEST_FUNC = langermann_func
    else:
        raise ValueError(f"Unsupported function {func_name}")
    logger.info(f"Set test function={TEST_FUNC_NAME}, domain=({SEARCH_MIN},{SEARCH_MAX}), dim={dimension}")

def evaluate_function(x: np.ndarray) -> float:
    return TEST_FUNC(x)

@dataclass
class SimpleGP:
    X: np.ndarray
    y: np.ndarray
    random_state: int = 0

    def __post_init__(self):
        self.gp = None
        self.X_scaler = None
        self.y_scaler = None

    def fit(self):
        if self.X.size == 0:
            return
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_s = self.X_scaler.fit_transform(self.X)
        y_s = self.y_scaler.fit_transform(self.y.reshape(-1,1)).ravel()
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )
        self.gp.fit(X_s, y_s)

    def predict(self, Xtest: np.ndarray):
        if self.gp is None or self.X.size == 0:
            return np.zeros(Xtest.shape[0]), np.full(Xtest.shape[0], 1e4)
        Xtest_s = self.X_scaler.transform(Xtest)
        mu_s, std_s = self.gp.predict(Xtest_s, return_std=True)
        mu = self.y_scaler.inverse_transform(mu_s.reshape(-1,1)).ravel()
        scale_y = self.y_scaler.scale_[0]
        std = std_s * scale_y
        return mu, std

def expected_improvement(Xcand: np.ndarray, gp: SimpleGP, f_star: float):
    if gp.X.size == 0:
        return np.full(Xcand.shape[0], 1e4)
    mu, std = gp.predict(Xcand)
    eis = []
    for m_val, s_val in zip(mu, std):
        if s_val < 1e-12:
            eis.append(0.0)
        else:
            diff = f_star - m_val
            Z = diff / s_val
            cdf = 0.5 * (1 + np.math.erf(Z / np.sqrt(2)))
            pdf = 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * (Z**2))
            ei = diff*cdf + s_val*pdf
            eis.append(max(ei, 0))
    return np.array(eis)

def point_in_region(point: np.ndarray, region: List[Tuple[float, float]]) -> bool:
    for i, (low, high) in enumerate(region):
        if not (low <= point[i] <= high):
            return False
    return True

def in_any_subregion(point: np.ndarray, subregions: List[List[Tuple[float, float]]]) -> bool:
    return any(point_in_region(point, reg) for reg in subregions)

def filter_data_to_subregions(X_data: np.ndarray, y_data: np.ndarray, subregions: List[List[Tuple[float, float]]]):
    if X_data.size == 0 or len(subregions) == 0:
        return np.empty((0, DIMENSION)), np.empty((0,))
    mask = [in_any_subregion(X_data[i], subregions) for i in range(X_data.shape[0])]
    return X_data[np.array(mask)], y_data[np.array(mask)]

def dimension_cut_weighted(region: List[Tuple[float, float]], agent_X: np.ndarray, rng: np.random.Generator):
    variances = np.var(agent_X, axis=0)
    if np.sum(variances) < 1e-12:
        d = rng.integers(DIMENSION)
    else:
        p = variances / np.sum(variances)
        d = rng.choice(np.arange(DIMENSION), p=p)
    low, high = region[d]
    if high <= low:
        return (region, None)
    cut = rng.uniform(low, high)
    left = list(region)
    right = list(region)
    left[d] = (low, cut)
    right[d] = (cut, high)
    return (left, right)

def bo_sample_region(X_data: np.ndarray, y_data: np.ndarray,
                     region: List[Tuple[float,float]], rng: np.random.Generator,
                     f_star: float, n_candidates=50, sample_logs=None):
    start_time = time.time()
    gp = SimpleGP(X_data, y_data, random_state=rng.integers(1e9))
    gp.fit()
    lowers = [b[0] for b in region]
    uppers = [b[1] for b in region]
    if any(uppers[d] <= lowers[d] for d in range(DIMENSION)):
        return X_data, y_data
    Xcand = rng.uniform(lowers, uppers, size=(n_candidates, DIMENSION))
    eis = expected_improvement(Xcand, gp, f_star)
    idx = np.argmax(eis)
    best_xy = Xcand[idx]
    val = evaluate_function(best_xy)
    X_data = np.vstack((X_data, best_xy))
    y_data = np.hstack((y_data, val))
    end_time = time.time()
    if sample_logs is not None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sample_logs.append({
            "point": best_xy.tolist(),
            "subregion": region,
            "f_value": float(val),
            "time": end_time - start_time,
            "timestamp": ts
        })
    return X_data, y_data

@dataclass
class AgentState:
    subregions: List[List[Tuple[float,float]]]  
    best_val: float
    X_data: np.ndarray  
    y_data: np.ndarray  
    xy_hist: List[Tuple[float,float]]
    active: bool

class MAROBOConfig:
    def __init__(self, agents: List[AgentState]):
        self.agents = agents
    def copy(self):
        new_agents = []
        for ag in self.agents:
            new_agents.append(AgentState(
                subregions=list(ag.subregions),
                best_val=ag.best_val,
                X_data=ag.X_data.copy(),
                y_data=ag.y_data.copy(),
                xy_hist=list(ag.xy_hist),
                active=ag.active
            ))
        return MAROBOConfig(new_agents)
    def global_best(self):
        vals = []
        for ag in self.agents:
            if ag.y_data.size>0:
                vals.append(np.min(ag.y_data))
        return float(np.min(vals)) if vals else float("inf")

def sequential_update(cfg: MAROBOConfig, rng: np.random.Generator, logger: logging.Logger,
                      f_star: float, sample_logs=None) -> MAROBOConfig:
    m = len(cfg.agents)
    eivals = []
    for i, ag in enumerate(cfg.agents):
        ag.X_data, ag.y_data = filter_data_to_subregions(ag.X_data, ag.y_data, ag.subregions)
        gp = SimpleGP(ag.X_data, ag.y_data, random_state=rng.integers(1e9))
        gp.fit()
        local_max = 0.0
        for reg in ag.subregions:
            lowers = [b[0] for b in reg]
            uppers = [b[1] for b in reg]
            if any(uppers[d] <= lowers[d] for d in range(DIMENSION)):
                continue
            Xc = rng.uniform(lowers, uppers, size=(30, DIMENSION))
            local_max = max(local_max, float(np.max(expected_improvement(Xc, gp, f_star))))
        eivals.append(local_max)
    sum_ei = sum(eivals)
    if sum_ei<1e-12:
        return cfg

    for i in range(m):
        p_jump = eivals[i]/(sum_ei+1e-12)
        if i==m-1:
            p_jump*=0.5
        if rng.random()< p_jump:
            cands = []
            for j in range(i+1,m):
                agj = cfg.agents[j]
                if len(agj.subregions)==0:
                    cands.append((j, None, [(SEARCH_MIN, SEARCH_MAX)]*DIMENSION))
                else:
                    for s_idx, s_reg in enumerate(agj.subregions):
                        cands.append((j, s_idx, s_reg))
            if cands:
                gp_i = SimpleGP(cfg.agents[i].X_data, cfg.agents[i].y_data, random_state=rng.integers(1e9))
                gp_i.fit()
                c_ei = []
                for (tj, ts_idx, s_reg) in cands:
                    lowers = [b[0] for b in s_reg]
                    uppers = [b[1] for b in s_reg]
                    if any(uppers[d] <= lowers[d] for d in range(DIMENSION)):
                        c_ei.append(0.0)
                        continue
                    Xcand = rng.uniform(lowers, uppers, size=(30, DIMENSION))
                    c_ei.append(float(np.max(expected_improvement(Xcand, gp_i, f_star))))
                sum_c = sum(c_ei)
                if sum_c>1e-12:
                    pvals = np.array(c_ei)/sum_c
                    r_ = rng.random()
                    cums = np.cumsum(pvals)
                    idx_ = np.searchsorted(cums, r_)
                    idx_ = min(idx_, len(cands)-1)
                    chosen_j, chosen_sidx, chosen_reg = cands[idx_]
                    if rng.random()< NO_SPLIT_PROB:
                        pass
                    else:
                        (halfA, halfB) = dimension_cut_weighted(chosen_reg, cfg.agents[i].X_data, rng)
                        if halfB is not None:
                            agent_j = cfg.agents[chosen_j]
                            if chosen_sidx is not None and chosen_sidx<len(agent_j.subregions):
                                agent_j.subregions.pop(chosen_sidx)
                            def regionEI(r_):
                                lowers_ = [b[0] for b in r_]
                                uppers_ = [b[1] for b in r_]
                                if any(uppers_[dd] <= lowers_[dd] for dd in range(DIMENSION)):
                                    return 0.0
                                X_ = rng.uniform(lowers_, uppers_, size=(30, DIMENSION))
                                return float(np.max(expected_improvement(X_, gp_i, f_star)))
                            eiA = regionEI(halfA)
                            eiB = regionEI(halfB)
                            if eiA>=eiB:
                                kept, other = halfA, halfB
                            else:
                                kept, other = halfB, halfA
                            cfg.agents[i].subregions.append(kept)
                            agent_j.subregions.append(other)
        # local sample
        agent_i = cfg.agents[i]
        agent_i.X_data, agent_i.y_data = filter_data_to_subregions(agent_i.X_data, agent_i.y_data, agent_i.subregions)
        gp_local = SimpleGP(agent_i.X_data, agent_i.y_data, random_state=rng.integers(1e9))
        gp_local.fit()
        best_reg, best_ei = None, -1
        for reg_ in agent_i.subregions:
            lowers_ = [b[0] for b in reg_]
            uppers_ = [b[1] for b in reg_]
            if any(uppers_[dd] <= lowers_[dd] for dd in range(DIMENSION)):
                continue
            X_ = rng.uniform(lowers_, uppers_, size=(30, DIMENSION))
            e_val = float(np.max(expected_improvement(X_, gp_local, f_star)))
            if e_val > best_ei:
                best_ei = e_val
                best_reg = reg_
        if best_reg is not None:
            agent_i.X_data, agent_i.y_data = bo_sample_region(
                agent_i.X_data, agent_i.y_data, best_reg, rng, f_star,
                n_candidates=50, sample_logs=sample_logs
            )
            if agent_i.y_data.size>0:
                agent_i.best_val = min(agent_i.best_val, np.min(agent_i.y_data))
    return cfg

def create_child_config(cfg, seed, logger, f_star):
    local_rng= np.random.default_rng(seed)
    child = cfg.copy()
    child= sequential_update(child, local_rng, logger, f_star)
    return child

def generate_candidate_configurations(cfg: MAROBOConfig, rng: np.random.Generator, logger: logging.Logger,
                                      f_star: float, num_candidates=NC):
    seeds= rng.integers(1e9, size=num_candidates)
    children= Parallel(n_jobs=-1)(
        delayed(create_child_config)(cfg.copy(), int(seeds[i]), logger, f_star)
        for i in range(num_candidates)
    )
    return children

def base_heuristic_update(cfg: MAROBOConfig, rng: np.random.Generator, logger: logging.Logger,
                          f_star: float, steps=HORIZON):
    new_cfg = cfg.copy()
    for _ in range(steps):
        new_cfg = sequential_update(new_cfg, rng, logger, f_star)
    return new_cfg

def rollout_cost(cfg: MAROBOConfig, rng: np.random.Generator, logger: logging.Logger,
                 f_star: float, steps=HORIZON, rep=N_RO):
    costs = []
    for _ in range(rep):
        c_temp = base_heuristic_update(cfg, rng, logger, f_star, steps)
        costs.append(c_temp.global_best())
    return float(np.mean(costs))

def save_samples_to_file(samples, seed, filepath):
    with open(filepath, 'w') as f:
        json.dump(samples.tolist(), f)


class MultiAgentPOCA:
    def __init__(self, m=4, total_iterations=30, horizon=HORIZON, n_children=NC,
                 logger=None, rep=1, seed=42):
        self.m = m
        self.total_iterations = total_iterations
        self.horizon = horizon
        self.n_children = n_children
        self.rep = rep
        self.seed = seed
        self.total_samples = 0

        if logger is None:
            logger = logging.getLogger("ma_logger")
            logger.setLevel(logging.INFO)
        self.logger = logger

        self.iter_logs = []
        self.sample_logs = []
        self.best_vs_iter = []       # best across all agents each iteration
        self.best_vs_samples = []    # (total_samples, global best)
        self.agent_best_trace = []   # agent_best_trace[i][it] = best_val for agent i at iteration it
        self.global_best_trace = []  # global best at each iteration

        self.sample_file = f"{TEST_FUNC_NAME}_{DIMENSION}dim_{self.m}agents_{self.total_iterations}iter_{self.horizon}horizon_{self.n_children}NC_rep{self.rep}.json"

        rng = np.random.default_rng(self.seed)
        if self.m > 1:
            self.width = (SEARCH_MAX - SEARCH_MIN) / self.m
        else:
            self.width = (SEARCH_MAX - SEARCH_MIN)

        self.n0_agent = max(1, (10 * DIMENSION) // self.m)

        # Attempt to load preexisting samples for consistency:
        if os.path.exists(self.sample_file):
            self.logger.info(f"Loading initial samples from {self.sample_file}...")
            loaded_list = json.load(open(self.sample_file, "r"))
            initial_samples = np.array([np.array(item, dtype=float) for item in loaded_list])
            if initial_samples.shape[0] != self.m:
                self.logger.info(f"Mismatch in loaded sample count (got {initial_samples.shape[0]}, expected {self.m}). Regenerating.")
                initial_samples = self.generate_initial_samples(rng)
                save_samples_to_file(initial_samples, self.seed, self.sample_file)
        else:
            self.logger.info(f"Generating new initial samples for seed={self.seed}, m={self.m} ...")
            initial_samples = self.generate_initial_samples(rng)
            save_samples_to_file(initial_samples, self.seed, self.sample_file)

        # Create agent states
        agents = self.create_agents(initial_samples)
        self.current_cfg = MAROBOConfig(agents)

        # Prepare agent best arrays
        for _ in range(self.m):
            self.agent_best_trace.append([])

        self.start_time = time.time()
        self.logger.info(f"Initialize MAROBO: dimension={DIMENSION}, #agents={m}, iteration={self.total_iterations}, "
                         f"horizon={self.horizon}, n_children={self.n_children}, rep={self.rep}, seed={self.seed}")
        self.logger.info(f"Domain=({SEARCH_MIN},{SEARCH_MAX}), initial #samples per agent={self.n0_agent}")

    def generate_initial_samples(self, rng):
        """
        Produce shape (m, n0_agent, DIMENSION) array of initial LHS samples 
        stored as a python list-of-lists for JSON.
        """
        all_samples = []
        for i in range(self.m):
            x0_min = SEARCH_MIN + i * self.width
            x0_max = SEARCH_MIN + (i+1) * self.width
            sampler = qmc.LatinHypercube(d=DIMENSION)
            sample = sampler.random(n=self.n0_agent)
            lower_bounds = [x0_min] + [SEARCH_MIN]*(DIMENSION-1)
            upper_bounds = [x0_max] + [SEARCH_MAX]*(DIMENSION-1)
            scaled = qmc.scale(sample, lower_bounds, upper_bounds)
            all_samples.append(scaled.tolist())
        return np.array(all_samples, dtype=object)

    def create_agents(self, initial_samples):
        """
        Build AgentState objects with subregion and data in ND.
        subreg = [ (xmin,xmax), (ymin,ymax), ... ] up to DIMENSION dims
        For i'th agent, we modify the first dimension from x0_min to x0_max, 
        but keep [SEARCH_MIN, SEARCH_MAX] for other dims.
        """
        agents = []
        for i in range(self.m):
            # subregion for agent i
            x0_min = SEARCH_MIN + i * self.width
            x0_max = SEARCH_MIN + (i+1) * self.width
            # for dimension=1 => (x0_min, x0_max)
            # for dimension>1 => first dimension is that, others are (SEARCH_MIN, SEARCH_MAX)
            subreg = [(x0_min, x0_max)] + [(SEARCH_MIN, SEARCH_MAX)]*(DIMENSION-1)
            scaled = np.array(initial_samples[i], dtype=float)  # shape (n0_agent, DIMENSION)
            vals = []
            for pt in scaled:
                t0 = time.time()
                val = evaluate_function(pt)
                t1 = time.time()
                self.sample_logs.append({
                    "point": pt.tolist(),
                    "subregion": subreg,
                    "f_value": float(val),
                    "time": t1 - t0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                vals.append(val)
                self.total_samples += 1
            vals = np.array(vals)
            bestv = float(np.min(vals)) if len(vals)>0 else float("inf")
            agents.append(AgentState(
                subregions=[subreg],
                best_val=bestv,
                X_data=scaled,
                y_data=vals,
                xy_hist=[],
                active=True
            ))
        return agents

    def run(self):
        rng = np.random.default_rng()
        for it in range(1, self.total_iterations+1):
            it_t0 = time.time()
            f_star = self.current_cfg.global_best()
            if f_star == float("inf"):
                f_star = 1e4

            # generate Nc children configs in parallel
            children = generate_candidate_configurations(self.current_cfg, rng, self.logger, f_star, self.n_children)
            costs = []
            for child in children:
                c_val = rollout_cost(child, rng, self.logger, f_star, steps=self.horizon, rep=N_RO)
                costs.append(c_val)
            best_idx = int(np.argmin(costs))
            self.current_cfg = children[best_idx]

            # local acquisitions
            for i, ag in enumerate(self.current_cfg.agents):
                self.logger.info(f"[Iteration {it}] Agent {i}, subregions={ag.subregions}, best_val={ag.best_val}")
                for _ in range(LOCAL_ACQ_PER_ITER):
                    ag.X_data, ag.y_data = filter_data_to_subregions(ag.X_data, ag.y_data, ag.subregions)
                    gp = SimpleGP(ag.X_data, ag.y_data, random_state=rng.integers(1e9))
                    gp.fit()
                    best_reg, best_ei = None, -1
                    for reg_ in ag.subregions:
                        lowers = [b[0] for b in reg_]
                        uppers = [b[1] for b in reg_]
                        if any(uppers[d]<=lowers[d] for d in range(DIMENSION)):
                            continue
                        Xc = rng.uniform(lowers, uppers, size=(30, DIMENSION))
                        e_val = float(np.max(expected_improvement(Xc, gp, f_star)))
                        if e_val>best_ei:
                            best_ei = e_val
                            best_reg = reg_
                    if best_reg is not None:
                        before = ag.X_data.shape[0]
                        ag.X_data, ag.y_data = bo_sample_region(
                            ag.X_data, ag.y_data, best_reg, rng, f_star,
                            n_candidates=50, sample_logs=self.sample_logs
                        )
                        after = ag.X_data.shape[0]
                        self.total_samples += (after - before)
                        if ag.y_data.size>0:
                            ag.best_val = min(ag.best_val, np.min(ag.y_data))

            # Update agent best arrays
            for i, agent_ in enumerate(self.current_cfg.agents):
                self.agent_best_trace[i].append(agent_.best_val)

            gb = self.current_cfg.global_best()
            self.global_best_trace.append(gb)

            dt = time.time() - it_t0
            tot_time = time.time() - self.start_time
            self.logger.info(f"[Iteration {it}] global best={gb:.4f}, iteration_time={dt:.2f}s, total_time={tot_time:.2f}s")
            self.logger.info(f"Total samples so far: {self.total_samples}")

            # store iteration-level best + total samples
            self.best_vs_iter.append(gb)
            self.best_vs_samples.append((self.total_samples, gb))

            iteration_info = {
                "iteration": it,
                "global_best": gb,
                "iteration_time": dt,
                "total_time": tot_time,
                "total_samples": self.total_samples,
                "agents": []
            }
            for i, ag in enumerate(self.current_cfg.agents):
                iteration_info["agents"].append({
                    "agent_id": i,
                    "best_val": ag.best_val,
                    "num_samples": int(ag.X_data.shape[0]),
                    "subregions": ag.subregions
                })
            self.iter_logs.append(iteration_info)

        final_best = self.current_cfg.global_best()
        final_diff = abs(final_best - GLOBAL_MIN_VAL)
        total_t = time.time() - self.start_time

        self.logger.info(f"[FINAL] best= {final_best:.4f}, diff= {final_diff:.4f}")
        self.logger.info(f"Time taken: {round(total_t,3)}s.")
        self.logger.info(f"Exp: {TEST_FUNC_NAME} d={DIMENSION}, m={self.m}, h={self.horizon}, Nc={self.n_children}, rep={self.rep}, seed={self.seed} => final best={final_best:.4f}")
        self.logger.info(f"Total samples used in experiment: {self.total_samples}")

        # Single plot with each agent line + global best + known min
        self.plot_combined_convergence()
        self.save_sample_logs()
        self.save_results_csv()
        return self.iter_logs

    def plot_combined_convergence(self):
        """
        Single combined plot:
         - x-axis is iteration (plus total samples in parentheses)
         - each agent's best trace is a separate line
         - global best is another line
         - known minimum is a horizontal line
        """
        it_range = range(1, self.total_iterations+1)
        fig, ax = plt.subplots(figsize=(9,5))

        # plot each agent line
        colors = ["red","blue","green","orange","purple","cyan","magenta","brown"]
        for i in range(self.m):
            c_ = colors[i % len(colors)]
            ax.plot(it_range, self.agent_best_trace[i], color=c_, lw=2, label=f"Agent {i} Best")

        # plot global best line
        ax.plot(it_range, self.global_best_trace, color="black", lw=2, label="Global Best")

        # known min
        if GLOBAL_MIN_VAL is not None:
            ax.axhline(y=GLOBAL_MIN_VAL, color="gray", ls=":", lw=2, label="Known Min")

        # X-axis ticks => iteration + (samples)
        xticks = list(it_range)
        xticklabels = []
        for i, it_info in enumerate(self.iter_logs):
            it_ = it_info["iteration"]
            total_samp_ = it_info["total_samples"]
            xticklabels.append(f"{it_} ({total_samp_})")

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45)
        ax.set_xlabel("Iteration (Total Samples)")
        ax.set_ylabel("Function Value (Best So Far)")
        ax.set_title(f"{TEST_FUNC_NAME}_{DIMENSION}dim_{self.m}agents_{self.total_iterations}iter_{self.horizon}horizon_{self.n_children}NC_rep{self.rep} Combined Convergence")
        ax.legend()
        plt.tight_layout()
        outname = f"{TEST_FUNC_NAME}_{DIMENSION}dim_{self.m}agents_{self.total_iterations}iter_{self.horizon}horizon_{self.n_children}NC_rep{self.rep}_combined.png"
        plt.savefig(os.path.join("outputs", outname), dpi=150)
        plt.close()

    def save_sample_logs(self):
        fname = f"{TEST_FUNC_NAME}_{DIMENSION}dim_{self.m}agents_{self.total_iterations}iter_{self.horizon}horizon_{self.n_children}NC_rep{self.rep}_samples.csv"
        path = os.path.join("outputs", fname)
        with open(path,"w") as f:
            f.write("point,subregion,f_value,time,timestamp\n")
            for row in self.sample_logs:
                pt = row["point"]
                sub = row["subregion"]
                sub_str = ";".join([f"{low:.6f}:{high:.6f}" for (low, high) in sub])
                val = row["f_value"]
                tm = row["time"]
                ts = row["timestamp"]
                f.write(f"{pt},{sub_str},{val:.6f},{tm:.6f},{ts}\n")

    def save_results_csv(self):
        fname = f"{TEST_FUNC_NAME}_{DIMENSION}dim_{self.m}agents_{self.total_iterations}iter_{self.horizon}horizon_{self.n_children}NC_rep{self.rep}_results.csv"
        path = os.path.join("outputs", fname)
        with open(path, "w") as f:
            f.write("Iteration,Best So Far,Total Samples\n")
            for it_info in self.iter_logs:
                it_ = it_info["iteration"]
                gb_ = it_info["global_best"]
                ts_ = it_info["total_samples"]
                f.write(f"{it_},{gb_},{ts_}\n")

def main():
    parser = argparse.ArgumentParser(description="MAROBO code with single combined convergence plot (agent lines + global best + known min).")
    parser.add_argument("--func", type=str, default="shubert", help="shubert, rastrigin, langermann, schwefel")
    parser.add_argument("--dim", type=int, default=2, help="Dimension.")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents.")
    parser.add_argument("--iterations", type=int, default=30, help="Number of outer iterations.")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Rollout horizon.")
    parser.add_argument("--n_children", type=int, default=NC, help="Number of candidate children.")
    parser.add_argument("--rep", type=int, default=1, help="Macro replicate index.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible initial samples.")
    args = parser.parse_args()

    logger = setup_logger(args.func, args.dim, args.agents, args.iterations, args.horizon, args.n_children, args.rep)
    set_test_function_choice(args.func, args.dim, logger)
    logger.info(f"Starting MAROBO with: func={args.func}, dim={args.dim}, m={args.agents}, "
                f"iters={args.iterations}, horizon={args.horizon}, Nc={args.n_children}, "
                f"rep={args.rep}, seed={args.seed}")

    experiment = MultiAgentPOCA(
        m=args.agents,
        total_iterations=args.iterations,
        horizon=args.horizon,
        n_children=args.n_children,
        logger=logger,
        rep=args.rep,
        seed=args.seed
    )
    experiment.run()

if __name__ == "__main__":
    main()
