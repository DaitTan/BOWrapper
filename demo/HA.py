from scipy.linalg import eigh
from scipy.optimize import Bounds, minimize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

# from kriging_gpr.interface.OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
# from kriging_gpr.interface.OK_Rpredict import OK_Rpredict
# from kriging_gpr.utils.OK_Rlh_kd_nugget import OK_Rlh_kd_nugget
# from kriging_gpr.utils.OK_corr import OK_corr
# from kriging_gpr.utils.OK_regr import OK_regr

from scipy.optimize import linprog
      
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dataclasses import dataclass
import polytope as pc
from staliro.specifications import RTAMTDense
from shapely.geometry import Polygon, Point
from matplotlib.patches import Polygon as mpl_polygon
import matplotlib

@dataclass
class Config:
    total_time: float
    delta_time: float
    state_sets: dict

class State:
    def __init__(self, id, dynamics, next_neighbor, config):
        self.id = id
        self.region = config.state_sets[self.id]
        self.config = config
        self.dynamics_function = dynamics
        self.next_neighbour = next_neighbor
        
    
    def get_next_neighbor(self, past_data):
        return self.next_neighbour(past_data, self.config.state_sets)

    def _n_step_solver(self, current_state, time):
        
        time_points = time+np.arange(0., self.config.delta_time, self.config.delta_time/5)
        trajectory = odeint(self.dynamics_function, current_state, time_points)
        # print(trajectory)
        # get_next_neighbor()
        next_neighbour = self.get_next_neighbor(trajectory[-1])
        
        return (self.id, trajectory[-1], self.config.delta_time, next_neighbour)
    
    

class HA:
    def __init__(self, states, init_point, config) -> None:
        self.states = states
        self.init_point = init_point
        self.config = config
        
        
    def _state_extractor(self, current_point):
        for j in self.states:
            if current_point in j.region:
                return j.id
            

    def generate_trajectory(self):
        times = np.arange(0, self.config.total_time+self.config.delta_time, self.config.delta_time)
        trajectory = [self.init_point]
        get_current_state = self._state_extractor(trajectory[-1])
        found_object = next((obj for obj in self.states if obj.id == get_current_state), None)
        x = []
        for time in times[1:]:
            
            next_point_data = found_object._n_step_solver(trajectory[-1], time)
            next_point = next_point_data[1]
            x.append(next_point_data[-1])
            trajectory.append(next_point)
            found_object = next((obj for obj in self.states if obj.id == next_point_data[-1]), None)

        return np.stack(trajectory), x, times

    
class HA_Wrapper:
    def __init__(self, states, config, specification) -> None:
        self.states = states
        self.config = config
        self.specification = specification
    
    def get_robustness(self, init_point):
        ha = HA(self.states, init_point, self.config)
        traj, d, times = ha.generate_trajectory()
        # print(times)
        rob = self.specification.evaluate(traj.T, times.T)
        
        
        return rob
    
    def state_extractor(self, init_point):
        for j in self.states:
            if init_point in j.region:
                return j.id

    
    



###################################################################################### 
# Define the sets in terms of polytopes

# x <= 1
# x >= -1
set_1_A = np.array([
                [1, 0],
                [-1,0],
                [0, 1],
                [0,-1],
            ])

set_1_b = np.array([1, 1, 1, 1])
set_1_p = pc.Polytope(set_1_A, set_1_b)

set_2_A = np.array([
                [1, 0],
                [-1,0],
                [0, 1],
                [0,-1],
            ])

# original
# set_2_b = np.array([0.95, -0.55, 0.95, -0.55])
# set_3_b = np.array([-0.55, 0.95, -0.55, 0.95])

# 2:1:1
# set_2_b = np.array([0.95, -0.05, 0.95, -0.05])
# set_3_b = np.array([ 0.95, -0.05, -0.05, 0.95])

# 2:5:3
# set_2_b = np.array([0.95, -0.05, 0.95, 0.95])
# set_3_b = np.array([ -0.05, 0.95, 0.95, 0.95])

# 1:1:1
set_2_b = np.array([0.95, -0.05, 0.65, 0.65])
set_3_b = np.array([ -0.05, 0.95, 0.95, 0.95])

set_2_p = pc.Polytope(set_2_A, set_2_b)


set_3_A = np.array([
                [1, 0],
                [-1,0],
                [0, 1],
                [0,-1],
            ])


set_3_p = pc.Polytope(set_3_A, set_3_b)

green_set_region = (set_1_p.diff(set_2_p)).diff(set_3_p)
yellow_set_region = set_2_p
blue_set_region = set_3_p



######################################################################################
# Define the dynamics

def dynamics_green_set(y, t):
    x1, x2 = y

    derivs = [x1 - x2 + 0.1*t,
            x2*np.cos(2*np.pi*x2) - x1*np.sin(2*np.pi*x1) + 0.1*t]
    
    return derivs


def dynamics_yellow_set(y, t):
        x1, x2 = y
        derivs = [x1,
                -1*x1 + x2]
        return derivs

def dynamics_blue_set(y, t):
        x1, x2 = y
        derivs = [-x1 + x2,
                    x1]
        return derivs

######################################################################################
# Next neighborung state generator


def get_next_neighbor_state1(curr_data_point, state_sets):
    
    if curr_data_point in state_sets[2]:
        return 2
    elif curr_data_point in state_sets[3]:
        return 3
    else:
        return 1
    
def get_next_neighbor_state2(curr_data_point, state_sets):
    
    return 2

def get_next_neighbor_state3(curr_data_point, state_sets):
    
    return 3

######################################################################################
# Create a Config instance
state_sets = {
    1: green_set_region,
    2: yellow_set_region,
    3: blue_set_region
}

simulation_config = Config(total_time=2, delta_time=0.01, state_sets=state_sets)
        
# Create a State instance
green_set = State(id = 1, dynamics=dynamics_green_set, config=simulation_config, next_neighbor=get_next_neighbor_state1)
yellow_set = State(id = 2, dynamics=dynamics_yellow_set, config=simulation_config, next_neighbor=get_next_neighbor_state2)
blue_set = State(id = 3, dynamics=dynamics_blue_set, config=simulation_config, next_neighbor=get_next_neighbor_state3)
states = [green_set, yellow_set, blue_set]

phi_1_x = "x_pos <= -0.85 and x_pos >= -1.15"
phi_1_y = "y_pos <= -2.0 and y_pos >= -2.5"
phi_1 = f"({phi_1_x}) and ({phi_1_y})"


phi_2_x = "x_pos <= 4 and x_pos >= 3"
phi_2_y = "y_pos <= -3 and y_pos >= -4"
phi_2 = f"({phi_2_x}) and ({phi_2_y})"

phi = f"(G[0,2] (not ({phi_1}))) and (G[0,2] (not ({phi_2})))"
specification = RTAMTDense(phi, {"x_pos" : 0, "y_pos": 1})



ha = HA_Wrapper(states, simulation_config, specification)
# init_point = np.array([-0.6, -0.6])
# rob = ha.get_robustness(init_point)
# set = ha.state_extractor(init_point)


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

pts = 100
# Create data
x = np.linspace(-1, 1, pts)
y = np.linspace(-1, 1, pts)
x, y = np.meshgrid(x, y)

# Evaluate your function for each (x, y) pair
count = 0
z = np.zeros_like(x)
for i in range(len(x)):
    for j in range(len(y)):
        print(f"{count/(pts*pts)} done")
        count += 1
        tem = ha.get_robustness(np.array([x[i, j], y[i, j]]))
        z[i, j] = tem



# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D mesh
ax.plot_surface(x, y, z, cmap='viridis')

# Customize the plot
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Mesh Plot')

# Show the plot
plt.show()