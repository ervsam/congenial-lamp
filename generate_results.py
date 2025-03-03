# %% [markdown]
# # Generate results
# 
# results/
#     generate_results/
#         PBS/
#             warehouse_1/
#                 throughput.npy (different number of agents)
#                 runtime.npy (different number of agents)
#                 throughput_plot.png
#                 runtime_plot.png
#                 logger.txt
#         RANDOM/
#             ... same as above
#         QTRAN/
# 
# maps/
#     warehouse_1/
#         grid_map.npy
#         heuristic.npy
#         start_options.npy
#         goal_options.npy
#         start_locs_xN.npy # x: number of agents
#         goal_locs_xN.npy
#     random_1/
#         ...
# 
# ### Methods:
# 1. trained model
# 2. random PP
# 3. PBS
# 
# ### Maps
# 1. Warehouse
# 2. random with varying obstacle density
# 
# ## transfer to unseen maps

# %%
import numpy as np
import os.path
import yaml
import pickle

from utils import *
from Environment import Environment
from Model import QNetwork

# set np seed
np.random.seed(0)

import os
file_path = "config.yaml"

# Load the YAML config file
with open(file_path, "r") as file:
    config_file = yaml.safe_load(file)

# %%
def simulate(env, logger, algorithm, q_net=None):
    logger.reset()
    goals_reached = []
    runtimes = []

    for idx in range(TIMESTEP):
        logger.print("Step", idx)
        solved = False

        close_pairs = env.get_close_pairs()
        if close_pairs == []:
            logger.print("No close pairs, skipping instance\n")
            continue

        start_time = time.time()
        if algorithm == "random_PP":
            while time.time() - start_time < TIMEOUT:
                priorities = list(range(env.num_agents))
                np.random.shuffle(priorities)
                new_start, new_goals = env.step(priorities)
                if new_start is not None:
                    solved = True
                    break
        elif algorithm == "PBS":
            result = priority_based_search(env.grid_map, env.starts, env.goals, env.window_size, max_time=TIMEOUT)

            if result == "No Solution":
                break
            else:
                paths, priority_order = result
                paths = paths.values()

                new_start, new_goals = env.step_PBS(paths)
                solved = True

        elif algorithm == "QTRAN":
            obs_fovs = env.get_obs()
            pair_enc, q_vals = q_net(obs_fovs.unsqueeze(0), [close_pairs])
            # sample priorities
            priorities, partial_prio, pred_value = sample_priorities(env, logger, close_pairs, q_vals[0], policy='greedy')
            if priorities:
                new_start, new_goals = env.step(priorities)
                if new_start is not None:
                    solved = True
            if priorities is None or new_start is None:
                logger.print("greedy policy failed, switching to stochastic policy")
                while time.time() - start_time < TIMEOUT:
                    priorities, partial_prio, pred_value = sample_priorities(env, logger, close_pairs, q_vals[0], policy='stochastic')
                    
                    new_start, new_goals = env.step(priorities)
                    if new_start is not None:
                        solved = True
                        break

        if not solved:
            logger.print("Could not solve instance")
            return goals_reached, runtimes
        else:
            runtimes.append(time.time()-start_time)
            goals_reached.append(env.goal_reached)
            
            if algorithm != "PBS":
                logger.print("Priority ordering:", priorities)
            logger.print("Time to solve instance:", round(time.time()-start_time, 3), "\n")

    return goals_reached, runtimes

# %%

TIMEOUT = 60
TIMESTEP = 5000
num_of_agents = [10, 20, 30, 40, 50]

grid_map_list = ["random_20", "random_40", "random_60"]
algorithm_list = ["random_PP", "PBS", "QTRAN"]
# algorithm = "PBS"
# grid_map = "warehouse_2"

for grid_map in grid_map_list:
    for algorithm in algorithm_list:
        # CONFIGS
        config = config_file["generate_results"]
        DIR = config["root"] + config["subroot"]
        map_path = config["maps"][grid_map]
        env_config = config["environment"]
        dir_results = DIR + algorithm + "/" + grid_map + "/"
        if not os.path.exists(dir_results):
            os.makedirs(dir_results)

        # map
        GRID_MAP_FILE = map_path + config["map_file"]
        HEURISTIC_MAP_FILE = map_path + config["heur_file"]
        START_OPTIONS = map_path + config["start_options"]
        GOAL_OPTIONS = map_path + config["goal_options"]

        log_path = dir_results + "log/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        logger = Logger(log_path+"log.txt")

        # ### Load model
        q_net = None
        if algorithm == "QTRAN":
            q_net = QNetwork()
            q_net.load_state_dict(torch.load('./results/warehouse/q_net_model/q_net_1420000.pth'))
            # q_net.eval()
            # q_net.train()

        # ### Create environment
        # for different number of agents
        # simulate for 5000 steps on given start and goal locations

        logger.print("simulating for", num_of_agents, "using", algorithm, "on", grid_map, "\n")

        results = {}
        for num_agents in num_of_agents:
            results[num_agents] = {}
            START_LOCS = map_path + config["start_locs"] + f"{num_agents}N.pkl"
            GOAL_LOCS = map_path + config["goal_locs"] + f"{num_agents}N.pkl"

            env_config['NUM_AGENTS'] = num_agents

            env = Environment(env_config, logger=logger, grid_map_file=GRID_MAP_FILE, start_loc_options=START_OPTIONS,
                        goal_loc_options=GOAL_OPTIONS, heuristic_map_file=HEURISTIC_MAP_FILE, start_loc_file=START_LOCS, goal_loc_file=GOAL_LOCS)

            num_goals_reached, runtimes = simulate(env, logger, algorithm, q_net)
            results[num_agents]['num_goals_reached'] = num_goals_reached
            results[num_agents]['runtimes'] = runtimes

            # append to results file
            with open(dir_results+"results.pkl", "wb") as f:
                pickle.dump(results, f)

