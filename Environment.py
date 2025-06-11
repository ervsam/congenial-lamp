from generate_map import generate_map
from st_astar import space_time_astar

import numpy as np
import os
from collections import defaultdict
import pickle
from concurrent.futures import ThreadPoolExecutor

from utils import *

import time

np.random.seed(0)

class Environment:
    def __init__(self, config, logger=None, grid_map_file=None, start_loc_file=None, goal_loc_file=None, heuristic_map_file=None, start_loc_options=None, goal_loc_options=None):

        root = os.path.dirname(__file__) + '/'

        # GRID MAP
        # grid_map_file = os.path.join(os.path.dirname(__file__), grid_map_file)
        if grid_map_file is not None:
            self.grid_map = np.load(root+grid_map_file)
            self.size_y = self.grid_map.shape[0] - 2
            self.size_x = self.grid_map.shape[1] - 2
        else:
            self.obstacle_density = config['OBSTACLE_DENSITY']
            self.size_x = config['SIZE']
            self.size_y = config['SIZE']

            self.grid_map = generate_map(size_x=self.size_x, size_y=self.size_y, obstacle_density=self.obstacle_density)
            filename = 'random_grid_map_'
            i = 1
            while filename+str(i)+'.npy' in os.listdir():
                i += 1
            np.save(filename+str(i)+'.npy', self.grid_map)

        self.num_agents = config['NUM_AGENTS']
        self.fov = config['FOV']
        self.window_size = config['WINDOW_SIZE']

        self.logger = logger
        self.colors = [plt.cm.hsv(i / self.num_agents) for i in range(self.num_agents)]

        self.dynamic_constraints = dict()  # {(x, y, t): agent}
        self.edge_constraints = dict()  # {((1, 1, 2), (1, 2, 2)): agent}

        self.paths = [[] for _ in range(self.num_agents)]
        self.starts = []
        self.optimal_starts = []
        self.goals = []
        self.optimal_goal_reached_per_agent = []

        # to calculate delays
        self.actual_path_lengths = {}
        self.optimal_path_lengths = {}

        self.goal_reached = 0
        self.optimal_goal_reached = 0

        if start_loc_options:
            self.start_loc_options = np.load(root+start_loc_options)
        else:
            self.start_loc_options = []
            for y in range(self.size_y+2):
                for x in range(self.size_x+2):
                    if self.grid_map[y, x] == 0:
                        self.start_loc_options.append((y, x))
        if goal_loc_options:
            self.goal_loc_options = np.load(root+goal_loc_options)
        else:
            self.goal_loc_options = []
            for y in range(self.size_y+2):
                for x in range(self.size_x+2):
                    if self.grid_map[y, x] == 0:
                        self.goal_loc_options.append((y, x))

        if start_loc_file is not None:
            # self.starts = [tuple(x) for x in np.load(start_loc_file)]
            with open(start_loc_file, "rb") as f:
                self.starts = pickle.load(f)

        else:
            self.starts = self._get_start_locs()
        self.optimal_starts = self.starts.copy()

        if goal_loc_file is not None:
            with open(goal_loc_file, "rb") as f:
                self.goals = pickle.load(f)
        else:
            self.goals = self._get_goals_locs()

        # generate heuristic map
        if heuristic_map_file and os.path.exists(os.path.join(os.path.dirname(__file__), heuristic_map_file)):
            logger.print("Environment.__init__: loading heuristic map from file")
            self.heuristic_map = np.load(root+heuristic_map_file, allow_pickle=True).item()
        else:
            self.heuristic_map = self._get_heuristic_map()
            np.save(root+heuristic_map_file, self.heuristic_map)

        # FOR WHEN USING TRAINED MODEL
        self.use_QTRAN = True
        if self.use_QTRAN:
            self.DHC_heur = self._get_DHC_heur()

    def _get_start_locs(self):
        # choose self.num_agents random starting positions
        idxs = np.random.choice(len(self.start_loc_options), self.num_agents, replace=False)
        starts = [tuple(self.start_loc_options[ind]) for ind in idxs]
        return starts

    def _get_goals_locs(self):
        goals = []
        for agent in range(self.num_agents):
            goals_per_agent = []
            
            while len(goals_per_agent) == 0 or np.abs(goals_per_agent[-1][0] - self.starts[agent][0]) + np.abs(goals_per_agent[-1][1] - self.starts[agent][1]) < self.window_size:
                idxs = np.random.choice(len(self.goal_loc_options), 1, replace=False)
                new_goal = [tuple(self.goal_loc_options[idx]) for idx in idxs]
                goals_per_agent += new_goal

            # add extras just in case
            for _ in range(3):
                idxs = np.random.choice(len(self.goal_loc_options), 1, replace=False)
                new_goal = [tuple(self.goal_loc_options[idx]) for idx in idxs]
                goals_per_agent += new_goal

            goals.append(goals_per_agent)
        return goals

    def _get_heuristic_map(self):
        heuristic_map = dict()
        # for each cell
        for y in range(self.size_y+2):
            for x in range(self.size_x+2):
                # find length from shortest path from all other cells using A*
                if self.grid_map[y, x] == 1:
                    continue
                self.logger.print("Environment.__init__: find shortest distance from", (y, x), "to all other cells")
                heuristic_map[(y, x)] = np.zeros((self.size_y+2, self.size_x+2))
                for y2 in range(self.size_y+2):
                    for x2 in range(self.size_x+2):
                        if self.grid_map[y2, x2] == 1:
                            heuristic_map[(y, x)][y2, x2] = np.inf
                            continue
                        heuristic_map[(y, x)][y2, x2] = len(space_time_astar(self.grid_map, (y, x), [(y2, x2)], set(), set())) - 1
        return heuristic_map

    def _get_DHC_heur_to_goal(self, goal):
        heur = np.zeros((4, *self.grid_map.shape))
        heur_map = self.heuristic_map[goal]
        for y in range(self.grid_map.shape[0]):
            for x in range(self.grid_map.shape[1]):
                if self.grid_map[y, x] == 0:
                    # up
                    if y > 0 and heur_map[y-1, x] < heur_map[y, x]:
                        assert heur_map[y-1, x] == heur_map[y, x]-1
                        heur[0, y, x] = 1
                    # down
                    if y < self.grid_map.shape[0]-1 and heur_map[y+1, x] < heur_map[y, x]:
                        assert heur_map[y+1, x] == heur_map[y, x]-1
                        heur[1, y, x] = 1
                    # left
                    if x > 0 and heur_map[y, x-1] < heur_map[y, x]:
                        assert heur_map[y, x-1] == heur_map[y, x]-1
                        heur[2, y, x] = 1
                    # right
                    if x < self.grid_map.shape[1]-1 and heur_map[y, x+1] < heur_map[y, x]:
                        assert heur_map[y, x+1] == heur_map[y, x]-1
                        heur[3, y, x] = 1
        return heur

    def _get_DHC_heur(self):
        # DHC HEURISTIC MAP
        DHC_heur = []
        for agent in range(self.num_agents):
            # number of goals, 4 directions, map size
            heur = np.zeros((len(self.goals[agent]), 4, *self.grid_map.shape))
            for i, goal in enumerate(self.goals[agent]):
                heur[i] = self._get_DHC_heur_to_goal(goal)
            DHC_heur.append(heur)
        return DHC_heur

    def reset(self, num_agents=None):
        if num_agents is not None:
            self.num_agents = num_agents
            self.colors = [plt.cm.hsv(i / num_agents) for i in range(num_agents)]

        self.dynamic_constraints = dict()  # {(x, y, t): agent}
        self.edge_constraints = dict()  # {((1, 1, 2), (1, 2, 2)): agent}

        self.paths = [[] for _ in range(self.num_agents)]
        self.starts = []
        self.optimal_starts = []
        self.goals = []
        self.optimal_goal_reached_per_agent = []

        self.goal_reached = 0
        self.optimal_goal_reached = 0

        # generate random starts and goals on empty cells
        self.starts = self._get_start_locs()
        self.optimal_starts = self.starts.copy()

        self.goals = self._get_goals_locs()

        self.DHC_heur = self._get_DHC_heur()

    

    def step(self, priorities):
        # Prepare optimal path planning in parallel:
        def plan_optimal(args):
            agent, starts, goals, grid_map, window_size = args
            return space_time_astar(grid_map, starts[agent], goals, {}, {})
        
        self.goal_reached = 0
        self.optimal_goal_reached = 0
        self.goal_reached_per_agent = [0 for _ in range(self.num_agents)]
        self.optimal_goal_reached_per_agent = [0 for _ in range(self.num_agents)]
        # clear the dynamic constraints and edge constraints
        self.dynamic_constraints.clear()
        self.edge_constraints.clear()
        # clear the paths
        self.paths = [[] for _ in range(self.num_agents)]

        self.actual_path_lengths = {}
        self.optimal_path_lengths = {}

        temp_paths = []
        temp_opt_paths = []

        goals_to_plan = []
        for idx, agent in enumerate(priorities):
            # goals_tmp = self.goals[agent][:self.window_size+1]

            goals_within = []
            curr_pos = self.starts[agent]
            cumulative_dist = 0
            for g in self.goals[agent]:
                dist = self.heuristic_map[curr_pos][g[0], g[1]]
                if cumulative_dist + dist > self.window_size:
                    break
                cumulative_dist += dist
                goals_within.append(g)
                curr_pos = g  # Next segment is from here
            extra_goal = None
            for g in self.goals[agent][len(goals_within):]:
                dist = self.heuristic_map[curr_pos][g[0], g[1]]
                if cumulative_dist + dist > self.window_size:
                    extra_goal = g
                    break
            if extra_goal is not None:
                goals_tmp = goals_within + [extra_goal]
            else:
                goals_tmp = goals_within
            if not goals_tmp:
                goals_tmp = [self.goals[agent][0]]

            goals_to_plan.append(goals_tmp)

        # Prepare the argument list for all agents:
        opt_args = [
            (agent, self.starts, goals, self.grid_map, self.window_size)
            for (agent, goals) in zip(priorities, goals_to_plan)
        ]
        with ThreadPoolExecutor(max_workers=8) as executor:
            temp_opt_paths = list(executor.map(plan_optimal, opt_args))

        for idx, agent in enumerate(priorities):
            assert len(priorities) == self.num_agents, "Environment.step(): priorities length is not equal to number of agents"
            
            # print("Env.step: Agent", agent, self.starts[agent], "planning for goals", goals_to_plan[idx])

            path = space_time_astar(self.grid_map, self.starts[agent], goals_to_plan[idx], self.dynamic_constraints, self.edge_constraints)
            # print(f"Env.step: Agent {agent}", path)

            if path is None:
                return None, None

            temp_paths.append(path)
            self.actual_path_lengths[agent] = path

            # optimal_path = space_time_astar(self.grid_map, self.starts[agent], goals_to_plan, {}, {})
            # temp_opt_paths.append(optimal_path)
            self.optimal_path_lengths[agent] = temp_opt_paths[idx]
            self.optimal_starts[agent] = (temp_opt_paths[idx][self.window_size][0], temp_opt_paths[idx][self.window_size][1])

            # add path to dynamic constraints (only add the first window_size elements)
            for y, x, t in path:
                if t > self.window_size:
                    break
                self.dynamic_constraints[(y, x, t)] = agent
            # add edge constraints (only add the first window_size elements)
            try:
                # for i in range(self.window_size):
                for t, pos in enumerate(path):
                    if t > self.window_size:
                        break
                    edge = ((path[t][0], path[t][1], path[t][2]), (path[t+1][0], path[t+1][1], path[t+1][2]))
                    self.edge_constraints[edge] = agent
            except IndexError:
                self.logger.print("IndexError: path is too short")
                self.logger.print(path)

        # update the starts and goals
        for agent, path, opt_path in zip(priorities, temp_paths, temp_opt_paths):
            # update the start position of the agent
            self.starts[agent] = (path[self.window_size][0], path[self.window_size][1])

            # check if a goal is reached by going through the path
            goal_idx = 0
            for y, x, t in path[1:self.window_size+1]:
                coord = (y, x)
                # the agent finish its current task
                if self.goals[agent] and coord == self.goals[agent][goal_idx]:
                    self.goal_reached += 1
                    self.goal_reached_per_agent[agent] += 1
                    # self.logger.print(f"Environment.step(): Agent {agent} reached goal {goal_idx}")
                    # look at next goal, did it reach that too?
                    goal_idx += 1

            # how many goals reached by optimal path
            goal_idx_opt = 0
            for y, x, t in opt_path[1:self.window_size+1]:
                coord = (y, x)
                # the agent finish its current task
                if self.goals[agent] and coord == self.goals[agent][goal_idx_opt]:
                    self.optimal_goal_reached += 1
                    self.optimal_goal_reached_per_agent[agent] += 1
                    # look at next goal, did it reach that too?
                    goal_idx_opt += 1

            # remove goals that are reached
            self.goals[agent] = self.goals[agent][goal_idx:]
            
            # remove DHC heurs of goals that are reached
            if self.use_QTRAN:
                self.DHC_heur[agent] = self.DHC_heur[agent][goal_idx:]

            while len(self.goals[agent]) <= self.window_size:
                idxs = np.random.choice(len(self.goal_loc_options), 1, replace=False)
                new_goal = [tuple(self.goal_loc_options[idx]) for idx in idxs]
                self.goals[agent] += new_goal

                if self.use_QTRAN:
                    new_DHC_heur = self._get_DHC_heur_to_goal(new_goal[0])
                    self.DHC_heur[agent] = np.concatenate([self.DHC_heur[agent], np.expand_dims(new_DHC_heur, axis=0)], axis=0)

            self.paths[agent] = path[:self.window_size+1]

            if self.use_QTRAN:
                assert len(self.goals[agent]) == self.DHC_heur[agent].shape[0]

        return self.starts, self.goals
    
    def step_per_group(self, priorities):
        self.goal_reached = 0
        self.optimal_goal_reached = 0
        self.goal_reached_per_agent = [0 for _ in range(self.num_agents)]
        self.optimal_goal_reached_per_agent = [0 for _ in range(self.num_agents)]
        # clear the dynamic constraints and edge constraints
        self.dynamic_constraints.clear()
        self.edge_constraints.clear()
        # clear the paths
        self.paths = [[] for _ in range(self.num_agents)]

        self.actual_path_lengths = {}
        self.optimal_path_lengths = {}

        temp_paths = []
        temp_opt_paths = []

        for agent in priorities:
            goals_to_plan = self.goals[agent][:self.window_size]
            path = space_time_astar(self.grid_map, self.starts[agent], goals_to_plan, self.dynamic_constraints, self.edge_constraints)

            if path is None:
                return None

            temp_paths.append(path)
            self.actual_path_lengths[agent] = path

            optimal_path = space_time_astar(self.grid_map, self.starts[agent], goals_to_plan, {}, {})

            temp_opt_paths.append(optimal_path)
            self.optimal_path_lengths[agent] = optimal_path

            self.optimal_starts[agent] = (optimal_path[self.window_size][0], optimal_path[self.window_size][1])

            # add path to dynamic constraints (only add the first window_size elements)
            for y, x, t in path:
                if t > self.window_size:
                    break
                self.dynamic_constraints[(y, x, t)] = agent
            # add edge constraints (only add the first window_size elements)
            try:
                # for i in range(self.window_size):
                for t, pos in enumerate(path):
                    if t > self.window_size:
                        break
                    edge = ((path[t][0], path[t][1], path[t][2]), (path[t+1][0], path[t+1][1], path[t+1][2]))
                    self.edge_constraints[edge] = agent
            except IndexError:
                self.logger.print("IndexError: path is too short")
                self.logger.print(path)

        delays = {}
        for agent in priorities:
            delays[agent] = len(self.actual_path_lengths[agent]) - len(self.optimal_path_lengths[agent])
        return delays

    
    def step_PBS(self, paths):
        self.goal_reached = 0
        self.goal_reached_per_agent = [0 for _ in range(self.num_agents)]
        self.paths = [[] for _ in range(self.num_agents)]

        # update the starts and goals
        for agent, path in enumerate(paths):
            # update the start position of the agent
            self.starts[agent] = (path[self.window_size][0], path[self.window_size][1])

            # check if a goal is reached by going through the path
            goal_idx = 0
            for y, x in path[1:self.window_size+1]:
                coord = (y, x)
                # the agent finish its current task
                if self.goals[agent] and coord == self.goals[agent][goal_idx]:
                    self.goal_reached += 1
                    self.goal_reached_per_agent[agent] += 1
                    # self.logger.print(f"Environment.step(): Agent {agent} reached goal {goal_idx}")
                    # look at next goal, did it reach that too?
                    goal_idx += 1

            # remove goals that are reached
            self.goals[agent] = self.goals[agent][goal_idx:]
            
            while len(self.goals[agent]) <= self.window_size:
                idxs = np.random.choice(len(self.goal_loc_options), 1, replace=False)
                new_goal = [tuple(self.goal_loc_options[idx]) for idx in idxs]
                self.goals[agent] += new_goal

            self.paths[agent] = path[:self.window_size+1]

        return self.starts, self.goals


    def _get_fov(self, grid_map, x, y, fov):
        '''
        Get the field of view centered at (x, y) with a size of fov x fov

        Params
        ======
            grid_map (numpy array): grid map
            x (int): x coordinate
            y (int): y coordinate
            fov (int): field of view size

        Returns
        =======
            numpy array: field of view
        '''
        max_val = np.max(grid_map)
        padded_grid = np.pad(grid_map, pad_width=fov//2, mode='constant', constant_values=0)
        return padded_grid[x:x+fov, y:y+fov]

    def get_obs(self):
        '''
        Get the field of view for all agents

        Returns
        =======
            torch tensor: field of view for all agents
        '''

        layers = 8

        obs = np.zeros((self.num_agents, layers, self.fov, self.fov), dtype=np.float32)

        starts, goals = self.starts, self.goals

        for agent, (agent_pos, goal) in enumerate(zip(starts, goals)):
            x, y = agent_pos

            # 1. SURROUNDING OBSTACLES
            obs[agent, 0] = self._get_fov(self.grid_map, x, y, self.fov)

            # 2. SURROUNDING AGENTS
            agent_map = np.zeros((self.grid_map.shape))
            arr = np.array(self.starts)
            agent_map[arr[:,0], arr[:,1]] = 1
            obs[agent, 1] = self._get_fov(agent_map, x, y, self.fov)

            # 2. HEURISTIC TO GOAL
            # heur = self._get_fov(self.heuristic_map[goal[0]], x, y, self.fov)
            padded_grid = np.pad(self.heuristic_map[goal[0]], pad_width=self.fov//2, mode='constant', constant_values=np.inf)
            heur = padded_grid[x:x+self.fov, y:y+self.fov]
            # normalize the heuristic map
            max_val = np.max(heur[heur < np.inf])
            obs[agent, 2] = heur / max_val

            # 3. COMBINED HEURISTIC MAP
            # neighbours = self._get_neighboring_agents(agent)
            # agent_heur = np.zeros((self.fov, self.fov))
            # for n in neighbours:
            #     # get heuristic map of each agent and combine the heuristic maps
            #     agent_heur += self._get_fov(
            #         self.heuristic_map[goals[n][0]], x, y, self.fov)
            # # d. normalize the combined heuristic map
            # max_val = np.max(agent_heur[agent_heur < np.inf])
            # obs[agent, 3] = agent_heur / (max_val + 1e-10)

            # 4. DHC HEURISTIC LAYER
            obs[agent, 3] = self._get_fov(self.DHC_heur[agent][0][0], x, y, self.fov)
            obs[agent, 4] = self._get_fov(self.DHC_heur[agent][0][1], x, y, self.fov)
            obs[agent, 5] = self._get_fov(self.DHC_heur[agent][0][2], x, y, self.fov)
            obs[agent, 6] = self._get_fov(self.DHC_heur[agent][0][3], x, y, self.fov)

            normalized_coord = np.zeros((self.fov, self.fov))
            normalized_coord[0, 0] = y / self.size_x
            normalized_coord[0, 1] = x / self.size_y
            obs[agent, 7] = normalized_coord

        obs_fovs = torch.tensor(obs)
        obs_fovs = torch.where(torch.isinf(obs_fovs), torch.tensor(1), obs_fovs)

        return obs_fovs

    def get_neighbor_goal_heuristics_as_patches(self):
        """
        For each agent, returns a list of FOV patches for all its neighbors' goal heuristics.
        Returns:
            List[List[Tensor]]: shape (num_agents, num_neighbors_i, C, fov, fov)
        """
        num_agents = self.num_agents
        neighbor_features = []
        for agent in range(num_agents):
            x, y = self.starts[agent]
            neighbors = self._get_neighboring_agents(agent)
            patches = []
            for neighbor in neighbors:
                neighbor_goal = self.goals[neighbor][0]
                heur_map = self.heuristic_map[neighbor_goal]

                padded_grid = np.pad(heur_map, pad_width=self.fov//2, mode='constant', constant_values=np.inf)
                heur = padded_grid[x:x+self.fov, y:y+self.fov]
                # normalize the heuristic map
                max_val = np.max(heur[heur < np.inf])
                heur_fov = heur / max_val
                
                # Optionally expand dims to match encoder input shape, e.g., (1, fov, fov)
                # and to float32 tensor
                heur_fov = torch.tensor(heur_fov, dtype=torch.float32).unsqueeze(0)
                heur_fov = torch.where(torch.isinf(heur_fov), torch.tensor(1), heur_fov)
                patches.append(heur_fov)
            neighbor_features.append(patches)
        return neighbor_features


    def _get_neighboring_agents(self, agent):
        neighbors = []
        for a in range(self.num_agents):
            if a == agent:
                continue
            x, y = self.starts[a]
            i, j = self.starts[agent]
            if self.heuristic_map[(x, y)][i, j] <= self.window_size*2:
                neighbors.append(a)
        return neighbors

    def get_close_pairs(self) -> list[tuple[int, int]]:
        close_pairs = []
        for agent in range(self.num_agents):
            for neighbor in self._get_neighboring_agents(agent):
                if neighbor > agent:
                    close_pairs.append((agent, neighbor))
        return close_pairs

    def get_delays(self):
        # UPDATE: delay = actual path length - "if no other agents" path length
        delays = []
        for agent in range(self.num_agents):
            # self.logger.print(f"Env.get_delays: agent {agent} actual path: {self.actual_path_lengths[agent]}, optimal path: {self.optimal_path_lengths[agent]}")
            delays.append(len(self.actual_path_lengths[agent]) - len(self.optimal_path_lengths[agent]))

        return delays


    class UnionFind:
        def __init__(self):
            self.parent = {}

        def find(self, node):
            if self.parent.get(node, node) != node:
                self.parent[node] = self.find(self.parent[node])  # Path compression
            return self.parent.get(node, node)

        def union(self, node1, node2):
            root1, root2 = self.find(node1), self.find(node2)
            if root1 != root2:
                self.parent[root2] = root1  # Union by linking roots

    def connected_edge_groups(self):
        uf = self.UnionFind()
        edges = self.get_close_pairs()

        # Register all nodes in the union-find structure
        for u, v in edges:
            uf.union(u, v)

        # Group edges by connected components
        edge_groups = defaultdict(list)
        for u, v in edges:
            root = uf.find(u)
            edge_groups[root].append((u, v))

        return list(edge_groups.values())

    def show_current_state(self):
        plt.imshow(self.grid_map, cmap='gray_r')

        # Scatter plot of agent starting positions
        for agent, (x, y) in enumerate(self.starts):
            plt.text(y, x, agent, c=self.colors[agent], size=6, ha='center', va='center')

        # Scatter plot of goal locations
        # for agent, gs in enumerate(self.goals):
        #     for idx, (x, y) in enumerate(gs):
        #         plt.text(y, x, idx, c=self.colors[agent], size=6, ha='right', va='baseline')
        for agent, gs in enumerate(self.goals):
            x, y = gs[0]
            plt.text(y, x, agent, c=self.colors[agent], size=3, ha='right', va='baseline')

        # save figure
        plt.savefig('current_state.png')
        plt.show()