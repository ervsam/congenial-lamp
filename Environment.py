from generate_map import generate_map
from st_astar import space_time_astar

import numpy as np
import os
from collections import defaultdict

from utils import *

# set np seed
np.random.seed(0)

class Environment:
    def __init__(self, config, logger=None, grid_map_file=None, starts=None, goals=None, heuristic_map_file=None, start_loc_options=None, goal_loc_options=None):

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


        self._old_goals = []
        self._old_starts = []
        self._old_heur_fovs = []

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

        if starts is not None:
            self.starts = starts.copy()
        else:
            self.starts = self._get_start_locs()
        self.optimal_starts = self.starts.copy()

        if goals is not None:
            self.goals = goals.copy()
        else:
            self.goals = self._get_goals_locs()

        # generate heuristic map
        if heuristic_map_file and os.path.exists(heuristic_map_file):
            logger.print("Environment.__init__: loading heuristic map from file")
            self.heuristic_map = np.load(root+heuristic_map_file, allow_pickle=True).item()
        else:
            self.heuristic_map = self._get_heuristic_map()
            np.save(root+heuristic_map_file, self.heuristic_map)

        # old heuristic map
        for agent in range(self.num_agents):
            heur = self._get_fov(self.heuristic_map[self.goals[agent][0]], self.starts[agent][0], self.starts[agent][1], self.fov)
            heur /= np.max(heur[heur < np.inf])
            self._old_heur_fovs.append(heur)

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

    def _get_DHC_heur(self):
        # DHC HEURISTIC MAP
        DHC_heur = []
        for agent in range(self.num_agents):
            # number of goals, 4 directions, map size
            heur = np.zeros((len(self.goals[agent]), 4, *self.grid_map.shape))
            for i, goal in enumerate(self.goals[agent]):
                heur_map = self.heuristic_map[goal]
                for y in range(self.grid_map.shape[0]):
                    for x in range(self.grid_map.shape[1]):
                        if self.grid_map[y, x] == 0:
                            # up
                            if y > 0 and heur_map[y-1, x] < heur_map[y, x]:
                                assert heur_map[y-1, x] == heur_map[y, x]-1
                                heur[i, 0, y, x] = 1
                            # down
                            if y < self.grid_map.shape[0]-1 and heur_map[y+1, x] < heur_map[y, x]:
                                assert heur_map[y+1, x] == heur_map[y, x]-1
                                heur[i, 1, y, x] = 1
                            # left
                            if x > 0 and heur_map[y, x-1] < heur_map[y, x]:
                                assert heur_map[y, x-1] == heur_map[y, x]-1
                                heur[i, 2, y, x] = 1
                            # right
                            if x < self.grid_map.shape[0]-1 and heur_map[y, x+1] < heur_map[y, x]:
                                assert heur_map[y, x+1] == heur_map[y, x]-1
                                heur[i, 3, y, x] = 1
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

        self._old_goals = []
        self._old_starts = []
        self._old_heur_fovs = []

        # generate random starts and goals on empty cells
        self.starts = self._get_start_locs()
        self.optimal_starts = self.starts.copy()

        self.goals = self._get_goals_locs()

        self.DHC_heur = self._get_DHC_heur()


        # old heuristic map
        for agent in range(self.num_agents):
            heur = self._get_fov(self.heuristic_map[self.goals[agent][0]], self.starts[agent][0], self.starts[agent][1], self.fov)
            heur /= np.max(heur[heur < np.inf])
            self._old_heur_fovs.append(heur)

    def step(self, priorities):
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

        self._old_goals = self.goals.copy()
        self._old_starts = self.starts.copy()

        temp_paths = []
        temp_opt_paths = []

        for agent in priorities:
            assert len(priorities) == self.num_agents, "Environment.step(): priorities length is not equal to number of agents"
            
            # TODO: update ST A* to handle multiple goals
            goals_to_plan = self.goals[agent][:self.window_size]
            path = space_time_astar(self.grid_map, self.starts[agent], goals_to_plan, self.dynamic_constraints, self.edge_constraints)

            print(f"Agent {agent}:", path)

            if path is None:
                return None, None

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

            # # if last goal is self.window_size away, add a new goal
            # while len(self.goals[agent]) == 0 or np.abs(self.goals[agent][-1][0] - self.starts[agent][0]) + np.abs(
            #         self.goals[agent][-1][1] - self.starts[agent][1]) < self.window_size:
            #     new_goal = (np.random.randint(1, self.size),
            #                 np.random.randint(1, self.size))
            #     while self.grid_map[new_goal[0], new_goal[1]] == 1:
            #         new_goal = (np.random.randint(1, self.size),
            #                     np.random.randint(1, self.size))
            #     self.goals[agent].append(new_goal)
            # OR while len(goal) is less than window_size
            while len(self.goals[agent]) <= self.window_size:
                idxs = np.random.choice(len(self.goal_loc_options), 1, replace=False)
                new_goal = [tuple(self.goal_loc_options[idx]) for idx in idxs]
                self.goals[agent] += new_goal

            # WRONG: if i keep adding goals on each env.step(), goal list will keep increasing and ST A* will take longer to compute
            # # add extra just in case
            # new_goal = (np.random.randint(1, self.size),
            #             np.random.randint(1, self.size))
            # while self.grid_map[new_goal[0], new_goal[1]] == 1:
            #     new_goal = (np.random.randint(1, self.size),
            #                 np.random.randint(1, self.size))
            # self.goals[agent].append(new_goal)

            self.paths[agent] = path[:self.window_size+1]

        # update the heuristic map
        # self.DHC_heur = self._get_DHC_heur()

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
        padded_grid = np.pad(grid_map, pad_width=fov//2, mode='constant', constant_values=max_val)
        return padded_grid[x:x+fov, y:y+fov]

    def get_obs(self):
        '''
        Get the field of view for all agents

        Returns
        =======
            torch tensor: field of view for all agents
        '''

        layers = 7

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
            heur = self._get_fov(self.heuristic_map[goal[0]], x, y, self.fov)
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


        self._old_heur_fovs = [obs[i, 2] for i in range(self.num_agents)]

        obs_fovs = torch.tensor(obs)
        obs_fovs = torch.where(torch.isinf(obs_fovs), torch.tensor(-1), obs_fovs)
        return obs_fovs


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
        updated_delays = []
        for agent in range(self.num_agents):
            updated_delays.append(len(self.actual_path_lengths[agent]) - len(self.optimal_path_lengths[agent]))

        # WRONG (and complicated) CALCULATION OF DELAYS
        # # if no goal reached by optimal path
        # # delay = dist(current, optimal)
        # # for goal reached by optimal path, but not reached by true path
        # # delay = dist(current, G1) + dist(G1, G2) + ... + dist(Gn, optimal)
        # # where Gx is goals unreached by true path
        # delays = []
        # for (n_goals_reached, n_optimal_goals_reached, goals, loc, opt_loc) in zip(self.goal_reached_per_agent, self.optimal_goal_reached_per_agent, self._old_goals, self.starts, self.optimal_starts):
        #     if n_optimal_goals_reached - n_goals_reached > 0:
        #         # curr to goal unreached by true path
        #         delay = self.heuristic_map[goals[n_goals_reached]
        #                                    ][loc[0], loc[1]]

        #         # goal unreached by true path to goal reached by optimal path
        #         for i in range(0, n_optimal_goals_reached-1):
        #             delay += self.heuristic_map[goals[n_goals_reached+i]
        #                                         ][goals[n_goals_reached+i+1][0], goals[n_goals_reached+i+1][1]]

        #         delay += self.heuristic_map[goals[n_optimal_goals_reached-1]
        #                                     ][opt_loc[0], opt_loc[1]]
        #     else:
        #         delay = self.heuristic_map[goals[n_goals_reached]][loc[0], loc[1]] - \
        #             self.heuristic_map[goals[n_goals_reached]
        #                                ][opt_loc[0], opt_loc[1]]
        #     delays.append(delay)
        #     # heur of heur(current position) - heur(optimal position)
        #     # delays.append(
        #     #     self.heuristic_map[goals[0]][loc[0], loc[1]] -
        #     #     self.heuristic_map[goals[0]][opt_loc[0], opt_loc[1]])

        # if (delays != updated_delays):
        #     self.logger.print("Environment.get_delays(): delays and updated_delays are not the same", delays, updated_delays)

        #     for agent, delay in enumerate(delays):
        #         if len(self.actual_path_lengths[agent]) - len(self.optimal_path_lengths[agent]) != delay:
        #             self.logger.print("agent:", agent, "\nactual:", self.actual_path_lengths[agent], "\noptimal", self.optimal_path_lengths[agent])

        return updated_delays


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
            plt.text(y, x, agent, c=self.colors[agent], size=12, ha='center', va='center')

        # Scatter plot of goal locations
        for agent, gs in enumerate(self.goals):
            for idx, (x, y) in enumerate(gs):
                plt.text(y, x, idx, c=self.colors[agent], size=6, ha='right', va='baseline')

        plt.show()