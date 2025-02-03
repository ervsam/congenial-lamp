from generate_map import generate_map
from st_astar import space_time_astar

import numpy as np
import os
from collections import defaultdict

from utils import *

# set np seed
np.random.seed(0)


class Environment:
    def __init__(self, size=14, num_agents=4, obstacle_density=0.5, fov=7, window_size=3, logger=None):
        self.logger = logger
        self.colors = [plt.cm.hsv(i / num_agents) for i in range(num_agents)]


        self.size = size
        self.num_agents = num_agents
        self.obstacle_density = obstacle_density
        self.fov = fov
        self.window_size = window_size

        self.grid_map = generate_map(
            size=self.size, obstacle_density=self.obstacle_density)

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

        self.heuristic_map = dict()

        self._old_goals = []
        self._old_starts = []
        self._old_heur_fovs = []

        # generate random starts and goals on empty cells
        for agent in range(self.num_agents):
            new_start = (np.random.randint(1, self.size),
                         np.random.randint(1, self.size))
            while self.grid_map[new_start[0], new_start[1]] == 1 or new_start in self.starts:
                new_start = (np.random.randint(1, self.size),
                             np.random.randint(1, self.size))
            self.starts.append(new_start)
            self.optimal_starts.append(new_start)

        for agent in range(self.num_agents):
            goals = []
            while len(goals) == 0 or np.abs(goals[-1][0] - self.starts[agent][0]) + np.abs(goals[-1][1] - self.starts[agent][1]) < self.window_size:
                new_goal = (np.random.randint(1, self.size),
                            np.random.randint(1, self.size))
                while self.grid_map[new_goal[0], new_goal[1]] == 1:
                    new_goal = (np.random.randint(1, self.size),
                                np.random.randint(1, self.size))
                goals.append(new_goal)

            # add extras just in case
            for _ in range(3):
                new_goal = (np.random.randint(1, self.size),
                            np.random.randint(1, self.size))
                while self.grid_map[new_goal[0], new_goal[1]] == 1:
                    new_goal = (np.random.randint(1, self.size),
                                np.random.randint(1, self.size))
                goals.append(new_goal)

            self.goals.append(goals)

        # generate heuristic map
        if os.path.exists("heuristic_map.npy"):
            logger.print("Environment.__init__: loading heuristic map from file")
            self.heuristic_map = np.load("heuristic_map.npy", allow_pickle=True).item()
        else:
            # for each cell
            for y in range(self.size+2):
                for x in range(self.size+2):
                    # find length from shortest path from all other cells using A*
                    if self.grid_map[y, x] == 1:
                        continue
                    self.logger.print("Environment.__init__: find shortest distance from",
                                    (y, x), "to all other cells")
                    self.heuristic_map[(y, x)] = np.zeros((self.size+2, self.size+2))
                    for y2 in range(self.size+2):
                        for x2 in range(self.size+2):
                            if self.grid_map[y2, x2] == 1:
                                self.heuristic_map[(y, x)][y2, x2] = np.inf
                                continue
                            self.heuristic_map[(y, x)][y2, x2] = len(space_time_astar(self.grid_map, (y, x), [(y2, x2)], set(), set())) - 1
            
            # save heuristic map to file
            np.save("heuristic_map.npy", self.heuristic_map)

        # old heuristic map
        for agent in range(self.num_agents):
            heur = self._get_fov(
                self.heuristic_map[self.goals[agent][0]], self.starts[agent][0], self.starts[agent][1], self.fov)
            heur /= np.max(heur[heur < np.inf])
            self._old_heur_fovs.append(heur)

        self.get_DHCheur()

    def get_DHCheur(self):
        # DHC HEURISTIC MAP
        self.DHC_heur = []
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
            self.DHC_heur.append(heur)

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
        for agent in range(self.num_agents):
            new_start = (np.random.randint(1, self.size),
                         np.random.randint(1, self.size))
            while self.grid_map[new_start[0], new_start[1]] == 1 or new_start in self.starts:
                new_start = (np.random.randint(1, self.size),
                             np.random.randint(1, self.size))
            self.starts.append(new_start)
            self.optimal_starts.append(new_start)

        for agent in range(self.num_agents):
            goals = []
            while len(goals) == 0 or np.abs(goals[-1][0] - self.starts[agent][0]) + np.abs(goals[-1][1] - self.starts[agent][1]) < self.window_size:
                new_goal = (np.random.randint(1, self.size),
                            np.random.randint(1, self.size))
                while self.grid_map[new_goal[0], new_goal[1]] == 1:
                    new_goal = (np.random.randint(1, self.size),
                                np.random.randint(1, self.size))
                goals.append(new_goal)

            # add extras just in case
            for _ in range(3):
                new_goal = (np.random.randint(1, self.size),
                            np.random.randint(1, self.size))
                while self.grid_map[new_goal[0], new_goal[1]] == 1:
                    new_goal = (np.random.randint(1, self.size),
                                np.random.randint(1, self.size))
                goals.append(new_goal)

            self.goals.append(goals)

        self.get_DHCheur()


        # old heuristic map
        for agent in range(self.num_agents):
            heur = self._get_fov(
                self.heuristic_map[self.goals[agent][0]], self.starts[agent][0], self.starts[agent][1], self.fov)
            heur /= np.max(heur[heur < np.inf])
            self._old_heur_fovs.append(heur)

    def step(self, priorities):
        self.goal_reached = 0
        self.optimal_goal_reached = 0
        self.goal_reached_per_agent = [0 for _ in range(self.num_agents)]
        self.optimal_goal_reached_per_agent = [
            0 for _ in range(self.num_agents)]
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
            # TODO: update ST A* to handle multiple goals
            path = space_time_astar(
                self.grid_map, self.starts[agent], self.goals[agent], self.dynamic_constraints, self.edge_constraints)

            if path is None:
                return None, None

            temp_paths.append(path)
            self.actual_path_lengths[agent] = path

            optimal_path = space_time_astar(
                self.grid_map, self.starts[agent], self.goals[agent], {}, {})

            temp_opt_paths.append(optimal_path)
            self.optimal_path_lengths[agent] = optimal_path

            self.optimal_starts[agent] = (
                optimal_path[self.window_size][0], optimal_path[self.window_size][1])

            # add path to dynamic constraints (only add the first window_size elements)
            for y, x, t in path[:self.window_size+1]:
                self.dynamic_constraints[(y, x, t)] = agent
            # add edge constraints (only add the first window_size elements)
            try:
                for i in range(self.window_size):
                    self.edge_constraints[(
                        (path[i][0], path[i][1], path[i][2]), (path[i+1][0], path[i+1][1], path[i+1][2]))] = agent
            except IndexError:
                self.logger.print("IndexError: path is too short")
                self.logger.print(path)

        # update the starts and goals
        for agent, path, opt_path in zip(priorities, temp_paths, temp_opt_paths):
            # update the start position of the agent
            self.starts[agent] = (path[self.window_size]
                                  [0], path[self.window_size][1])

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
                new_goal = (np.random.randint(1, self.size),
                            np.random.randint(1, self.size))
                while self.grid_map[new_goal[0], new_goal[1]] == 1:
                    new_goal = (np.random.randint(1, self.size),
                                np.random.randint(1, self.size))
                self.goals[agent].append(new_goal)

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
        self.get_DHCheur()

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