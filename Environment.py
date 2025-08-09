from generate_map import generate_map
from st_astar import space_time_astar

import numpy as np
import os
from collections import defaultdict
import pickle
from concurrent.futures import ThreadPoolExecutor
import time

from utils import *
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

np.random.seed(0)

class Environment:
    def __init__(self, config, logger=None, grid_map_file=None, start_loc_file=None, goal_loc_file=None, heuristic_map_file=None, start_loc_options=None, goal_loc_options=None, device=None):

        self.device = device

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

        grid_np = self.grid_map.astype(np.float32)
        pad = self.fov // 2
        self.padded_grid_map = torch.from_numpy(np.pad(grid_np, pad_width=pad, mode='constant', constant_values=0))
        self.padded_grid_map_cuda = self.padded_grid_map.to(self.device, non_blocking=True)  # (H+2pad, W+2pad)

        self.logger = logger
        self.colors = [plt.cm.hsv(i / self.num_agents) for i in range(self.num_agents)]

        self.starts = []
        self.optimal_starts = []
        self.goals = []

        # to calculate delays
        self.actual_path_lengths = {}
        self.optimal_path_lengths = {}

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

        # Pre-pad heuristic maps as CPU torch tensors to avoid per-call padding in get_obs
        self._padded_heuristic_map = {}
        for goal, heur_arr in self.heuristic_map.items():
            # Convert to tensor and pad with inf
            t = torch.from_numpy(heur_arr.astype(np.float32))  # (H, W)
            # add batch and channel dims for F.pad, then remove them
            t_padded = F.pad(t.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), value=float('inf'))
            self._padded_heuristic_map[goal] = t_padded.squeeze(0).squeeze(0)  # (H+2pad, W+2pad)

        _DHC_heur_map = {}
        for goal, heur_map in self.heuristic_map.items():
            dhc = np.zeros((4, *self.grid_map.shape), dtype=np.float32)
            for y in range(self.grid_map.shape[0]):
                for x in range(self.grid_map.shape[1]):
                    if self.grid_map[y, x] == 0:
                        if y > 0 and heur_map[y-1, x] < heur_map[y, x]:
                            dhc[0, y, x] = 1
                        if y < self.grid_map.shape[0]-1 and heur_map[y+1, x] < heur_map[y, x]:
                            dhc[1, y, x] = 1
                        if x > 0 and heur_map[y, x-1] < heur_map[y, x]:
                            dhc[2, y, x] = 1
                        if x < self.grid_map.shape[1]-1 and heur_map[y, x+1] < heur_map[y, x]:
                            dhc[3, y, x] = 1
            _DHC_heur_map[goal] = dhc

        pad = self.fov // 2
        self.heuristic_map_array = np.array(list(self.heuristic_map.values()), dtype=np.float32)  # (G, H+2pad, W+2pad)
        # build padded heuristic tensor for fast neighbor patch extraction
        self._heur_t = torch.from_numpy(self.heuristic_map_array)  # (G, H, W)
        self._padded_heur = F.pad(self._heur_t, (pad, pad, pad, pad), value=float('inf'))  # (G, H+2pad, W+2pad)
        self._padded_heur_cuda = self._padded_heur.to(self.device, non_blocking=True)  # (G, H+2pad, W+2pad)

        # (a) Build one giant CPU tensor of shape (G, 1, H+2pad, W+2pad) where G = number of unique goals
        _goal_list = list(self.heuristic_map.keys())
        # maps = []
        # for goal in _goal_list:
        #     arr = self.heuristic_map[goal].astype(np.float32)
        #     maps.append(np.pad(arr, pad_width=pad, mode='constant', constant_values=np.inf))
        # # Stack once and move to device
        # self._padded_maps = torch.from_numpy(np.stack(maps, axis=0)).unsqueeze(1)  # (G,1,H',W')
        # (b) A lookup from goal→index in that tensor
        self._goal_index = {g:i for i,g in enumerate(_goal_list)}
        
        # 3. DHC windows for each goal in _goal_list, shape (G,4,H,W,fov,fov)
        self.DHC_heur_arr = np.array(list(_DHC_heur_map.values()), dtype=np.float32)
        self.DHC_heur_arr_cuda = torch.from_numpy(self.DHC_heur_arr).to(self.device, non_blocking=True)

        # dhc_wins = []
        # for goal in tqdm(_goal_list):
        #     # shape (4,H,W)
        #     arr = _DHC_heur_map[goal].astype(np.float32)
        #     t = torch.from_numpy(arr)  # (4,H,W)
        #     t = F.pad(t, (pad, pad, pad, pad))  # (4,H',W')
        #     # unfold for each direction: (4,H',W') -> (4,H,W,fov,fov)
        #     t_unf = t.unfold(1, self.fov, 1).unfold(2, self.fov, 1)  # (4,H,W,fov,fov)
        #     dhc_wins.append(t_unf)
        # # Keep full DHC windows on CPU to avoid GPU OOM; move slices to GPU in get_obs
        # self._dhc_windows = torch.stack(dhc_wins, dim=0).cpu()


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

    # def _get_DHC_heur(self):
    #     return [
    #         np.stack([self._DHC_heur_map[goal] for goal in agent_goals], axis=0) for agent_goals in self.goals
    #     ]

    def _get_fov(self, grid_map, x, y, fov):
        padded_grid = np.pad(grid_map, pad_width=fov//2, mode='constant', constant_values=0)
        return padded_grid[x:x+fov, y:y+fov]
    
    def get_obs(self, agents):
        # GPU vectorized version using torch.nn.functional.unfold
        device = self.device
        N = len(agents)
        fov, pad = self.fov, self.fov // 2
        # Prepare coords:
        starts_tensor = torch.tensor([self.starts[a] for a in agents], device=device)
        ys, xs = starts_tensor[:,0], starts_tensor[:,1]
        # 1) obstacles+agent occupancy
        base = self.padded_grid_map_cuda.unsqueeze(0).unsqueeze(0)  # (1,1,H',W')
        occ  = torch.zeros_like(self.padded_grid_map_cuda, device=device)
        coords_all = torch.tensor(self.starts, device=device)
        occ[coords_all[:,0], coords_all[:,1]] = 1
        occ = occ.unsqueeze(0).unsqueeze(0)  # (1,1,H',W')
        all_feats = torch.cat([base, occ], dim=1)     # (1,2,H',W')
        patches = F.unfold(all_feats, kernel_size=fov).view(1, 2, fov*fov, -1)  # (1,2,fov*fov,#windows)
        # Select windows at our positions
        idxs = ys * (self.grid_map.shape[1]) + xs
        obs0 = patches[0,:, :, idxs].view(2*N, fov, fov)  # (2N,fov,fov)
        # 2) heuristics: unfold and gather
        heur_windows = self._padded_heur_cuda.unsqueeze(1)  # (G,1,H',W')
        heur_patches = heur_windows.unfold(2,fov,1).unfold(3,fov,1)  # (G,1,H,W,fov,fov)
        g_idxs = torch.tensor([self._goal_index[self.goals[a][0]] for a in agents], device=device)
        hpatch = heur_patches[g_idxs,0, ys, xs]  # (N,fov,fov)
        # mask and normalize
        mask = hpatch != float('inf')
        finite = torch.nan_to_num(hpatch, nan=0.0, posinf=0.0, neginf=0.0)
        maxv = torch.where(mask, finite, torch.tensor(0., device=finite.device)).amax(dim=(1,2)).clamp(min=1.0).view(-1,1,1)
        norm = torch.where(mask, finite / maxv, torch.tensor(1.0, device=finite.device))
        hpatch = norm
        # 3) DHC: pad and unfold like heuristics, index by goal + agent pos
        # build and pad the DHC heuristic tensor on GPU
        DHC_heur = self.DHC_heur_arr_cuda  # (G,4,H,W)
        # pad height and width by `pad` on both sides
        DHC_heur = F.pad(DHC_heur, (pad, pad, pad, pad))  # now (G,4,H+2pad,W+2pad)
        # unfold spatial dims (2 -> height, 3 -> width) to extract fov×fov windows
        dhc_windows = DHC_heur.unfold(2, fov, 1).unfold(3, fov, 1)  # (G,4,H,W,fov,fov)
        dhc_patch = dhc_windows[g_idxs, :, ys, xs].to(device, non_blocking=True)  # (N,4,fov,fov)
        # 4) coordinate channel: build tensor directly on GPU
        coord = torch.zeros((N,1,fov,fov), device=device)
        coord[:,0,0,0] = xs / self.size_x
        coord[:,0,0,1] = ys / self.size_y
        # 5) stack all layers: obstacle, occ, norm-heur, dhc (4), coord → (N,8,fov,fov)
        obs_out = torch.cat([obs0.view(N,2,fov,fov), hpatch.unsqueeze(1), dhc_patch, coord], dim=1)
        return obs_out

        # OLD
        # import time
        # t0 = time.perf_counter()
        # # Build entire obs_fovs on CPU, then move once to GPU
        # N = len(agents)
        # fov = self.fov
        # pad = fov // 2
        # layers = 8

        # # 1) Gather agent positions and goals
        # starts = [self.starts[a] for a in agents]
        # goals0 = [self.goals[a][0] for a in agents]

        # # 2) Allocate a CPU tensor for obs_fovs
        # obs_fovs_cpu = torch.empty((N, layers, fov, fov), dtype=torch.float32)

        # # 4) Agent occupancy layer (CPU) using padded_grid_map shape
        # # build padded agent occupancy tensor once
        # padded_agent = torch.zeros_like(self.padded_grid_map)
        # coords = torch.tensor(self.starts, dtype=torch.long)
        # coords_pad = coords + pad
        # padded_agent[coords_pad[:, 0], coords_pad[:, 1]] = 1.0

        # # 3) Static obstacle map (CPU)
        # for i, ((y0, x0), goal) in enumerate(zip(starts, goals0)):
        #     obs_fovs_cpu[i, 0] = self.padded_grid_map[y0:y0+fov, x0:x0+fov]
        #     obs_fovs_cpu[i, 1] = padded_agent[y0:y0+fov, x0:x0+fov]

        #     # extract the (H+2pad, W+2pad) padded tensor for this goal
        #     t_pad = self._padded_heuristic_map[goal]  # torch.Tensor
        #     # slice out the fov x fov window
        #     patch = t_pad[y0:y0+fov, x0:x0+fov]
        #     # create mask for finite entries
        #     mask = patch != float('inf')
        #     # compute max over valid entries, default to 1.0 if none
        #     if mask.any():
        #         maxv = patch[mask].max()
        #     else:
        #         maxv = torch.tensor(1.0, dtype=patch.dtype)
        #     # normalize valid cells and set invalid cells to 1.0
        #     norm = torch.where(mask, patch / maxv, torch.tensor(1.0, dtype=patch.dtype))
        #     obs_fovs_cpu[i, 2] = norm

        #     g_idx = self._goal_index[ goals0[i] ]
        #     # all4 = self._dhc_windows[ g_idx, :, y0, x0 ]   # → shape (4, fov, fov)

        #     arr = self.DHC_heur_arr[g_idx]
        #     t = torch.from_numpy(arr)  # (4,H,W)
        #     t = F.pad(t, (pad, pad, pad, pad))  # (4,H',W')
        #     all4 = t[:, y0:y0+fov, x0:x0+fov]
            
        #     obs_fovs_cpu[i, 3:7] = all4

        # # 7) Coordinate channel (CPU)
        # for i, (y0, x0) in enumerate(starts):
        #     coord_map = torch.zeros((fov, fov), dtype=torch.float32)
        #     coord_map[0, 0] = x0 / self.size_x
        #     coord_map[0, 1] = y0 / self.size_y
        #     obs_fovs_cpu[i, 7] = coord_map

        # return obs_fovs_cpu

    def get_neighbor_goal_heuristics_as_patches(self, agents):
        """
        Vectorized extraction of neighbor goal heuristic patches.
        Returns lists (per agent) of feature tensors of shape (P_i,1,fov,fov)
        and coordinate tensors of shape (P_i,2).
        """
        device = self.device

        t0 = time.perf_counter()
        torch.cuda.synchronize()

        N = len(agents)
        fov = self.fov
        pad = fov // 2

        # 1) Gather all neighbor entries
        # for each agent, get list of neighbors and record (agent_idx, goal_idx, y,x)
        entries = []
        for ai, a in enumerate(agents):
            y0, x0 = self.starts[a]
            nbrs = self._get_neighboring_agents(a, 20)
            for nbr in nbrs:
                gidx = self._goal_index[self.goals[nbr][0]]
                entries.append((ai, gidx, y0, x0, nbr))
        torch.cuda.synchronize()
        # t1 = time.perf_counter(); print(f"[timing] build entries: {t1-t0:.6f}s")

        if not entries:
            return [[] for _ in agents], [[] for _ in agents]

        # unzip entries
        A, G, Y, X, Nbr = zip(*entries)

        A = torch.tensor(A, device=device, dtype=torch.long)
        G = torch.tensor(G, device=device, dtype=torch.long)
        Y = torch.tensor(Y, device=device, dtype=torch.long)
        X = torch.tensor(X, device=device, dtype=torch.long)
        torch.cuda.synchronize()
        # t2 = time.perf_counter(); print(f"[timing] tensorize indices: {t2-t1:.6f}s")

        # 2) extract fov patches from the single padded heuristic tensor
        # self._padded_heur: (G_total, H', W')
        # use advanced indexing
        # Instead, unfold once: (G_total, H', W') -> (G_total, H_orig, W_orig, fov, fov)
        windows = self._padded_heur_cuda.unfold(1, fov, 1).unfold(2, fov, 1)  # (G_total, H_orig, W_orig, fov, fov)
        torch.cuda.synchronize()
        # t3 = time.perf_counter(); print(f"[timing] unfold windows: {t3-t2:.6f}s")
        patches = windows[G, Y, X]  # (P, fov, fov)
        torch.cuda.synchronize()
        # t4 = time.perf_counter(); print(f"[timing] index windows: {t4-t3:.6f}s")

        # 3) mask, normalize, and reshape
        mask = patches != float('inf')
        torch.cuda.synchronize()
        # t5 = time.perf_counter(); print(f"[timing] build mask: {t5-t4:.6f}s")
        finite = torch.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
        torch.cuda.synchronize()
        # t6 = time.perf_counter(); print(f"[timing] nan_to_num: {t6-t5:.6f}s")
        maxv = finite.amax(dim=(1,2)).clamp(min=1.0).view(-1,1,1)
        torch.cuda.synchronize()
        # t7 = time.perf_counter(); print(f"[timing] compute max: {t7-t6:.6f}s")
        norm = torch.where(mask, finite / maxv, torch.tensor(1.0, device=patches.device))
        torch.cuda.synchronize()
        # t8 = time.perf_counter(); print(f"[timing] normalize: {t8-t7:.6f}s")

        patches = norm.unsqueeze(1)  # (P,1,fov,fov)

        # 4) build coords tensor
        coords = torch.tensor(
            [(self.starts[n][1] / self.size_x, self.starts[n][0] / self.size_y) for n in Nbr],
            dtype=torch.float32, device=patches.device
        )
        torch.cuda.synchronize()
        # t9 = time.perf_counter(); print(f"[timing] build coords: {t9-t8:.6f}s")

        # 5) group back into per-agent lists
        sorted_A, perm = A.sort()
        patches = patches[perm]
        coords = coords[perm]
        counts = torch.bincount(sorted_A, minlength=N).tolist()
        feats = list(torch.split(patches, counts))
        coords_list = list(torch.split(coords, counts))
        torch.cuda.synchronize()
        # t10 = time.perf_counter(); print(f"[timing] group by agent: {t10-t9:.6f}s")

        # ensure empty tensors where needed
        neighbor_features = [f if f.numel() else torch.empty((0,1,fov,fov), device=device) for f in feats]
        neighbor_coords   = [c if c.numel() else torch.empty((0,2), device=device) for c in coords_list]
        torch.cuda.synchronize()
        # t11 = time.perf_counter(); print(f"[timing] finalize lists: {t11-t10:.6f}s")

        # print(f"[timing] total: {t11-t0:.6f}s")

        return neighbor_features, neighbor_coords
    
    def _get_neighboring_agents(self, agent, k=None):
        """Return neighbor agent IDs sorted by distance to `agent`.
        Optionally keep only the top-k closest."""
        neighbors_with_d = []
        i, j = self.starts[agent]
        for a in range(self.num_agents):
            if a == agent:
                continue
            x, y = self.starts[a]
            d = self.heuristic_map[(x, y)][i, j]
            if d <= self.window_size * 2:
                neighbors_with_d.append((a, d))
        # sort by distance ascending
        neighbors_with_d.sort(key=lambda t: t[1])
        if k is not None:
            neighbors_with_d = neighbors_with_d[:k]
        return [a for a, _ in neighbors_with_d]

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