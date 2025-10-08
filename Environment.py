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
import math

# --- Heuristic profiling toggles ---
PROFILE_HEUR = False

np.random.seed(0)

class Environment:
    def __init__(self, config, logger=None, grid_map_file=None, start_loc_file=None, goal_loc_file=None, start_loc_options=None, goal_loc_options=None, device=None):

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

        # Derive heuristic_map_file from grid_map_file if possible
        if grid_map_file is not None:
            base, ext = os.path.splitext(grid_map_file)
            self._heuristic_map_file = f"{base}_heur{ext}"
            self._heuristic_map_file_path = os.path.join(root, self._heuristic_map_file)
        else:
            # If grid_map_file is None, fallback to a default name
            self._heuristic_map_file = "heuristic_map_heur.npy"
            self._heuristic_map_file_path = os.path.join(root, self._heuristic_map_file)

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
        self.torch_starts = None
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
        if os.path.exists(self._heuristic_map_file_path):
            logger.print("Environment.__init__: loading heuristic map from file took", end=' ')
            start_time = time.time()
            self.heuristic_map = np.load(self._heuristic_map_file_path, allow_pickle=True).item()
            logger.print(f"{time.time() - start_time:.2f} seconds")
        else:
            self.heuristic_map = self._get_heuristic_map()
            np.save(self._heuristic_map_file_path, self.heuristic_map)

        # Pre-pad heuristic maps as CPU torch tensors to avoid per-call padding in get_obs
        self._padded_heuristic_map = {}
        for goal, heur_arr in self.heuristic_map.items():
            # Convert to tensor and pad with inf
            t = torch.from_numpy(heur_arr.astype(np.float32))  # (H, W)
            # add batch and channel dims for F.pad, then remove them
            t_padded = F.pad(t.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), value=float('inf'))
            self._padded_heuristic_map[goal] = t_padded.squeeze(0).squeeze(0)  # (H+2pad, W+2pad)

        # ==== DHC & distance stack: load cache if available, else compute & save ====
        pad = self.fov // 2
        # Derive DHC cache path from grid_map_file, if provided
        if grid_map_file is not None:
            base, ext = os.path.splitext(grid_map_file)
            dhc_cache_path = os.path.join(root, f"{base}_DHC_heur.npz")
        else:
            dhc_cache_path = os.path.join(root, "heuristic_map_DHC_heur.npz")

        if os.path.exists(dhc_cache_path):
            logger.print(f"Environment.__init__: loading DHC+dist cache from {dhc_cache_path} took", end=' ')
            t0 = time.time()
            data = np.load(dhc_cache_path, allow_pickle=False)
            goals_list = [tuple(g) for g in data['goals']]            # (G,2)
            heur_array = data['dist'].astype(np.float32)              # (G,H,W)
            dhc_array  = data['dhc'].astype(np.float32)               # (G,4,H,W)
            logger.print(f"{time.time() - t0:.2f} seconds")
        else:
            logger.print(f"Environment.__init__: computing DHC+dist cache and saving to {dhc_cache_path} took", end=' ')
            t0 = time.time()
            # Compute DHC for each goal and also build a consistent-ordered distance stack
            goals_list = list(self.heuristic_map.keys())
            heur_array = np.stack([self.heuristic_map[g] for g in goals_list], axis=0).astype(np.float32)  # (G,H,W)

            _DHC_heur_map = {}
            H, W = self.grid_map.shape
            for goal in goals_list:
                heur_map = self.heuristic_map[goal]
                dhc = np.zeros((4, H, W), dtype=np.float32)
                for y in range(H):
                    for x in range(W):
                        if self.grid_map[y, x] == 0:
                            if y > 0 and heur_map[y-1, x] < heur_map[y, x]:
                                dhc[0, y, x] = 1
                            if y < H-1 and heur_map[y+1, x] < heur_map[y, x]:
                                dhc[1, y, x] = 1
                            if x > 0 and heur_map[y, x-1] < heur_map[y, x]:
                                dhc[2, y, x] = 1
                            if x < W-1 and heur_map[y, x+1] < heur_map[y, x]:
                                dhc[3, y, x] = 1
                _DHC_heur_map[goal] = dhc
            dhc_array = np.stack([_DHC_heur_map[g] for g in goals_list], axis=0).astype(np.float32)  # (G,4,H,W)

            logger.print(f"{time.time() - t0:.2f} seconds")

            # Save cache
            try:
                np.savez_compressed(os.path.splitext(dhc_cache_path)[0],
                                    goals=np.array(goals_list, dtype=np.int32),
                                    dist=heur_array,
                                    dhc=dhc_array)
                logger.print(f"Environment.__init__: saved DHC+dist cache to {dhc_cache_path}")
            except Exception as e:
                logger.print(f"Environment.__init__: WARNING could not save DHC cache: {e}")


        # Build padded heuristic tensor using the cached/stacked array
        self.heuristic_map_array = heur_array  # (G,H,W)
        self._heur_t = torch.from_numpy(self.heuristic_map_array)  # (G,H,W)
        self._padded_heur = F.pad(self._heur_t, (pad, pad, pad, pad), value=float('inf'))  # (G,H+2pad,W+2pad)
        self._padded_heur_cuda = self._padded_heur.to(self.device, non_blocking=True)

        # Goal index consistent with the stacked arrays
        self._goal_index = {g: i for i, g in enumerate(goals_list)}

        # Build a dense grid mapping (y,x) -> goal index for fast tensor indexing (store on device)
        H_grid, W_grid = self.grid_map.shape
        goal_grid = torch.full((H_grid, W_grid), -1, dtype=torch.long)
        for g, i in self._goal_index.items():
            gy, gx = int(g[0]), int(g[1])
            if 0 <= gy < H_grid and 0 <= gx < W_grid:
                goal_grid[gy, gx] = i
        # move to target device once so calls don't re-transfer each time
        goal_grid = goal_grid.to(self.device, non_blocking=True)
        self._goal_index_grid_t = goal_grid
        # Also keep a flattened view + width to avoid advanced indexing overhead later
        self._W_grid = int(W_grid)
        self._goal_index_grid_flat = goal_grid.view(-1).contiguous()  # (H_grid*W_grid,)

        # DHC array tensor
        self.DHC_heur_arr = dhc_array  # (G,4,H,W)
        self.DHC_heur_arr_cuda = torch.from_numpy(self.DHC_heur_arr).to(self.device, non_blocking=True)

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

    def _get_fov(self, grid_map, x, y, fov):
        padded_grid = np.pad(grid_map, pad_width=fov//2, mode='constant', constant_values=0)
        return padded_grid[x:x+fov, y:y+fov]
    
    def get_obs(self, agents):
        # --- Synchronized CUDA timing/profiling ---
        import time, torch
        def _sync():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        # _sync(); t_start = t0 = time.perf_counter()
        # GPU vectorized version using torch.nn.functional.unfold
        device = self.device
        N = len(agents)
        fov, pad = self.fov, self.fov // 2
        # Prepare coords:
        starts_tensor = torch.tensor([self.starts[a] for a in agents], device=device)
        ys, xs = starts_tensor[:,0], starts_tensor[:,1]
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after step1 load inputs: {(t1-t0)*1000:.2f}ms"); t0 = t1
        # 1) obstacles+agent occupancy
        base = self.padded_grid_map_cuda.unsqueeze(0).unsqueeze(0)  # (1,1,H',W')
        occ  = torch.zeros_like(self.padded_grid_map_cuda, device=device)
        coords_all = torch.tensor(self.starts, device=device)
        coords_all_pad = coords_all + pad
        occ[coords_all_pad[:,0], coords_all_pad[:,1]] = 1
        occ = occ.unsqueeze(0).unsqueeze(0)  # (1,1,H',W')
        all_feats = torch.cat([base, occ], dim=1)     # (1,2,H',W')
        patches = F.unfold(all_feats, kernel_size=fov).view(1, 2, fov*fov, -1)  # (1,2,fov*fov,#windows)
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after obstacles+occ unfold: {(t1-t0)*1000:.2f}ms"); t0 = t1

        # compute the correct linear indices for windows:
        Hp, Wp = self.padded_grid_map_cuda.shape      # H' , W'
        # number of valid top-left positions along width after unfold
        win_w = Wp - fov + 1
        # (with pad=fov//2, win_w == self.grid_map.shape[1], but compute it robustly)
        idxs = ys * win_w + xs                        # (N,) long
        # reshape to (1, 2, fov*fov, L) then gather
        patches = patches.view(1, 2, fov*fov, -1)     # (1,2,fov*fov,L)
        obs0    = patches[0, :, :, idxs]              # (2, fov*fov, N)
        obs0    = obs0.permute(2, 0, 1).contiguous()  # (N, 2, fov*fov)
        obs0    = obs0.view(N, 2, fov, fov)           # (N, 2, fov, fov)
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after gather obs0: {(t1-t0)*1000:.2f}ms"); t0 = t1

        # 2) heuristics: unfold and gather
        heur_windows = self._padded_heur_cuda.unsqueeze(1)  # (G,1,H',W')
        heur_patches = heur_windows.unfold(2,fov,1).unfold(3,fov,1)  # (G,1,H,W,fov,fov)
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after heur_windows.unfold: {(t1-t0)*1000:.2f}ms"); t0 = t1
        g_idxs = torch.tensor([self._goal_index[self.goals[a][0]] for a in agents], device=device)
        hpatch = heur_patches[g_idxs,0, ys, xs]  # (N,fov,fov)
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after heur gather/indexing: {(t1-t0)*1000:.2f}ms"); t0 = t1
        # mask and normalize
        mask = hpatch != float('inf')
        finite = torch.nan_to_num(hpatch, nan=0.0, posinf=0.0, neginf=0.0)
        maxv = torch.where(mask, finite, torch.tensor(0., device=finite.device)).amax(dim=(1,2)).clamp(min=1.0).view(-1,1,1)
        norm = torch.where(mask, finite / maxv, torch.tensor(1.0, device=finite.device))
        hpatch = norm
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after heur normalization: {(t1-t0)*1000:.2f}ms"); t0 = t1
        # 3) DHC: pad and unfold like heuristics, index by goal + agent pos
        # build and pad the DHC heuristic tensor on GPU
        DHC_heur = self.DHC_heur_arr_cuda  # (G,4,H,W)
        # pad height and width by `pad` on both sides
        DHC_heur = F.pad(DHC_heur, (pad, pad, pad, pad))  # now (G,4,H+2pad,W+2pad)
        # unfold spatial dims (2 -> height, 3 -> width) to extract fov×fov windows
        dhc_windows = DHC_heur.unfold(2, fov, 1).unfold(3, fov, 1)  # (G,4,H,W,fov,fov)
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after DHC unfold: {(t1-t0)*1000:.2f}ms"); t0 = t1
        dhc_patch = dhc_windows[g_idxs, :, ys, xs].to(device, non_blocking=True)  # (N,4,fov,fov)
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after DHC gather/indexing: {(t1-t0)*1000:.2f}ms"); t0 = t1
        # 4) coordinate channel: build tensor directly on GPU
        coord = torch.zeros((N,1,fov,fov), device=device)
        coord[:,0,0,0] = xs / self.size_x
        coord[:,0,0,1] = ys / self.size_y
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after coord channel: {(t1-t0)*1000:.2f}ms"); t0 = t1
        # 5) stack all layers: obstacle, occ, norm-heur, dhc (4), coord → (N,8,fov,fov)
        obs_out = torch.cat([obs0.view(N,2,fov,fov), hpatch.unsqueeze(1), dhc_patch, coord], dim=1)
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] after final stack: {(t1-t0)*1000:.2f}ms"); t0 = t1
        # _sync(); t1 = time.perf_counter(); print(f"[get_obs] TOTAL: {(t1-t_start)*1000:.2f}ms");
        return obs_out

    def get_heur_matrix(self, num_agents, starts, _goal_index, _padded_heur_cuda, window_size, k=None, device=None):
        """
        Fast path: builds an N×N distance table D[a,i] = dist from agent a's start cell
        (treated as goal) to agent i's start cell, using the pre-padded heuristic tensor.
        """
        if num_agents == 0:
            return []

        # --- profiling setup ---
        use_events = PROFILE_HEUR and (device is not None) and torch.cuda.is_available()
        if use_events:
            ev = lambda: torch.cuda.Event(enable_timing=True)
            ev_total_s = ev(); ev_total_e = ev()
            ev_coords_s = ev(); ev_coords_e = ev()
            ev_pad_s    = ev(); ev_pad_e    = ev()
            ev_gidx_s   = ev(); ev_gidx_e   = ev()
            ev_layers_s = ev(); ev_layers_e = ev()
            ev_lin_s    = ev(); ev_lin_e    = ev()
            ev_mask_s   = ev(); ev_mask_e   = ev()
            ev_total_s.record()

        # starts are (y,x) in border-padded grid coordinates (with +1 wall). The
        # heuristic tensor _padded_heur_cuda has an *additional* pad of pad=fov//2
        # filled with +inf for window extraction. We must offset by this pad when
        # indexing directly into _padded_heur_cuda.
        if use_events: ev_coords_s.record()
        ys = torch.tensor([s[0] for s in starts], device=device, dtype=torch.long)
        xs = torch.tensor([s[1] for s in starts], device=device, dtype=torch.long)
        if use_events: ev_coords_e.record()

        # Compute fov and pad for the heuristic tensor
        if use_events: ev_pad_s.record()
        window_size_int = int(window_size)
        fov = window_size_int * 4 + 1
        pad = fov // 2

        ys_p = ys + pad
        xs_p = xs + pad
        if use_events: ev_pad_e.record()

        # map each start cell → goal-layer index (still border-padded grid coords, no extra pad)
        if use_events: ev_gidx_s.record()
        try:
            gidx = torch.tensor([_goal_index[(y.item(), x.item())] for y, x in zip(ys, xs)],
                                device=device, dtype=torch.long)
        except KeyError as e:
            # Fallback to slow path if any start is missing from the index
            print(f"get_heur_matrix: WARNING start {e} not in goal index, falling back to slow path")
            if use_events: ev_gidx_e.record()
            return self.get_close_pairs(num_agents, starts, heuristic_map=None, window_size=window_size)
        if use_events: ev_gidx_e.record()

        G_total, Hp, Wp = _padded_heur_cuda.shape
        ys_cpu = [int(s[0]) + pad for s in starts]
        xs_cpu = [int(s[1]) + pad for s in starts]
        assert 0 <= min(ys_cpu) and max(ys_cpu) < Hp, (min(ys_cpu), max(ys_cpu), Hp)
        assert 0 <= min(xs_cpu) and max(xs_cpu) < Wp, (min(xs_cpu), max(xs_cpu), Wp)

        gidx_list = [int(_goal_index[(int(s[0]), int(s[1]))]) for s in starts]
        assert 0 <= min(gidx_list) and max(gidx_list) < G_total, (min(gidx_list), max(gidx_list), G_total)

        # Select only the N layers we need (one per agent goal). Shape: (N, H', W')
        if use_events: ev_layers_s.record()
        layers = _padded_heur_cuda.index_select(0, gidx)
        if use_events: ev_layers_e.record()

        # Gather distances into an (N, N) matrix using flattened gather to avoid broadcast artifacts
        # layers: (N, Hp, Wp) → layers_flat: (N, Hp*Wp)
        if use_events: ev_lin_s.record()
        Hp = layers.size(1)
        Wp = layers.size(2)
        lin_idx = (ys_p * Wp + xs_p)                      # (N,)
        layers_flat = layers.view(num_agents, -1)         # (N, Hp*Wp)
        D = torch.index_select(layers_flat, 1, lin_idx)   # (N, N)
        if use_events: ev_lin_e.record()

        # Threshold by radius r = 2*window_size (graph distance); mask out self
        r = window_size_int * 2

        diag_idx = torch.arange(num_agents, device=device)
        D[diag_idx, diag_idx] = float('inf')
        if use_events: ev_mask_s.record()
        mask = D <= r
        if use_events: ev_mask_e.record()

        N = D.size(0)
        diag = torch.arange(N, device=D.device)
        finite = torch.isfinite(D).clone()
        finite[diag, diag] = True  # allow our explicit self=inf

        if use_events:
            ev_total_e.record(); torch.cuda.synchronize()
            coords_ms = ev_coords_s.elapsed_time(ev_coords_e)
            pad_ms    = ev_pad_s.elapsed_time(ev_pad_e)
            gidx_ms   = ev_gidx_s.elapsed_time(ev_gidx_e)
            layers_ms = ev_layers_s.elapsed_time(ev_layers_e)
            lin_ms    = ev_lin_s.elapsed_time(ev_lin_e)
            mask_ms   = ev_mask_s.elapsed_time(ev_mask_e)
            total_ms  = ev_total_s.elapsed_time(ev_total_e)
            print((
                f"[heur] N={num_agents} k={k} | coords={coords_ms:.1f}ms, padcalc={pad_ms:.1f}ms, "
                f"gidx={gidx_ms:.1f}ms, layers={layers_ms:.1f}ms, lin={lin_ms:.1f}ms, "
                f"mask={mask_ms:.1f}ms, total={total_ms:.1f}ms"
            ))


        # If a top-k is requested, prune the mask to keep only the k closest per column
        if k is not None and k > 0:
            # Replace non-candidates with +inf, then take topk along columns
            D_masked = torch.where(mask, D, torch.full_like(D, float('inf')))
            k_eff = min(k, max(1, num_agents - 1))
            vals, idxs = torch.topk(D_masked, k=k_eff, dim=0, largest=False, sorted=True)
            # Build a pruned boolean mask: only k closest finite entries per column
            mask_k = torch.zeros_like(mask)
            finite_vals = torch.isfinite(vals)
            # Scatter finite flags into rows selected by idxs per column
            mask_k.scatter_(0, idxs, finite_vals)
            return D, mask_k

        return D, mask

    def get_neighbor_goal_heuristics_as_patches(self, agents):
        """
        Vectorized extraction of neighbor goal heuristic patches.
        Returns lists (per agent) of feature tensors of shape (P_i,1,fov,fov)
        and coordinate tensors of shape (P_i,2).
        """
        device = self.device

        # ---- profiling (mirrors get_heur_matrix style) ----
        use_events = PROFILE_HEUR and torch.cuda.is_available()
        if use_events:
            ev = lambda: torch.cuda.Event(enable_timing=True)
            ev_total_s = ev(); ev_total_e = ev()
            ev_hm_s    = ev(); ev_hm_e    = ev()   # get_heur_matrix
            ev_npneigh_s=ev(); ev_npneigh_e=ev()   # build neighbors_per_agent
            ev_tens_s  = ev(); ev_tens_e  = ev()   # tensorize indices
            ev_unf_s   = ev(); ev_unf_e   = ev()   # unfold windows
            ev_gath_s  = ev(); ev_gath_e  = ev()   # index windows (gather)
            ev_mask_s  = ev(); ev_mask_e  = ev()   # mask (finite)
            ev_nan_s   = ev(); ev_nan_e   = ev()   # nan_to_num
            ev_amax_s  = ev(); ev_amax_e  = ev()   # amax
            ev_norm_s  = ev(); ev_norm_e  = ev()   # normalize
            ev_coord_s = ev(); ev_coord_e = ev()   # build coords
            ev_group_s = ev(); ev_group_e = ev()   # group by agent
            ev_total_s.record()

        k = 20
        N = len(agents)
        fov = self.fov
        pad = fov // 2

        # (Optional) autocast acceleration for heavy normalization math
        use_amp = torch.cuda.is_available()

        # 1) Gather neighbor entries: (agent_idx, goal_idx, y, x, neighbor_id)
        if use_events: ev_hm_s.record()
        # D, mask = self.get_heur_matrix(self.num_agents, self.starts, self._goal_index, self._padded_heur_cuda, self.window_size, k=k, device=device)
        # D_cpu = D.cpu().numpy()
        # mask_cpu = mask.cpu().numpy()
        if use_events: ev_hm_e.record()

        # neighbors_per_agent[i] = list of neighbor agent indices within radius, sorted by distance asc
        if use_events: ev_npneigh_s.record()
        # neighbors_per_agent = [[] for _ in range(self.num_agents)]
        # for i in range(self.num_agents):
        #     cand = np.where(mask_cpu[:, i])[0]  # agents a where dist(a -> i) <= r
        #     if cand.size:
        #         dcol = D_cpu[cand, i]
        #         order = np.argsort(dcol, kind='mergesort')  # stable, ascending
        #         sorted_idx = cand[order]
        #         neighbors_per_agent[i] = [int(a) for a in sorted_idx if int(a) != i]

        #         if k is not None:
        #             neighbors_per_agent[i] = neighbors_per_agent[i][:k]

        # 1) Gather all neighbor entries
        entries = []
        for ai, a in enumerate(agents):
            y0, x0 = self.starts[a]
            nbrs = self._get_neighboring_agents(a, k=k)
            # nbrs = neighbors_per_agent[a]
            if nbrs == []:
                print(f"agent {a} has no neighbors within radius")
            for nbr in nbrs:
                gidx = self._goal_index[self.goals[nbr][0]]
                entries.append((ai, gidx, y0, x0, nbr))

        if use_events: ev_npneigh_e.record()

        if not entries:
            return [[] for _ in agents], [[] for _ in agents]

        # ---- Fast path: build all indices in bulk on CPU, one H2D copy ----
        if use_events: ev_tens_s.record()
        import numpy as _np

        # Convert lists to NumPy arrays in one go
        A_np   = _np.fromiter((e[0] for e in entries), dtype=_np.int64, count=len(entries))  # agent_idx in [0,len(agents))
        G_np   = _np.fromiter((e[1] for e in entries), dtype=_np.int64, count=len(entries))  # goal layer index
        Y_np   = _np.fromiter((e[2] for e in entries), dtype=_np.int64, count=len(entries))  # y
        X_np   = _np.fromiter((e[3] for e in entries), dtype=_np.int64, count=len(entries))  # x
        Nbr_np = _np.fromiter((e[4] for e in entries), dtype=_np.int64, count=len(entries))  # neighbor agent id

        # Stack columns to (P,4) and do a single pinned transfer
        cols_np = _np.stack([A_np, G_np, Y_np, X_np], axis=1)  # (P,4)
        cols_t  = torch.from_numpy(cols_np).to(device, non_blocking=True)
        A, G, Y, X = cols_t[:,0].long(), cols_t[:,1].long(), cols_t[:,2].long(), cols_t[:,3].long()
        # Keep Nbr on CPU for coord build; we'll transfer coords in one go below
        if use_events: ev_tens_e.record()

        # 2) extract fov patches from the (cached) padded heuristic tensor
        if use_events: ev_unf_s.record()
        if not hasattr(self, '_heur_windows_cache') or self._heur_windows_cache is None:
            self._heur_windows_cache = self._padded_heur_cuda.unfold(1, fov, 1).unfold(2, fov, 1)
        windows = self._heur_windows_cache  # (G_total, H_orig, W_orig, fov, fov)
        if use_events: ev_unf_e.record()
        if use_events: ev_gath_s.record()
        with torch.cuda.amp.autocast(enabled=use_amp):
            patches = windows[G, Y, X]  # (P, fov, fov)
        if use_events: ev_gath_e.record()

        # 3) mask, normalize, and reshape (in-place & memory-light)
        if use_events: ev_mask_s.record()
        mask = torch.isfinite(patches)  # True where finite, False for inf/nan
        if use_events: ev_mask_e.record()

        # Zero-out non-finite in-place to compute max correctly
        # (keep a copy of mask for later restore-to-1)
        if use_events: ev_nan_s.record()
        patches = patches.clone(memory_format=torch.contiguous_format)
        patches[~mask] = 0
        if use_events: ev_nan_e.record()

        # Compute per-patch max; clamp to avoid divide-by-zero
        if use_events: ev_amax_s.record()
        maxv = patches.amax(dim=(1, 2))
        maxv.clamp_(min=1.0)
        maxv = maxv.view(-1, 1, 1)
        if use_events: ev_amax_e.record()

        # Normalize in-place, then restore masked (non-finite) entries to 1.0
        if use_events: ev_norm_s.record()
        patches.mul_(1.0).div_(maxv)
        patches[~mask] = 1.0
        if use_events: ev_norm_e.record()

        patches = patches.unsqueeze(1)  # (P,1,fov,fov)

        # 4) build coords tensor (vectorized via NumPy)
        if use_events: ev_coord_s.record()
        starts_np = _np.asarray(self.starts, dtype=_np.int64)  # (N_agents,2) as (y,x)
        # Gather neighbor coords and normalize on CPU
        neigh_xy = starts_np[Nbr_np]  # (P,2) [y,x]
        coords_np = _np.stack([
            neigh_xy[:,1] / float(self.size_x),
            neigh_xy[:,0] / float(self.size_y)
        ], axis=1).astype(_np.float32)  # (P,2)
        coords = torch.from_numpy(coords_np).to(patches.device, non_blocking=True)
        if use_events: ev_coord_e.record()

        # 5) group back into per-agent lists
        if use_events: ev_group_s.record()
        sorted_A, perm = A.sort()
        patches = patches[perm]
        coords = coords[perm]
        counts = torch.bincount(sorted_A, minlength=N).tolist()
        feats = list(torch.split(patches, counts))
        coords_list = list(torch.split(coords, counts))
        if use_events: ev_group_e.record()

        # ensure empty tensors where needed
        neighbor_features = [f if f.numel() else torch.empty((0,1,fov,fov), device=device) for f in feats]
        neighbor_coords   = [c if c.numel() else torch.empty((0,2), device=device) for c in coords_list]

        if use_events:
            ev_total_e.record(); torch.cuda.synchronize()
            print((
                f"[neigh heur] N={self.num_agents} k={k} | getHM={ev_hm_s.elapsed_time(ev_hm_e):.1f}ms, "
                f"buildNeigh={ev_npneigh_s.elapsed_time(ev_npneigh_e):.1f}ms, tens={ev_tens_s.elapsed_time(ev_tens_e):.1f}ms, "
                f"unfold={ev_unf_s.elapsed_time(ev_unf_e):.1f}ms, gather={ev_gath_s.elapsed_time(ev_gath_e):.1f}ms, "
                f"mask={ev_mask_s.elapsed_time(ev_mask_e):.1f}ms, nan2num={ev_nan_s.elapsed_time(ev_nan_e):.1f}ms, "
                f"amax={ev_amax_s.elapsed_time(ev_amax_e):.1f}ms, norm={ev_norm_s.elapsed_time(ev_norm_e):.1f}ms, "
                f"coords={ev_coord_s.elapsed_time(ev_coord_e):.1f}ms, group={ev_group_s.elapsed_time(ev_group_e):.1f}ms, "
                f"total={ev_total_s.elapsed_time(ev_total_e):.1f}ms"
            ))

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


    def get_close_pairs_fast(self, k=None):
        """
        Returns pairs (i, a) where D[a,i] <= 2*window_size and a>i. Optional top-k per i.
        """
        D, mask = self.get_heur_matrix(self.num_agents, self.starts, self._goal_index, self._padded_heur_cuda, self.window_size, k=None, device=self.device)

        if k is not None and k > 0:
            # Keep top-k closest neighbors per column (per target agent i)
            # Replace non-candidates with +inf then take topk with largest=False
            D_masked = torch.where(mask, D, torch.full_like(D, float('inf')))
            k_eff = min(k, max(1, self.num_agents - 1))
            vals, idxs = torch.topk(D_masked, k=k_eff, dim=0, largest=False, sorted=True)
            # Build pairs (i, a) from columns; filter +inf
            pairs = []
            for i in range(self.num_agents):
                for j in range(k_eff):
                    a = idxs[j, i].item()
                    if math.isfinite(float(vals[j, i].item())) and a > i:
                        pairs.append((i, a))
            return pairs
        else:
            # No k: take *all* within radius, ordered by ascending distance per i
            pairs = []
            D_cpu = D.cpu().numpy()
            mask_cpu = mask.cpu().numpy()

            for i in range(self.num_agents):
                cand = np.where(mask_cpu[:, i])[0]

                if cand.size:
                    # sort by distance ascending
                    order = np.argsort(D_cpu[cand, i], kind='mergesort')
                    for a in cand[order]:
                        if a > i:
                            pairs.append((i, int(a)))
            return pairs

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