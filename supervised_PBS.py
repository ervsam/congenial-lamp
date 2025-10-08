import os
import sys
import pdb
import glob
import random
import pickle
import time
import yaml
from collections import Counter, defaultdict
import ast
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
import math
import argparse
from sklearn.metrics import confusion_matrix
import csv

from Environment import Environment
from Model import QNetwork
from utils import Logger

# --- Collate profiling toggles ---
PROFILE_COLLATE = False
PROFILE_EVERY   = 1
_COLLATE_CALLS  = 0

class EMA:
    """Exponential Moving Average with shape-safety for lazy params.
    Handles modules like nn.LazyConv2d where parameter shapes materialize after first forward.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # Do not pre-populate with possibly uninitialized shapes.
        for name, param in model.named_parameters():
            if param.requires_grad and param.data is not None:
                try:
                    # Some lazy params may have shape torch.Size([]) or 0-sized before first forward.
                    _ = param.data.shape
                    if param.data.numel() > 0:
                        self.shadow[name] = param.data.detach().clone()
                except Exception:
                    pass

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow or self.shadow[name].shape != param.data.shape:
                # (Re)initialize shadow to current param shape
                self.shadow[name] = param.data.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=(1.0 - self.decay))

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Only swap if we have a matching-shaped shadow
            shadow_t = self.shadow.get(name, None)
            if shadow_t is None or shadow_t.shape != param.data.shape:
                # Initialize shadow to current param so evaluation can proceed safely
                self.shadow[name] = param.data.detach().clone()
                shadow_t = self.shadow[name]
            self.backup[name] = param.data.detach().clone()
            param.data = shadow_t.detach().clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.data = self.backup[name].detach().clone()
        self.backup = {}

# --- Focal Cross-Entropy Loss for multi-class ---
class FocalCE(nn.Module):
    """Multi-class focal cross-entropy for logits of shape [B, C].
    L = - alpha_y * (1 - p_y)^gamma * log p_y
    - gamma >= 0 controls down-weighting of easy examples
    - alpha can be None or a list/tuple of per-class weights (len == C)
    """
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is None:
            self.register_buffer('alpha', None)
        else:
            a = torch.tensor(alpha, dtype=torch.float32)
            # avoid zero-sum; do not normalize forcibly, trust user input
            self.register_buffer('alpha', a)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        logp = F.log_softmax(logits, dim=1)
        p    = logp.exp()
        idx  = torch.arange(logits.size(0), device=logits.device)
        pt   = p[idx, targets]
        fl   = - (1.0 - pt).clamp(min=1e-12).pow(self.gamma) * logp[idx, targets]
        if self.alpha is not None:
            a = self.alpha.to(logits.device)
            # if alpha length mismatched, fall back to no-alpha
            if a.numel() == logits.size(1):
                fl = a[targets] * fl
        return fl.mean()



class PairDataset(Dataset):
    def __init__(self, data_txt, env, undersample=True, use_hardneg=False, hardneg_radius=5, hardneg_frac=0.5):
        self.env   = env
        self.pairs = []      # will hold (line_idx, a, b, label)
        self.raw_data = []
        self.use_hardneg = use_hardneg
        self.hardneg_radius = int(hardneg_radius)
        self.hardneg_frac = float(hardneg_frac)
        # indices of class-2 pairs that are near ("hard negatives")
        self._class2_hard = []

        row, col = self.env.grid_map.shape
        row = row - 2
        col = col - 2

        # ---- Profiling accumulators ----
        prof_counts = 0
        prof_sum = {
            'split': 0.0,
            'ast': 0.0,
            'idx2coord': 0.0,
            'prio': 0.0,
            'set_starts': 0.0,
            'get_pairs': 0.0,
            'label_loop': 0.0,
            'total': 0.0,
        }

        with open(data_txt) as f:
            print(f"Reading from {data_txt}...")
            for line_idx, line in tqdm(enumerate(f)):
                t0 = time.perf_counter()

                # split
                t = time.perf_counter()
                starts_str, goals_str, priorities_str = line.split(';')
                prof_sum['split'] += (time.perf_counter() - t)

                # parse AST
                t = time.perf_counter()
                starts = ast.literal_eval(starts_str)
                goals = ast.literal_eval(goals_str)
                priorities = ast.literal_eval(priorities_str.replace(': [', ':['))
                prof_sum['ast'] += (time.perf_counter() - t)

                # revert from idx to coord
                t = time.perf_counter()
                starts = [(start//col + 1, start%col + 1) for start in starts]
                goals = [[(g//col + 1, g%col + 1) for g in goal] for goal in goals]
                prof_sum['idx2coord'] += (time.perf_counter() - t)

                # build partial priorities list
                t = time.perf_counter()
                partial_prio = []
                for low, highs in priorities.items():
                    for high in highs:
                        partial_prio.append((high, low))
                prof_sum['prio'] += (time.perf_counter() - t)

                # cache raw episode
                self.raw_data.append((starts, goals, partial_prio))

                # set env.starts
                t = time.perf_counter()
                self.env.starts = starts
                prof_sum['set_starts'] += (time.perf_counter() - t)

                # get close pairs
                t = time.perf_counter()
                close_pairs = self.env.get_close_pairs_fast()
                prof_sum['get_pairs'] += (time.perf_counter() - t)

                # ---- Vectorized labeling & hard-negative marking ----
                t = time.perf_counter()
                P0 = len(self.pairs)

                if len(close_pairs) > 0:
                    # Build label matrix L (N x N), default 2 (no prio)
                    Nloc = len(starts)
                    L = np.full((Nloc, Nloc), 2, dtype=np.uint8)
                    # partial_prio stores (high, low): label 0 → (a,b) means a before b
                    for (hi, lo) in partial_prio:
                        if 0 <= hi < Nloc and 0 <= lo < Nloc:
                            L[hi, lo] = 0
                            L[lo, hi] = 1  # opposite direction

                    pairs_np = np.asarray(close_pairs, dtype=np.int64)  # (P,2) with columns (a,b)
                    a_idx = pairs_np[:, 0]
                    b_idx = pairs_np[:, 1]

                    labels_np = L[a_idx, b_idx]

                    # Manhattan distance per pair (vectorized)
                    starts_np = np.asarray(starts, dtype=np.int32)  # (N,2) as (y,x)
                    ya = starts_np[a_idx, 0]; xa = starts_np[a_idx, 1]
                    yb = starts_np[b_idx, 0]; xb = starts_np[b_idx, 1]
                    manh = np.abs(yb - ya) + np.abs(xb - xa)

                    # Hard-negatives: class 2 and within radius
                    hn_mask = (labels_np == 2) & (manh <= int(self.hardneg_radius))
                    if np.any(hn_mask):
                        hn_positions = np.nonzero(hn_mask)[0]
                        # Indices in self.pairs after extension will be P0 .. P0+P-1
                        self._class2_hard.extend((P0 + hn_positions).tolist())

                    # Bulk-extend pairs list
                    self.pairs.extend([(line_idx, int(a), int(b), int(lbl))
                                       for a, b, lbl in zip(a_idx.tolist(), b_idx.tolist(), labels_np.tolist())])
                prof_sum['label_loop'] += (time.perf_counter() - t)

                prof_sum['total'] += (time.perf_counter() - t0)
                prof_counts += 1

                # Optional: print every 50 episodes
                if prof_counts % 100000000 == 0:
                    avg = {k: (v/max(1,prof_counts))*1000.0 for k,v in prof_sum.items()}
                    print((
                        f"[pairdata prof @{prof_counts}] split={avg['split']:.2f}ms, ast={avg['ast']:.2f}ms, "
                        f"idx2coord={avg['idx2coord']:.2f}ms, prio={avg['prio']:.2f}ms, set_starts={avg['set_starts']:.2f}ms, "
                        f"get_pairs={avg['get_pairs']:.2f}ms, label_loop={avg['label_loop']:.2f}ms | total={avg['total']:.2f}ms/ep"
                    ))

        if prof_counts > 0:
            avg = {k: (v/prof_counts)*1000.0 for k,v in prof_sum.items()}
            print((
                f"[pairdata prof FINAL N={prof_counts}] split={avg['split']:.2f}ms, ast={avg['ast']:.2f}ms, "
                f"idx2coord={avg['idx2coord']:.2f}ms, prio={avg['prio']:.2f}ms, set_starts={avg['set_starts']:.2f}ms, "
                f"get_pairs={avg['get_pairs']:.2f}ms, label_loop={avg['label_loop']:.2f}ms | total={avg['total']:.2f}ms/ep"
            ))

        # Print class counts before undersampling
        pre_counts = Counter([lbl for (_, _, _, lbl) in self.pairs])
        print(f"PairDataset: class counts before undersampling: {{0}}={pre_counts[0]}, {{1}}={pre_counts[1]}, {{2}}={pre_counts[2]}")

        if undersample:
            # --- undersample to smallest class count ---
            label_counts = Counter([label for (_, _, _, label) in self.pairs])
            min_count = min(label_counts.values())
            # group indices by label
            indices_by_label = {0: [], 1: [], 2: []}
            for idx, (_, _, _, lbl) in enumerate(self.pairs):
                indices_by_label[lbl].append(idx)

            if not self.use_hardneg:
                # uniform undersample as before
                selected_indices = []
                for lbl, idxs in indices_by_label.items():
                    selected_indices.extend(random.sample(idxs, min_count))
            else:
                # hard-negative mix for class 2
                hard2 = set(self._class2_hard)
                class2_all = indices_by_label[2]
                class2_hard = [i for i in class2_all if i in hard2]
                class2_easy = [i for i in class2_all if i not in hard2]
                nhard_target = min(int(round(self.hardneg_frac * min_count)), len(class2_hard))
                neasy_target = max(0, min_count - nhard_target)
                selected_indices = []
                selected_indices.extend(random.sample(indices_by_label[0], min_count))
                selected_indices.extend(random.sample(indices_by_label[1], min_count))
                if nhard_target > 0 and len(class2_hard) > 0:
                    selected_indices.extend(random.sample(class2_hard, nhard_target))
                if neasy_target > 0 and len(class2_easy) > 0:
                    neasy_target = min(neasy_target, len(class2_easy))
                    selected_indices.extend(random.sample(class2_easy, neasy_target))

            # rebuild pairs to undersampled set
            self.pairs = [self.pairs[i] for i in selected_indices]
            # Print class counts after undersampling
            post_counts = Counter([lbl for (_, _, _, lbl) in self.pairs])
            if self.use_hardneg:
                # recount hard negatives in the sampled set
                sampled_hard2 = 0
                for idx, (li, a, b, lbl) in enumerate(self.pairs):
                    if lbl != 2:
                        continue
                    ya, xa = self.raw_data[li][0][a]
                    yb, xb = self.raw_data[li][0][b]
                    if abs(yb - ya) + abs(xb - xa) <= self.hardneg_radius:
                        sampled_hard2 += 1
                print(f"PairDataset(HN): class2 hard={sampled_hard2}/{post_counts[2]} (radius={self.hardneg_radius}, frac={self.hardneg_frac})")
            print(f"PairDataset: class counts after undersampling: {{0}}={post_counts[0]}, {{1}}={post_counts[1]}, {{2}}={post_counts[2]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t0 = time.perf_counter()
        t_prev = t0
        line_idx, a, b, label = self.pairs[idx]
        t1 = time.perf_counter()
        # print(f"__getitem__ load_pair: {t1 - t_prev:.6f}s")
        t_prev = t1
        starts, goals, priorities = self.raw_data[line_idx]
        t2 = time.perf_counter()
        # print(f"__getitem__ load_raw_data: {t2 - t_prev:.6f}s")
        
        # print(f"__getitem__ total: {time.perf_counter() - t0:.6f}s")
        return (line_idx, starts, goals, (a, b), label)

def custom_collate(batch, env):
    """Collate by grouping items sharing the same line_idx (episode).
    For each episode, compute get_obs() and get_neighbor_goal_heuristics_as_patches()
    once over the unique agents participating in that episode's pairs, then
    assemble per-pair tensors in the ORIGINAL batch order.
    """

    # helper: safe CUDA sync (no-op on CPU)
    def _sync():
        try:
            if isinstance(env.device, torch.device) and env.device.type == 'cuda':
                torch.cuda.synchronize(env.device)
            else:
                torch.cuda.synchronize()
        except Exception:
            pass

    t0 = time.perf_counter()
    # accumulators
    t_unpack = 0.0
    t_group  = 0.0
    t_getobs = 0.0
    t_getnb  = 0.0
    t_assem  = 0.0
    t_pad    = 0.0
    ep_stats = []  # list of (agents_in_ep, pairs_in_ep, max_nb_ep)

    # Unpack batch tuples: (line_idx, starts, goals, (a,b), label)
    line_idx, starts_list, goals_list, agent_pairs, labels = zip(*batch)
    B_total = len(agent_pairs)
    t_unpack = time.perf_counter() - t0
    t1 = time.perf_counter()

    # Labels (preserve original order)
    labels_batch = torch.tensor(labels, dtype=torch.long)

    # Group indices by episode (line_idx) while preserving insertion order
    groups = defaultdict(list)  # line_idx -> list of indices within this batch
    for i, li in enumerate(line_idx):
        groups[li].append(i)

    t_group = time.perf_counter() - t1
    t2 = time.perf_counter()

    # Preallocate containers in ORIGINAL order
    obs_chunks  = [None] * B_total                # each slot: (2, C, fov, fov)
    neigh_flat  = [None] * (2 * B_total)          # slot 2*i and 2*i+1 for pair i
    coords_flat = [None] * (2 * B_total)          # same indexing as neigh_flat
    dists = [None] * B_total  # per-pair L1 distance (|dx|+|dy|)
    dx_list = [None] * B_total  # integer dx per pair (xb - xa)
    dy_list = [None] * B_total  # integer dy per pair (yb - ya)

    # Process one episode at a time (to avoid recomputation), but fill by original indices
    for li, idxs in groups.items():
        # episode data (identical for all items with this line_idx)
        k0 = idxs[0]
        starts = starts_list[k0]
        goals  = goals_list[k0]

        # All pairs in this episode, keep their original batch order
        ep_pairs = [agent_pairs[i] for i in idxs]

        # Unique agents involved in this episode's pairs (sorted for deterministic mapping)
        ep_agents = sorted({a for (a, b) in ep_pairs} | {b for (a, b) in ep_pairs})
        agent_to_pos = {a: j for j, a in enumerate(ep_agents)}

        # If Environment is stateful
        env.starts = starts
        env.torch_starts = torch.tensor(starts, dtype=torch.long, device=env.device)
        env.goals  = goals

        # 1) Compute per-agent observations ONCE for this episode
        _sync(); t_obs0 = time.perf_counter()
        obs_K = env.get_obs(ep_agents)
        _sync(); t_obs1 = time.perf_counter()

        if obs_K.device != env.device:
            t_mv0 = time.perf_counter(); _sync()
            obs_K = obs_K.to(env.device, non_blocking=True)
            _sync(); t_mv1 = time.perf_counter()
            t_getobs += (t_obs1 - t_obs0) + (t_mv1 - t_mv0)
        else:
            t_getobs += (t_obs1 - t_obs0)

        # 2) Compute per-agent neighbor heuristic patches ONCE for this episode
        _sync(); t_nb0 = time.perf_counter()
        nf_list, nc_list = env.get_neighbor_goal_heuristics_as_patches(ep_agents)
        _sync(); t_nb1 = time.perf_counter()
        t_getnb += (t_nb1 - t_nb0)
        # nf_list / nc_list are lists of length K with tensors for each agent

        # 3) Fill slots for each pair by its original batch index
        t_as0 = time.perf_counter()
        for i in idxs:
            a, b = agent_pairs[i]
            ia = agent_to_pos[a]
            ib = agent_to_pos[b]
            # observations for the pair → (2, C, fov, fov)
            pair_obs = torch.stack([obs_K[ia], obs_K[ib]], dim=0)

            # compute L1 distance between agents a and b
            ya, xa = starts[a]
            yb, xb = starts[b]
            dy = (yb - ya)
            dx = (xb - xa)
            l1 = abs(dy) + abs(dx)
            dists[i] = l1
            dx_list[i] = dx
            dy_list[i] = dy

            obs_chunks[i] = pair_obs
            # neighbor features/coords for A then B (keeps alignment with obs_pairs)
            neigh_flat[2 * i]     = nf_list[ia]
            neigh_flat[2 * i + 1] = nf_list[ib]
            coords_flat[2 * i]     = nc_list[ia]
            coords_flat[2 * i + 1] = nc_list[ib]
        t_assem += time.perf_counter() - t_as0
        _sync()
        ep_stats.append((len(ep_agents), len(idxs)))

    # Sanity: ensure all slots filled
    # (avoids silent misalignment if a bug slips in)
    assert all(x is not None for x in obs_chunks), "obs_chunks has unfilled slots"
    assert all(x is not None for x in neigh_flat), "neigh_flat has unfilled slots"
    assert all(x is not None for x in coords_flat), "coords_flat has unfilled slots"
    assert all(x is not None for x in dists), "dists has unfilled slots"

    # ---- Stack obs for all pairs in ORIGINAL order: (B,2,C,fov,fov) ----
    obs_batch = torch.stack(obs_chunks, dim=0)
    B    = obs_batch.size(0)
    C    = obs_batch.size(2)
    fov  = obs_batch.size(-1)

    # ---- Pad neighbors for all 2*B agents, then build mask ----
    _sync(); t_pad0 = time.perf_counter()
    if len(neigh_flat) == 0:
        device = env.device
        max_nb = 0
        padded_feats  = torch.zeros((2*B, 0, 1, fov, fov), device=device)
        padded_coords = torch.zeros((2*B, 0, 2), device=device)
        mask_flat     = torch.ones((2*B, 0), dtype=torch.bool, device=device)
    else:
        # find first real tensor to get device/dtype (handles possible empty tensors)
        first_t = next((t for t in neigh_flat if isinstance(t, torch.Tensor)), None)
        device = first_t.device if first_t is not None else env.device
        padded_feats  = pad_sequence(neigh_flat,  batch_first=True, padding_value=0.0)
        padded_coords = pad_sequence(coords_flat, batch_first=True, padding_value=0.0)
        # mask = True where padded rows (all zeros in feat patch)
        row_sum  = padded_feats.abs().sum(dim=(2, 3, 4))
        mask_flat = (row_sum == 0)
    _sync(); t_pad = time.perf_counter() - t_pad0

    max_nb       = padded_feats.size(1) if padded_feats.dim() > 1 else 0
    neigh_batch  = padded_feats.view(B, 2, max_nb, 1, fov, fov)
    coords_batch = padded_coords.view(B, 2, max_nb, 2)
    mask_batch   = mask_flat.view(B, 2, max_nb)

    dists_batch = torch.tensor(dists, dtype=torch.long)
    dx_batch = torch.tensor(dx_list, dtype=torch.long)
    dy_batch = torch.tensor(dy_list, dtype=torch.long)

    _sync(); t_total = time.perf_counter() - t0

    if PROFILE_COLLATE:
        global _COLLATE_CALLS
        _COLLATE_CALLS += 1
        if _COLLATE_CALLS % PROFILE_EVERY == 0:
            # Aggregate episode stats
            num_eps = len(set(line_idx))
            agents_sum = sum(a for (a, _) in ep_stats) if ep_stats else 0
            pairs_sum  = sum(p for (_, p) in ep_stats) if ep_stats else 0
            max_nb_val = padded_feats.size(1) if 'padded_feats' in locals() and padded_feats.dim() > 1 else 0
            print((
                f"[collate] B={B_total} eps={num_eps} agents_in_ep_sum={agents_sum} pairs_in_ep_sum={pairs_sum} "
                f"max_nb={max_nb_val} | times: unpack={t_unpack*1000:.1f}ms, group={t_group*1000:.1f}ms, "
                f"get_obs={t_getobs*1000:.1f}ms, get_neigh={t_getnb*1000:.1f}ms, assemble={t_assem*1000:.1f}ms, "
                f"pad={t_pad*1000:.1f}ms, total={t_total*1000:.1f}ms"
            ))

    return obs_batch, neigh_batch, coords_batch, labels_batch, mask_batch, dists_batch, dx_batch, dy_batch

# --- Training Loop ---
def train_on_dataset(env, model, optimizer, criterion, BATCH_SIZE, train_epochs, writer, model_file, device, sample_file=None, mode="auto", ema_decay=0.999, use_stability=True, use_hardneg=False, hardneg_radius=5, hardneg_frac=0.5):
    """
    mode:
      - "stacked"  : expects model(...) -> (_, bin_logits[B], dir_logits[B,2])
      - "threeway" : expects model(...) -> (.., class_logits[B,3]) or just class_logits[B,3]
      - "auto"     : infer from model outputs each step
    """
    # Allow criterion to be either a tuple (BCE, CE) for stacked OR a single CE for threeway
    is_tuple_crit = isinstance(criterion, (tuple, list)) and len(criterion) == 2
    if is_tuple_crit:
        bce_loss, dir_loss = criterion
    else:
        ce_loss = criterion  # CrossEntropy for threeway

    data = PairDataset(sample_file+'data.txt', env, undersample=True, use_hardneg=use_hardneg, hardneg_radius=hardneg_radius, hardneg_frac=hardneg_frac)
    balanced_loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch, env=env: custom_collate(batch, env),
        shuffle=True,
        # num_workers=8,
        # pin_memory=True,
        # persistent_workers=True
    )
    print(f"Number of batches: {len(balanced_loader)}")
    print()

    test_data = PairDataset(sample_file+'test_data.txt', env, undersample=True, use_hardneg=False)
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch, env=env: custom_collate(batch, env),
        shuffle=False,
        # pin_memory=True,
    )

    # --- Append dataset sizes to run_config.yaml ---
    out_dir = os.path.dirname(model_file)
    try:
        with open(os.path.join(out_dir, "run_config.yaml"), "a") as f:
            yaml.safe_dump({
                "num_train_samples": len(data),
                "num_test_samples": len(test_data),
            }, f, default_flow_style=False)
    except Exception as e:
        print(f"Warning: Could not append dataset sizes to run_config.yaml: {e}")

    # --- Scheduler (warmup + cosine) and EMA (optional) ---
    steps_per_epoch = max(1, len(balanced_loader))
    total_steps = train_epochs * steps_per_epoch
    global_step = 0

    if use_stability:
        warmup_steps = max(1, min(1000, total_steps // 10))  # up to 1k, or 10% of total
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        ema = EMA(model, decay=ema_decay)
    else:
        # identity scheduler; no EMA
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        ema = None

    model.train()
    best_acc = 0
    # --- Early stopping on validation (test) accuracy ---
    patience = 3
    no_improve = 0
    best_val_acc = -1.0
    for epoch in range(train_epochs):
        print(f"Training epoch {epoch + 1}/{train_epochs}...")
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        # storage for epoch-level metrics, depending on mode
        all_label_tensors = []
        all_bin_logits, all_dir_logits = [], []  # for stacked
        all_tri_logits = []                      # for threeway

        for batch_n, batch in tqdm(enumerate(balanced_loader)):
            obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, labels_batch, mask, dists_batch, dx_batch, dy_batch = batch

            # Transfers
            obs_fovs_batch = obs_fovs_batch.to(device, non_blocking=True)
            neighbor_features_batch = neighbor_features_batch.to(device, non_blocking=True)
            neigh_coords_batch = neigh_coords_batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            dx_batch = dx_batch.to(device, non_blocking=True)
            dy_batch = dy_batch.to(device, non_blocking=True)

            # --- Build pair_pointer based on pointer mode ---
            pointer_mode = os.environ.get('RHCR_POINTER_MODE', 'none')
            try:
                pointer_mode = getattr(args, 'pointer_mode', pointer_mode)
            except Exception:
                pass
            pointer_mode = str(pointer_mode).lower()

            pair_pointer = None
            if pointer_mode != 'none':
                Bp = obs_fovs_batch.size(0)
                fov_size  = obs_fovs_batch.size(-1)
                cen = fov_size // 2
                # split per-side images: (B,2,C,F,F) -> A: (B,C,F,F), Bim: (B,C,F,F)
                Aimg_full = obs_fovs_batch[:, 0]
                Bimg_full = obs_fovs_batch[:, 1]

                # Strip the coordinate channel (detected by two nonzeros at [0,0] and [0,1])
                def strip_coord(t):
                    first = t[0]  # (C,F,F)
                    C_in  = first.size(0)
                    nz = (first != 0).view(C_in, -1).sum(dim=1)
                    coord_idx = None
                    for c in range(C_in):
                        if nz[c].item() == 2 and bool(first[c,0,0] != 0) and bool(first[c,0,1] != 0):
                            coord_idx = c; break
                    if coord_idx is None:
                        coord_idx = C_in - 1
                    if coord_idx == 0:
                        return t[:, 1:]
                    elif coord_idx == C_in - 1:
                        return t[:, :-1]
                    else:
                        return torch.cat([t[:, :coord_idx], t[:, coord_idx+1:]], dim=1)

                Aimg = strip_coord(Aimg_full)
                Bimg = strip_coord(Bimg_full)

                dx = dx_batch
                dy = dy_batch
                y_ab = (cen + dy).clamp(0, F-1)
                x_ab = (cen + dx).clamp(0, F-1)
                y_ba = (cen - dy).clamp(0, F-1)
                x_ba = (cen - dx).clamp(0, F-1)

                if pointer_mode == 'raw':
                    rng = torch.arange(Aimg.size(0), device=Aimg.device)
                    f_a_at_b = Aimg[rng, :, y_ab, x_ab]
                    f_b_at_a = Bimg[rng, :, y_ba, x_ba]
                elif pointer_mode == 'trunk':
                    # Encode trunk features for both sides in one call
                    flat_agents = torch.cat([Aimg, Bimg], dim=0)  # (2B, C_in, F, F)
                    trunk_maps = model.encode_trunk(flat_agents)
                    Fa = trunk_maps[:Bp]
                    Fb = trunk_maps[Bp:]
                    rng = torch.arange(Fa.size(0), device=Fa.device)
                    f_a_at_b = Fa[rng, :, y_ab, x_ab]
                    f_b_at_a = Fb[rng, :, y_ba, x_ba]
                else:
                    f_a_at_b = None; f_b_at_a = None

                size_y_t = float(env.size_y)
                size_x_t = float(env.size_x)
                dxn = dx.float() / max(1.0, size_x_t)
                dyn = dy.float() / max(1.0, size_y_t)
                dist_l1 = (dx.abs() + dy.abs()).float()

                ptr_A = torch.cat([f_a_at_b, dxn.unsqueeze(1), dyn.unsqueeze(1), dist_l1.unsqueeze(1)], dim=1)  # (B, C_ptr+3)
                ptr_B = torch.cat([f_b_at_a, (-dxn).unsqueeze(1), (-dyn).unsqueeze(1), dist_l1.unsqueeze(1)], dim=1)
                pair_pointer = torch.stack([ptr_A, ptr_B], dim=1)  # (B,2,D_ptr)

            # Forward
            outputs = model(obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, mask, pair_pointer=pair_pointer)

            # Infer or respect mode
            effective_mode = mode
            # try to infer if auto
            if mode == "auto":
                if isinstance(outputs, tuple):
                    # stacked usually returns exactly 3: (enc, bin_logits, dir_logits)
                    if len(outputs) == 3 and outputs[1].dim() == 1 and outputs[2].dim() == 2:
                        effective_mode = "stacked"
                    else:
                        # treat any tuple with last element [B,3] as threeway
                        last = outputs[-1]
                        if isinstance(last, torch.Tensor) and last.dim() == 2 and last.size(-1) == 3:
                            effective_mode = "threeway"
                        else:
                            # fallback: stacked if bin-like
                            effective_mode = "stacked"
                else:
                    # single tensor -> assume threeway [B,3]
                    effective_mode = "threeway"

            if effective_mode == "stacked":
                # Expect (_, bin_logits[B], dir_logits[B,2])
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    _, bin_logits, dir_logits = outputs[-3], outputs[-2], outputs[-1]
                else:
                    raise RuntimeError("Mode 'stacked' selected but model did not return (_, bin_logits, dir_logits).")

                # Labels
                bin_labels = (labels_batch < 2).float()       # [B]
                dir_labels = labels_batch.clone()
                dir_labels[dir_labels == 2] = 0               # dummy for CE (ignored by mask)

                # Loss
                if not is_tuple_crit:
                    raise RuntimeError("For 'stacked' mode, criterion must be (BCEWithLogitsLoss, CrossEntropyLoss).")
                loss_bin = bce_loss(bin_logits, bin_labels)
                mask_prio = bin_labels.bool()
                if mask_prio.any():
                    loss_dir = dir_loss(dir_logits[mask_prio], dir_labels[mask_prio])
                else:
                    loss_dir = torch.tensor(0.0, device=device)
                loss = loss_bin + loss_dir

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                if use_stability:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update(model)
                # Log LR & grad-norm
                try:
                    lr0 = scheduler.get_last_lr()[0]
                except Exception:
                    lr0 = optimizer.param_groups[0].get('lr', 0.0)
                writer.add_scalar('LR', lr0, global_step)
                global_step += 1

                # Accumulate
                total_loss += loss * labels_batch.size(0)
                all_bin_logits.append(bin_logits.detach().cpu())
                all_dir_logits.append(dir_logits.detach().cpu())
                all_label_tensors.append(labels_batch.detach().cpu())

            elif effective_mode == "threeway":
                # Accept either (enc, class_logits) or class_logits directly
                if isinstance(outputs, tuple):
                    class_logits = outputs[-1]
                    if not (isinstance(class_logits, torch.Tensor) and class_logits.dim() == 2 and class_logits.size(-1) == 3):
                        raise RuntimeError("Mode 'threeway' expects last output to be class logits of shape [B,3].")
                else:
                    class_logits = outputs
                    if not (class_logits.dim() == 2 and class_logits.size(-1) == 3):
                        raise RuntimeError("Mode 'threeway' expects logits of shape [B,3].")

                loss = ce_loss(class_logits, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                if use_stability:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    try:
                        gn = total_norm.item()
                    except Exception:
                        gn = float(total_norm)
                    writer.add_scalar('GradNorm/preclip', gn, global_step)
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update(model)
                # Log LR
                try:
                    lr0 = scheduler.get_last_lr()[0]
                except Exception:
                    lr0 = optimizer.param_groups[0].get('lr', 0.0)
                writer.add_scalar('LR', lr0, global_step)
                global_step += 1

                total_loss += loss * labels_batch.size(0)
                all_tri_logits.append(class_logits.detach().cpu())
                all_label_tensors.append(labels_batch.detach().cpu())
            else:
                raise ValueError(f"Unknown training mode '{mode}'.")

        # ---------------- Epoch metrics ----------------
        labels_cat = torch.cat(all_label_tensors, dim=0)

        if len(all_tri_logits) > 0 and len(all_bin_logits) == 0:
            # threeway epoch
            tri_logits_cat = torch.cat(all_tri_logits, dim=0)
            preds_cat = torch.argmax(tri_logits_cat.to(device), dim=1)
        else:
            # stacked epoch
            bin_logits_cat = torch.cat(all_bin_logits, dim=0)
            dir_logits_cat = torch.cat(all_dir_logits, dim=0)
            bin_pred = (torch.sigmoid(bin_logits_cat.to(device)) > 0.5).long()
            dir_pred = torch.argmax(dir_logits_cat.to(device), dim=1)
            default_no_prio = torch.full_like(bin_pred, 2)
            preds_cat = torch.where(bin_pred == 1, dir_pred, default_no_prio)

        total_correct = (preds_cat.cpu() == labels_cat).sum().item()
        total_pred    = labels_cat.size(0)
        accuracy      = total_correct / total_pred if total_pred > 0 else 0.0

        all_preds_np  = preds_cat.cpu().numpy()
        all_labels_np = labels_cat.cpu().numpy()
        macro_f1      = f1_score(all_labels_np, all_preds_np, average='macro')
        num_classes   = len(set(all_labels_np.tolist()))
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss.item()/total_pred:.4f} | Accuracy: {accuracy:.4f} | F1: {macro_f1:.4f}")
        writer.add_scalar('Loss/train', total_loss.item() / total_pred, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)

        print(f"Train accuracy per class: ", end='')
        per_class_acc = []
        for cls in range(num_classes):
            cls_idx = (all_labels_np == cls)
            acc = float('nan') if cls_idx.sum() == 0 else (all_preds_np[cls_idx] == cls).sum() / cls_idx.sum()
            per_class_acc.append(acc)
            print(f"{acc:.3f}, ", end='')
        print()
        for cls, acc in enumerate(per_class_acc):
            writer.add_scalar(f'Accuracy/train_class_{cls}', acc, epoch)

        # Evaluate
        test_loss, test_acc, per_class_acc = evaluate(test_loader, model, epoch, criterion, writer, device, mode=mode, ema=ema)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        acc_1_2 = (per_class_acc[0] + per_class_acc[1]) / 2
        if acc_1_2 > best_acc:
            best_acc = acc_1_2
            print(f"New best accuracy: {best_acc:.4f}, saving model...")
            torch.save(model.state_dict(), model_file)
            print(f"Model saved as {model_file}")
        else:
            print(f"Best accuracy is still {best_acc:.4f}")
        print()

        # Early stopping based on validation (test) accuracy
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement in val acc for {patience} epochs). Best val acc: {best_val_acc:.4f}")
                break
    return best_acc

def evaluate(test_loader, model, epoch, criterion, writer, device, mode="auto", ema=None):
    model.eval()

    # Swap to EMA weights for evaluation if available
    using_ema = False
    if ema is not None:
        ema.apply_shadow(model)
        using_ema = True

    is_tuple_crit = isinstance(criterion, (tuple, list)) and len(criterion) == 2
    if is_tuple_crit:
        bce_loss, dir_loss = criterion
    else:
        ce_loss = criterion

    total_loss = 0.0
    total_correct = 0
    total_pred = 0
    all_preds = []
    all_labels = []
    all_probs = []      # per-sample class probabilities [3]
    all_dists = []      # per-sample L1 distance

    with torch.no_grad():
        for batch_n, batch in tqdm(enumerate(test_loader)):
            obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, labels_batch, mask, dists_batch, dx_batch, dy_batch = batch

            obs_fovs_batch = obs_fovs_batch.to(device, non_blocking=True)
            neighbor_features_batch = neighbor_features_batch.to(device, non_blocking=True)
            neigh_coords_batch = neigh_coords_batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            dx_batch = dx_batch.to(device, non_blocking=True)
            dy_batch = dy_batch.to(device, non_blocking=True)

            # --- Build pair_pointer based on pointer mode ---
            pointer_mode = os.environ.get('RHCR_POINTER_MODE', 'none')
            try:
                pointer_mode = getattr(args, 'pointer_mode', pointer_mode)
            except Exception:
                pass
            pointer_mode = str(pointer_mode).lower()

            pair_pointer = None
            if pointer_mode != 'none':
                Bp = obs_fovs_batch.size(0)
                fov_size  = obs_fovs_batch.size(-1)
                cen = fov_size // 2
                # split per-side images: (B,2,C,F,F) -> A: (B,C,F,F), Bim: (B,C,F,F)
                Aimg_full = obs_fovs_batch[:, 0]
                Bimg_full = obs_fovs_batch[:, 1]

                # Strip the coordinate channel (detected by two nonzeros at [0,0] and [0,1])
                def strip_coord(t):
                    first = t[0]  # (C,F,F)
                    C_in  = first.size(0)
                    nz = (first != 0).view(C_in, -1).sum(dim=1)
                    coord_idx = None
                    for c in range(C_in):
                        if nz[c].item() == 2 and bool(first[c,0,0] != 0) and bool(first[c,0,1] != 0):
                            coord_idx = c; break
                    if coord_idx is None:
                        coord_idx = C_in - 1
                    if coord_idx == 0:
                        return t[:, 1:]
                    elif coord_idx == C_in - 1:
                        return t[:, :-1]
                    else:
                        return torch.cat([t[:, :coord_idx], t[:, coord_idx+1:]], dim=1)

                Aimg = strip_coord(Aimg_full)
                Bimg = strip_coord(Bimg_full)

                dx = dx_batch
                dy = dy_batch
                y_ab = (cen + dy).clamp(0, F-1)
                x_ab = (cen + dx).clamp(0, F-1)
                y_ba = (cen - dy).clamp(0, F-1)
                x_ba = (cen - dx).clamp(0, F-1)

                if pointer_mode == 'raw':
                    rng = torch.arange(Aimg.size(0), device=Aimg.device)
                    f_a_at_b = Aimg[rng, :, y_ab, x_ab]
                    f_b_at_a = Bimg[rng, :, y_ba, x_ba]
                elif pointer_mode == 'trunk':
                    # Encode trunk features for both sides in one call
                    flat_agents = torch.cat([Aimg, Bimg], dim=0)  # (2B, C_in, F, F)
                    trunk_maps = model.encode_trunk(flat_agents)
                    Fa = trunk_maps[:Bp]
                    Fb = trunk_maps[Bp:]
                    rng = torch.arange(Fa.size(0), device=Fa.device)
                    f_a_at_b = Fa[rng, :, y_ab, x_ab]
                    f_b_at_a = Fb[rng, :, y_ba, x_ba]
                else:
                    f_a_at_b = None; f_b_at_a = None

                size_y_t = float(getattr(env, 'size_y', 1.0))
                size_x_t = float(getattr(env, 'size_x', 1.0))
                dxn = dx.float() / max(1.0, size_x_t)
                dyn = dy.float() / max(1.0, size_y_t)
                dist_l1 = (dx.abs() + dy.abs()).float()

                ptr_A = torch.cat([f_a_at_b, dxn.unsqueeze(1), dyn.unsqueeze(1), dist_l1.unsqueeze(1)], dim=1)  # (B, C_ptr+3)
                ptr_B = torch.cat([f_b_at_a, (-dxn).unsqueeze(1), (-dyn).unsqueeze(1), dist_l1.unsqueeze(1)], dim=1)
                pair_pointer = torch.stack([ptr_A, ptr_B], dim=1)  # (B,2,D_ptr)

            outputs = model(obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, mask, pair_pointer=pair_pointer)

            # infer mode if needed
            effective_mode = mode
            if mode == "auto":
                if isinstance(outputs, tuple):
                    if len(outputs) == 3 and outputs[1].dim() == 1 and outputs[2].dim() == 2:
                        effective_mode = "stacked"
                    else:
                        last = outputs[-1]
                        effective_mode = "threeway" if (isinstance(last, torch.Tensor) and last.dim() == 2 and last.size(-1) == 3) else "stacked"
                else:
                    effective_mode = "threeway"

            if effective_mode == "stacked":
                if not (isinstance(outputs, tuple) and len(outputs) >= 3):
                    raise RuntimeError("Mode 'stacked' selected but model did not return (_, bin_logits, dir_logits).")
                _, bin_logits, dir_logits = outputs[-3], outputs[-2], outputs[-1]

                # labels
                bin_labels = (labels_batch < 2).float()
                dir_labels = labels_batch.clone()
                dir_labels[dir_labels == 2] = 0

                # losses
                if not is_tuple_crit:
                    # if only CE was provided, create BCE here for eval
                    bce = nn.BCEWithLogitsLoss().to(device)
                    ce  = nn.CrossEntropyLoss().to(device)
                    loss_b = bce(bin_logits, bin_labels)
                    mask_prio = bin_labels.bool()
                    loss_d = ce(dir_logits[mask_prio], dir_labels[mask_prio]) if mask_prio.any() else torch.tensor(0.0, device=device)
                else:
                    loss_b = bce_loss(bin_logits, bin_labels)
                    mask_prio = bin_labels.bool()
                    loss_d = dir_loss(dir_logits[mask_prio], dir_labels[mask_prio]) if mask_prio.any() else torch.tensor(0.0, device=device)
                loss = loss_b + loss_d

                # predictions
                bin_pred  = (torch.sigmoid(bin_logits) > 0.5).long()
                dir_pred  = torch.argmax(dir_logits, dim=1)
                fallback2 = torch.full_like(bin_pred, 2)
                pred      = torch.where(bin_pred == 1, dir_pred, fallback2)

                # Build class probabilities: P2 = 1-s, P0/P1 = s * softmax(dir)
                s = torch.sigmoid(bin_logits)
                sd = F.softmax(dir_logits, dim=1)
                p2 = (1.0 - s).unsqueeze(1)
                p01 = s.unsqueeze(1) * sd
                probs = torch.cat([p01, p2], dim=1)  # [B,3] order: 0,1,2

            elif effective_mode == "threeway":
                if isinstance(outputs, tuple):
                    class_logits = outputs[-1]
                else:
                    class_logits = outputs

                if is_tuple_crit:
                    ce = dir_loss if isinstance(dir_loss, nn.Module) else nn.CrossEntropyLoss().to(device)
                else:
                    ce = ce_loss if isinstance(ce_loss, nn.Module) else nn.CrossEntropyLoss().to(device)

                loss = ce(class_logits, labels_batch)
                pred = torch.argmax(class_logits, dim=1)
                probs = F.softmax(class_logits, dim=1)
            else:
                raise ValueError(f"Unknown mode '{mode}'")

            total_loss   += loss.item() * labels_batch.size(0)
            total_correct += (pred == labels_batch).sum().item()
            total_pred   += labels_batch.size(0)
            all_preds.extend (pred.cpu().tolist())
            all_labels.extend(labels_batch.cpu().tolist())
            if isinstance(probs, torch.Tensor):
                all_probs.extend(probs.cpu().tolist())
            all_dists.extend(dists_batch.cpu().tolist())

    avg_loss = total_loss / total_pred
    accuracy = total_correct / total_pred if total_pred > 0 else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    num_classes = len(set(all_labels_np))

    print(f"Test accuracy per class: ", end='')
    per_class_acc = []
    for cls in range(num_classes):
        cls_idx = (all_labels_np == cls)
        if cls_idx.sum() == 0:
            acc = float('nan')
        else:
            acc = (all_preds_np[cls_idx] == cls).sum() / cls_idx.sum()
        per_class_acc.append(acc)
        print(f"{acc:.3f}, ", end='')
    print()

    for cls, acc in enumerate(per_class_acc):
        writer.add_scalar(f'Accuracy/test_class_{cls}', acc, epoch)

    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    print(f"Per-class F1 score: ", end='')
    for f1 in per_class_f1:
        print(f"{f1:.2f}, ", end='')
    print()

    # Confusion matrix
    try:
        cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
        print("Confusion Matrix (rows=true, cols=pred):\n", cm)
        # log as individual scalars
        for i in range(3):
            for j in range(3):
                writer.add_scalar(f'Confusion/cm_{i}{j}', cm[i,j], epoch)
    except Exception as _:
        pass

    # Distance-bin metrics
    dists_np = np.array(all_dists)
    bins = [(0,4), (5,8), (9,9999)]
    for (lo,hi) in bins:
        mask = (dists_np >= lo) & (dists_np <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = (all_preds_np[mask] == all_labels_np[mask]).mean()
        writer.add_scalar(f'Accuracy/test_dist_{lo}_{hi}', acc_bin, epoch)
        # class-0 accuracy in bin
        mask_c0 = mask & (all_labels_np == 0)
        if mask_c0.sum() > 0:
            acc_c0 = (all_preds_np[mask_c0] == 0).mean()
            writer.add_scalar(f'Accuracy/test_c0_dist_{lo}_{hi}', acc_c0, epoch)

    # Save predictions CSV
    try:
        outdir = getattr(writer, 'log_dir', '.')
        csv_path = os.path.join(outdir, f"preds_epoch{epoch:03d}.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["label","pred","p0","p1","p2","dist_l1"]) 
            for (y, yhat, p, d) in zip(all_labels, all_preds, all_probs, all_dists):
                w.writerow([y, yhat, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}", d])
        print(f"Saved predictions to {csv_path}")
    except Exception as e:
        print(f"[warn] failed to save predictions CSV: {e}")

    if using_ema:
        ema.restore(model)
    return avg_loss, accuracy, per_class_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_stability", type=int, default=1, help="1 to enable stability package, 0 for baseline")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name for TensorBoard run directory")
    parser.add_argument("--use_hardneg", type=int, default=0, help="1 to enable hard-negative sampling for class 2")
    parser.add_argument("--hard_radius", type=int, default=5, help="Manhattan radius for hard negatives (class 2)")
    parser.add_argument("--hard_frac", type=float, default=0.5, help="Fraction of class-2 batch to draw from hard negatives")
    parser.add_argument("--pointer_mode", type=str, default="none", choices=["none","raw","trunk"],
                        help="Pointer features for pair reasoning: none (off), raw (sample from input channels), trunk (sample from encoder trunk)")
    parser.add_argument("--criterion", type=str, default="ce", choices=["ce", "focal"], help="Loss for threeway head")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focusing parameter gamma for focal loss") 
    parser.add_argument("--focal_alpha", type=str, default="0.5,0.3,0.2", help="Comma-separated per-class alpha for focal loss (len=3)")
    args, unknown = parser.parse_known_args()
    try:
        # --- Configurations ---
        CONFIG_NAME = "warehouse_2"
        CONFIG_FILE = "config.yaml"
        # --- Load Config and Initialize Environment ---
        with open(os.path.join(os.path.dirname(__file__), CONFIG_FILE), "r") as file:
            config_file = yaml.safe_load(file)
        config = config_file[CONFIG_NAME]
        env_config = config["environment"]
        WINDOW_SIZE = env_config["WINDOW_SIZE"]
        FOV = env_config["FOV"]
        NUM_AGENTS = env_config["NUM_AGENTS"]

        train_config = config["training"]
        device = train_config["DEVICE"]
        BATCH_SIZE = train_config["BATCH_SIZE"]
        LR = float(train_config["LR"])
        EPOCHS = train_config["EPOCHS"]
        # Optional head mode: "stacked", "threeway", or "auto"
        MODE = train_config.get("MODE", "auto")

        env = Environment(
            env_config,
            logger=Logger(),  # Dummy logger
            grid_map_file=config["paths"]["map_file"],
            device=device
        )

        map_name = os.path.basename(config["paths"]["map_file"]).replace('.npy','')
        sample_file = os.path.join(os.path.dirname(__file__), f'data_gen/{map_name}/w{WINDOW_SIZE}/{NUM_AGENTS}/')

        # ---- Build descriptive model filename from training args ----
        def sanitize(s: str) -> str:
            # make safe for filesystem: replace commas and spaces; keep dots and dashes
            return s.replace(',', '-').replace(' ', '')


        if args.experiment_name:
            tag = args.experiment_name
        else:
            tag_parts = []
            tag_parts.append('stability' if args.use_stability else 'base')
            if args.use_hardneg:
                tag_parts.append(f"HN-r{int(args.hard_radius)}-f{args.hard_frac:g}")
            if MODE == 'threeway' and args.criterion == 'focal':
                # include gamma and alpha vector
                try:
                    alphas_str = sanitize(args.focal_alpha)
                except Exception:
                    alphas_str = 'na'
                tag_parts.append(f"focal-g{args.focal_gamma:g}-a{alphas_str}")
            # --- Add pointer mode info if not "none" ---
            if getattr(args, "pointer_mode", "none") != "none":
                tag_parts.append(f"ptr-{args.pointer_mode}")
            tag = '_'.join(tag_parts)
        os.makedirs(f"models/w{WINDOW_SIZE}/{NUM_AGENTS}", exist_ok=True)
        model_file = os.path.join(f"models/w{WINDOW_SIZE}/{NUM_AGENTS}", f"N{NUM_AGENTS}_w{WINDOW_SIZE}_{MODE}_{tag}.pth")

        run_root = os.path.join("runs", map_name, MODE, f"w{WINDOW_SIZE}", f"{NUM_AGENTS}")
        if args.experiment_name:
            run_dir = os.path.join(run_root, args.experiment_name)
        else:
            run_dir = os.path.join(run_root, tag)
        writer = SummaryWriter(log_dir=run_dir)

        # --- Persist run configuration for traceability ---
        try:
            os.makedirs(run_dir, exist_ok=True)
            run_cfg = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": MODE,
                "env": {
                    "NUM_AGENTS": NUM_AGENTS,
                    "WINDOW_SIZE": WINDOW_SIZE,
                    "FOV": FOV,
                },
                "train": {
                    "DEVICE": device,
                    "BATCH_SIZE": BATCH_SIZE,
                    "LR": LR,
                    "EPOCHS": EPOCHS,
                    "use_stability": bool(args.use_stability),
                },
                "options": {
                    "use_hardneg": bool(args.use_hardneg),
                    "hard_radius": int(args.hard_radius),
                    "hard_frac": float(args.hard_frac),
                    "criterion": args.criterion,
                    "focal_gamma": float(args.focal_gamma) if args.criterion == "focal" else None,
                    "focal_alpha": str(args.focal_alpha) if args.criterion == "focal" else None,
                    "pointer_mode": str(args.pointer_mode),
                },
                "artifacts": {
                    "model_file": model_file,
                    "run_dir": run_dir,
                },
            }
            with open(os.path.join(run_dir, "run_config.yaml"), "w") as f:
                yaml.safe_dump(run_cfg, f, sort_keys=False)
        except Exception as _e:
            print(f"[warn] failed to write run_config.yaml: {_e}")

        # --- Model, Optimizer, Loss ---
        model = QNetwork(fov=FOV, head_mode=MODE).to(device)
        # model = nn.DataParallel(model, device_ids=[1,2,3,4,5,6,7], output_device=1)
        # model = nn.DataParallel(model)
        if args.use_stability:
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=LR)

        # --- Loss / criterion (depends on head mode) ---
        if MODE == "threeway":
            if args.criterion == "focal":
                try:
                    alphas = [float(x) for x in args.focal_alpha.split(',')]
                except Exception:
                    alphas = None
                criterion = FocalCE(gamma=args.focal_gamma, alpha=alphas)
            else:
                # Standard CE; if stability is on, keep label smoothing
                criterion = nn.CrossEntropyLoss(label_smoothing=0.05) if args.use_stability else nn.CrossEntropyLoss()
        else:
            bce_loss = nn.BCEWithLogitsLoss()
            dir_loss = nn.CrossEntropyLoss()
            criterion = (bce_loss, dir_loss)

        best_acc = train_on_dataset(
            env, model, optimizer, criterion, BATCH_SIZE, EPOCHS, writer, model_file, device,
            sample_file=sample_file, mode=MODE, use_stability=bool(args.use_stability),
            use_hardneg=bool(args.use_hardneg), hardneg_radius=args.hard_radius, hardneg_frac=args.hard_frac
        )

        writer.close()

    except Exception as e:
        print(f"\nException caught: {e}\nStarting pdb...")
        pdb.post_mortem()
        sys.exit(1)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()