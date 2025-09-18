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


from Environment import Environment
from Model import QNetwork
from utils import Logger

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

        with open(data_txt) as f:
            print(f"Reading from {data_txt}...")
            for line_idx, line in tqdm(enumerate(f)):
                starts_str, goals_str, priorities_str = line.split(';')
                starts = ast.literal_eval(starts_str)
                goals = ast.literal_eval(goals_str)
                priorities = ast.literal_eval(priorities_str.replace(': [', ':['))
                # revert from idx to coord
                starts = [(start//col + 1, start%col + 1) for start in starts]
                goals = [[(g//col + 1, g%col + 1) for g in goal] for goal in goals]

                partial_prio = []
                for low, highs in priorities.items():
                    for high in highs:
                        partial_prio.append((high, low))

                self.raw_data.append((starts, goals, partial_prio))

                self.env.starts = starts
                close_pairs = self.env.get_close_pairs()
                for (a,b) in close_pairs:
                    if (a,b) in partial_prio:
                        label = 0
                    elif (b,a) in partial_prio:
                        label = 1
                    else:
                        label = 2
                    # mark hard negatives for class 2 based on Manhattan distance in starts
                    if label == 2:
                        ya, xa = starts[a]
                        yb, xb = starts[b]
                        manh = abs(yb - ya) + abs(xb - xa)
                        if manh <= self.hardneg_radius:
                            self._class2_hard.append(len(self.pairs))  # index that will be used when appended
                    self.pairs.append((line_idx, a, b, label))

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

def custom_collate(batch, env, add_pair_id=False, add_pair_rel=False):
    """Collate by grouping items sharing the same line_idx (episode).
    For each episode, compute get_obs() and get_neighbor_goal_heuristics_as_patches()
    once over the unique agents participating in that episode's pairs, then
    assemble per-pair tensors in the ORIGINAL batch order.
    """

    # Unpack batch tuples: (line_idx, starts, goals, (a,b), label)
    line_idx, starts_list, goals_list, agent_pairs, labels = zip(*batch)
    B_total = len(agent_pairs)

    # Labels (preserve original order)
    labels_batch = torch.tensor(labels, dtype=torch.long)

    # Group indices by episode (line_idx) while preserving insertion order
    groups = defaultdict(list)  # line_idx -> list of indices within this batch
    for i, li in enumerate(line_idx):
        groups[li].append(i)

    # Preallocate containers in ORIGINAL order
    obs_chunks  = [None] * B_total                # each slot: (2, C, fov, fov)
    neigh_flat  = [None] * (2 * B_total)          # slot 2*i and 2*i+1 for pair i
    coords_flat = [None] * (2 * B_total)          # same indexing as neigh_flat
    dists = [None] * B_total  # per-pair L1 distance (|dx|+|dy|)

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
        env.goals  = goals

        # 1) Compute per-agent observations ONCE for this episode
        obs_K = env.get_obs(ep_agents)  # (K, C, fov, fov)
        if obs_K.device != env.device:
            obs_K = obs_K.to(env.device, non_blocking=True)

        # 2) Compute per-agent neighbor heuristic patches ONCE for this episode
        nf_list, nc_list = env.get_neighbor_goal_heuristics_as_patches(ep_agents)
        # nf_list / nc_list are lists of length K with tensors for each agent

        # 3) Fill slots for each pair by its original batch index
        for i in idxs:
            a, b = agent_pairs[i]
            ia = agent_to_pos[a]
            ib = agent_to_pos[b]
            # observations for the pair → (2, C, fov, fov)
            pair_obs = torch.stack([obs_K[ia], obs_K[ib]], dim=0)

            # compute L1 distance between agents a and b
            ya, xa = starts[a]
            yb, xb = starts[b]
            l1 = abs(yb - ya) + abs(xb - xa)
            dists[i] = l1

            # Optional pair features
            if add_pair_id or add_pair_rel:
                fov_sz = pair_obs.size(-1)
                cen = fov_sz // 2

            if add_pair_id:
                # Neighbor-only 1-hot per perspective (no ego channel)
                pid = torch.zeros((2, 1, fov_sz, fov_sz), device=pair_obs.device, dtype=pair_obs.dtype)
                ya, xa = starts[a]
                yb, xb = starts[b]
                # A perspective: mark B relative to A
                dy, dx = (yb - ya), (xb - xa)
                y_rel, x_rel = cen + dy, cen + dx
                if 0 <= y_rel < fov_sz and 0 <= x_rel < fov_sz:
                    pid[0, 0, y_rel, x_rel] = 1.0
                # B perspective: mark A relative to B
                dy2, dx2 = (ya - yb), (xa - xb)
                y_rel2, x_rel2 = cen + dy2, cen + dx2
                if 0 <= y_rel2 < fov_sz and 0 <= x_rel2 < fov_sz:
                    pid[1, 0, y_rel2, x_rel2] = 1.0
                pair_obs = torch.cat([pair_obs, pid], dim=1)  # (2, C+1, fov, fov)

            if add_pair_rel:
                # Two broadcast channels with normalized (dx,dy) from ego->neighbor
                dx_norm = (xb - xa) / env.size_x
                dy_norm = (yb - ya) / env.size_y
                rel = torch.tensor([dx_norm, dy_norm], device=pair_obs.device, dtype=pair_obs.dtype)
                rel = rel.view(1, 2, 1, 1).expand(2, -1, fov_sz, fov_sz)
                pair_obs = torch.cat([pair_obs, rel], dim=1)  # (2, C+1(+2), fov, fov)

            obs_chunks[i] = pair_obs
            # neighbor features/coords for A then B (keeps alignment with obs_pairs)
            neigh_flat[2 * i]     = nf_list[ia]
            neigh_flat[2 * i + 1] = nf_list[ib]
            coords_flat[2 * i]     = nc_list[ia]
            coords_flat[2 * i + 1] = nc_list[ib]

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

    max_nb       = padded_feats.size(1) if padded_feats.dim() > 1 else 0
    neigh_batch  = padded_feats.view(B, 2, max_nb, 1, fov, fov)
    coords_batch = padded_coords.view(B, 2, max_nb, 2)
    mask_batch   = mask_flat.view(B, 2, max_nb)

    dists_batch = torch.tensor(dists, dtype=torch.long)
    return obs_batch, neigh_batch, coords_batch, labels_batch, mask_batch, dists_batch

# --- Training Loop ---
def train_on_dataset(env, model, optimizer, criterion, BATCH_SIZE, train_epochs, writer, model_file, device, sample_file=None, mode="auto", ema_decay=0.999, use_stability=True, add_pair_id=False, add_pair_rel=False, use_hardneg=False, hardneg_radius=5, hardneg_frac=0.5):
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
        collate_fn=lambda batch, env=env, add_pair_id=add_pair_id, add_pair_rel=add_pair_rel: custom_collate(batch, env, add_pair_id=add_pair_id, add_pair_rel=add_pair_rel),
        shuffle=True,
        # num_workers=8,
        # pin_memory=True,
        # persistent_workers=True
    )
    print(f"Number of batches: {len(balanced_loader)}")

    test_data = PairDataset(sample_file+'test_data.txt', env, undersample=True, use_hardneg=False)
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch, env=env, add_pair_id=add_pair_id, add_pair_rel=add_pair_rel: custom_collate(batch, env, add_pair_id=add_pair_id, add_pair_rel=add_pair_rel),
        shuffle=False,
        # pin_memory=True,
    )

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
            obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, labels_batch, mask, dists_batch = batch

            # Transfers
            obs_fovs_batch = obs_fovs_batch.to(device, non_blocking=True)
            neighbor_features_batch = neighbor_features_batch.to(device, non_blocking=True)
            neigh_coords_batch = neigh_coords_batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            # Forward
            outputs = model(obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, mask)

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
            obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, labels_batch, mask, dists_batch = batch

            obs_fovs_batch = obs_fovs_batch.to(device, non_blocking=True)
            neighbor_features_batch = neighbor_features_batch.to(device, non_blocking=True)
            neigh_coords_batch = neigh_coords_batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            outputs = model(obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, mask)

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
    parser.add_argument("--pair_id", type=int, default=0, help="1 to add neighbor one-hot channel (paired agent) to obs")
    parser.add_argument("--pair_rel", type=int, default=0, help="1 to add two broadcast channels with normalized (dx,dy) from ego->neighbor")
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
            heuristic_map_file=config["paths"]["heur_file"],
            device=device
        )

        map_name = os.path.basename(config["paths"]["map_file"]).replace('.npy','')
        sample_file = os.path.join(os.path.dirname(__file__), f'data_gen/{map_name}/{NUM_AGENTS}/w{WINDOW_SIZE}/')

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
            if args.pair_id:
                tag_parts.append("pairMask")
            if args.pair_rel:
                tag_parts.append("pairRel")
            if MODE == 'threeway' and args.criterion == 'focal':
                # include gamma and alpha vector
                try:
                    alphas_str = sanitize(args.focal_alpha)
                except Exception:
                    alphas_str = 'na'
                tag_parts.append(f"focal-g{args.focal_gamma:g}-a{alphas_str}")
            tag = '_'.join(tag_parts)
        os.makedirs("models", exist_ok=True)
        model_file = os.path.join("models", f"N{NUM_AGENTS}_w{WINDOW_SIZE}_{MODE}_{tag}.pth")

        run_root = os.path.join("runs", map_name, MODE, f"w{WINDOW_SIZE}", f"{NUM_AGENTS}")
        run_dir  = run_root
        if args.experiment_name:
            run_dir = os.path.join(run_root, args.experiment_name)
        else:
            tag = "stability" if args.use_stability else "base"
            if args.use_hardneg:
                tag += f"_HN-r{int(args.hard_radius)}-f{args.hard_frac:g}"
            if args.pair_id:
                tag += "_pairMask"
            if args.pair_rel:
                tag += "_pairRel"
            if train_config.get("MODE", "auto") == "threeway" and args.criterion == "focal":
                # include both gamma and alpha in run name
                alphas_str = sanitize(args.focal_alpha) if isinstance(args.focal_alpha, str) else "na"
                tag += f"_focal-g{args.focal_gamma:g}-a{alphas_str}"
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
                    "pair_id": bool(args.pair_id),
                    "pair_rel": bool(args.pair_rel),
                    "criterion": args.criterion,
                    "focal_gamma": float(args.focal_gamma) if args.criterion == "focal" else None,
                    "focal_alpha": str(args.focal_alpha) if args.criterion == "focal" else None,
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
            add_pair_id=bool(args.pair_id), add_pair_rel=bool(args.pair_rel),
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