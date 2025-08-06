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


from Environment import Environment
from Model import QNetwork
from utils import Logger

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

class PairDataset(Dataset):
    def __init__(self, data_txt, env, undersample=True):
        self.env   = env
        self.pairs = []      # will hold (line_idx, a, b, label)
        self.raw_data = []

        row, col = self.env.grid_map.shape
        row = row - 2
        col = col - 2
        assert (row, col) == (33, 46)

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
                    self.pairs.append((line_idx, a, b, label))

        # Print class counts before undersampling
        pre_counts = Counter([lbl for (_, _, _, lbl) in self.pairs])
        print(f"PairDataset: class counts before undersampling: {{0}}={pre_counts[0]}, {{1}}={pre_counts[1]}, {{2}}={pre_counts[2]}")

        if undersample:
            # --- undersample to smallest class count ---
            # count occurrences per label
            label_counts = Counter([label for (_, _, _, label) in self.pairs])
            min_count = min(label_counts.values())
            # group indices by label
            indices_by_label = {lbl: [] for lbl in label_counts}
            for idx, (_, _, _, lbl) in enumerate(self.pairs):
                indices_by_label[lbl].append(idx)
            # sample min_count indices per label
            selected_indices = []
            for lbl, idxs in indices_by_label.items():
                selected_indices.extend(random.sample(idxs, min_count))
            # rebuild pairs to undersampled set
            self.pairs = [self.pairs[i] for i in selected_indices]
            # Print class counts after undersampling
            post_counts = Counter([lbl for (_, _, _, lbl) in self.pairs])
            print(f"PairDataset: class counts after undersampling: {{0}}={post_counts[0]}, {{1}}={post_counts[1]}, {{2}}={post_counts[2]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # t0 = time.time()
        line_idx, a, b, label = self.pairs[idx]
        starts, goals, priorities = self.raw_data[line_idx]
        self.env.starts = starts
        self.env.goals = goals

        # t2 = time.time()
        obs_fovs = self.env.get_obs([a, b])
        # print(f"obs_fovs: {time.time()-t2:0.4f}")
        # t3 = time.time()
        neighbor_features, neighbor_coords = self.env.get_neighbor_goal_heuristics_as_patches([a, b])
        # print(f"get_neighbor_goal_heuristics_as_patches: {time.time()-t3:0.4f}")

        # Pad neighbor lists to fixed size = num_agents-1
        max_nb = self.env.num_agents - 1
        # obs_fovs is shape (2, C, fov, fov)
        # build tensor for neighbors: (2, max_nb, 1, fov, fov)
        padded_feats = torch.zeros((2, max_nb, 1, self.env.fov, self.env.fov), dtype=torch.float32)
        padded_coords = torch.zeros((2, max_nb, 2), dtype=torch.float32)
        mask = torch.ones((2, max_nb), dtype=torch.bool)
        for i, feats in enumerate(neighbor_features):
            n = feats.size(0)
            if n > max_nb:
                feats = feats[:max_nb]
                coords = neighbor_coords[i][:max_nb]
                n = max_nb
            else:
                coords = neighbor_coords[i]
            if n > 0:
                padded_feats[i, :n] = feats
                padded_coords[i, :n] = coords
                mask[i, :n] = False

        # print(f"total getitem: {time.time()-t0:.04f}")
        return obs_fovs, padded_feats, padded_coords, mask, label


def custom_collate(batch):
    # Unzip batch
    # t0 = time.time()
    obs_list, neigh_list, neigh_coords_list, mask_list, labels_list = zip(*batch)

    obs_batch = torch.stack(obs_list)
    neigh_batch = torch.stack(neigh_list, dim=0)
    neigh_coords_batch = torch.stack(neigh_coords_list, dim=0)
    labels_batch = torch.tensor(labels_list)
    mask_batch = torch.stack(mask_list, dim=0)

    # print(f"custom_collate: {time.time()-t0:.04f}")
    return obs_batch, neigh_batch, neigh_coords_batch, labels_batch, mask_batch

# --- Training Loop ---
def train_on_dataset(env, model, optimizer, criterion, BATCH_SIZE, train_epochs, writer, model_file, sample_file=None):
    bce_loss, dir_loss = criterion

    data = PairDataset(sample_file+'data.txt', env)
    balanced_loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        collate_fn=custom_collate,
        shuffle=True,
        # num_workers=8,
        pin_memory=True,
        # persistent_workers=True
    )
    print(f"Number of batches: {len(balanced_loader)}")

    test_data = PairDataset(sample_file+'test_data.txt', env, undersample=True)
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        collate_fn=custom_collate,
        shuffle=False,
        pin_memory=True,
    )

    model.train()
    best_acc = 0
    for epoch in range(train_epochs):
        print(f"Training epoch {epoch + 1}/{train_epochs}...")
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        # accumulate correct counts on GPU to avoid sync per-batch
        total_correct_tensor = torch.tensor(0, device=device)
        total_pred = 0
        # Use tensor lists for efficient batch aggregation
        all_pred_tensors = []
        all_label_tensors = []

        for batch_n, batch in tqdm(enumerate(balanced_loader)):
            obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, labels_batch, mask = batch
            # t0 = time.perf_counter()

            # Measure individual transfer times
            obs_fovs_batch = obs_fovs_batch.to(device, non_blocking=True)
            neighbor_features_batch = neighbor_features_batch.to(device, non_blocking=True)
            neigh_coords_batch = neigh_coords_batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            # print(f"to device: {time.perf_counter()-t0:.04f}")

            # Forward pass
            # t2 = time.perf_counter()
            # _, batch_q_vals = model(obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, mask)
            _, bin_logits, dir_logits = model(obs_fovs_batch,
                                        neighbor_features_batch,
                                        neigh_coords_batch,
                                        mask)
            # print(f"forward: {time.perf_counter()-t2:.04f}")

            # 1) build binary labels: classes 0 or 1 → 1, class 2 → 0
            # labels_batch is shape (B,), with values in {0,1,2}
            bin_labels = (labels_batch < 2).float()               # shape (B,)
            # 2) direction labels: only meaningful where bin_labels==1
            # we'll still keep tensor of shape (B,) but only compute CE on idxs
            dir_labels = labels_batch.clone()
            dir_labels[dir_labels == 2] = 0                       # placeholder for “no-priority” rows
            # 3) compute losses
            loss_bin = bce_loss(bin_logits, bin_labels)
            # pick out only the priority examples for the direction loss
            mask_prio = bin_labels.bool()                         # shape (B,)
            if mask_prio.any():
                # dir_logits[mask_prio] has shape (P,2), dir_labels[mask_prio] is (P,)
                loss_dir = dir_loss(dir_logits[mask_prio], dir_labels[mask_prio])
            else:
                loss_dir = torch.tensor(0.0, device=obs_fovs_batch.device)
            # 4) combine
            # You can weight them differently if you like:
            loss = loss_bin + loss_dir
            # backward as before
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # t3 = time.perf_counter()
            # loss = criterion(batch_q_vals, labels_batch)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # print(f"backward: {time.perf_counter()-t3:.04f}")

            # After you compute bin_logits (shape [B]) and dir_logits (shape [B,2]):
            # 1) Binary prediction: 1 if we think it’s priority (i.e. class 0 or 1), 0 if “no priority” (class 2)
            bin_pred = (torch.sigmoid(bin_logits) > 0.5).long()    # [B], values in {0,1}
            # 2) Direction prediction among the priority examples
            dir_pred = torch.argmax(dir_logits, dim=1)            # [B], values in {0,1}
            # 3) Fuse into a single 3-way prediction:
            #    wherever bin_pred==0 → class 2, else → dir_pred (0 or 1)
            default_no_prio = torch.full_like(bin_pred, 2)        # [B] all-2
            pred = torch.where(bin_pred == 1, dir_pred, default_no_prio)
            # now `pred` is shape [B], values in {0,1,2}
            # 4) Compare to the true labels_batch
            correct_tensor = (pred == labels_batch).sum()
            batch_n       = labels_batch.size(0)
            total_correct_tensor += correct_tensor
            total_pred           += batch_n
            all_pred_tensors.append(pred)
            all_label_tensors.append(labels_batch)

            # # torch.cuda.synchronize()
            # # _t0 = time.perf_counter()
            # pred = torch.argmax(batch_q_vals, dim=1)
            # # torch.cuda.synchronize()
            # # _t1 = time.perf_counter()
            # correct_tensor = (pred == labels_batch).sum()
            # # torch.cuda.synchronize()
            # # _t2 = time.perf_counter()
            # # _ta = time.perf_counter()
            # batch_n = obs_fovs_batch.size(0)
            # # _tb = time.perf_counter()
            # total_correct_tensor += correct_tensor
            # # _tc = time.perf_counter()
            # total_pred += batch_n
            # # _td = time.perf_counter()
            # total_loss += loss * batch_n
            # # _te = time.perf_counter()
            # all_pred_tensors.append(pred)
            # # _tf = time.perf_counter()
            # all_label_tensors.append(labels_batch)
            # _tg = time.perf_counter()
            # print(
            #     f"pred={_t1-_t0:.6f}s, "
            #     f"correct_tensor={_t2-_t1:.6f}s, "
            #     f"batch_n={_tb-_ta:.6f}s, "
            #     f"correct_update={_tc-_tb:.6f}s, "
            #     f"pred_count_update={_td-_tc:.6f}s, "
            #     f"loss_update={_te-_td:.6f}s, "
            #     f"append_pred={_tf-_te:.6f}s, "
            #     f"append_label={_tg-_tf:.6f}s"
            # )

            # print(f"total forward: {time.perf_counter()-t0:.04f}")

        # finalize correct count once, causing a single sync
        total_correct = total_correct_tensor.item()
        total_loss = total_loss.item()
        # Concatenate and move to CPU only once
        all_preds = torch.cat(all_pred_tensors).cpu().tolist()
        all_labels = torch.cat(all_label_tensors).cpu().tolist()
        accuracy = total_correct / total_pred if total_pred > 0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        num_classes = len(set(all_labels_np))
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/total_pred:.4f} | Accuracy: {accuracy:.4f} | F1: {macro_f1:.4f}")
        writer.add_scalar('Loss/train', total_loss / total_pred, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        print(f"Train accuracy per class: ", end='')
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
            writer.add_scalar(f'Accuracy/train_class_{cls}', acc, epoch)
        test_loss, test_acc, per_class_acc = evaluate(test_loader, model, BATCH_SIZE, epoch, criterion, writer)
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
    return best_acc

# --- Evaluation ---
def evaluate(test_loader, model, batch_size, epoch, criterion, writer):
    model.eval()
    bce_loss, dir_loss = criterion

    total_loss = 0
    total_correct = 0
    total_pred = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_n, batch in tqdm(enumerate(test_loader)):
            obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, labels_batch, mask = batch

            obs_fovs_batch = obs_fovs_batch.to(device, non_blocking=True)
            neighbor_features_batch = neighbor_features_batch.to(device, non_blocking=True)
            neigh_coords_batch = neigh_coords_batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            # Forward pass
            outputs = model(obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, mask)
            # model should now return something like (encodings, bin_logits, dir_logits)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                _, bin_logits, dir_logits = outputs
            else:
                raise RuntimeError("Expected model to return (enc, bin_logits, dir_logits)")
            # --- build the two sets of ground‐truth labels ---
            # binary: 1 for classes {0,1}, 0 for class 2
            bin_labels = (labels_batch < 2).float()  # [B]
            # direction: only meaningful where bin_labels==1; just reuse labels 0/1
            dir_labels = labels_batch.clone()
            dir_labels[dir_labels == 2] = 0          # dummy for the “no‐priority” rows

            # --- compute losses ---
            loss_b = bce_loss(bin_logits.view(-1), bin_labels)
            mask_prio = bin_labels.bool()
            if mask_prio.any():
                loss_d = dir_loss(dir_logits[mask_prio], dir_labels[mask_prio])
            else:
                loss_d = torch.tensor(0.0, device=device)
            loss = loss_b + loss_d

            # accumulate
            total_loss    += loss.item()
            total_pred     += labels_batch.size(0)

            # --- reconstruct final 3‐way prediction ---
            bin_pred   = (torch.sigmoid(bin_logits) > 0.5).long()   # [B] in {0,1}
            dir_pred   = torch.argmax(dir_logits, dim=1)           # [B] in {0,1}
            fallback2  = torch.full_like(bin_pred, 2)              # [B] all‐2
            pred       = torch.where(bin_pred == 1, dir_pred, fallback2)

            total_correct += (pred == labels_batch).sum().item()
            all_preds.extend (pred.cpu().tolist())
            all_labels.extend(labels_batch.cpu().tolist())

    avg_loss = total_loss / len(test_loader)
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

    # all_labels and all_preds should be 1D arrays/lists of ints
    assert isinstance(all_labels, list) and isinstance(all_preds, list), "Should be lists"
    assert all(isinstance(x, int) for x in all_labels), "all_labels not all ints"
    assert all(isinstance(x, int) for x in all_preds), "all_preds not all ints"
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    print(f"Per-class F1 score: ", end='')
    for f1 in per_class_f1:
        print(f"{f1:.2f}, ", end='')
    print()

    return avg_loss, accuracy, per_class_acc

def main():
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

        env = Environment(
            env_config,
            logger=Logger(),  # Dummy logger
            grid_map_file=config["paths"]["map_file"],
            heuristic_map_file=config["paths"]["heur_file"]
        )

        train_config = config["training"]
        DEVICE = train_config["DEVICE"]
        BATCH_SIZE = train_config["BATCH_SIZE"]
        LR = float(train_config["LR"])
        label_idx = train_config["N_ACTIONS"]
        EPOCHS = train_config["EPOCHS"]

        sample_file = os.path.join(os.path.dirname(__file__), f'data_gen/{NUM_AGENTS}/w{WINDOW_SIZE}/')

        USE_NEIGHCOORDS = True
        model_file = f"sup_pbs_{NUM_AGENTS}_w{WINDOW_SIZE}_binary.pth"
        writer = SummaryWriter(log_dir=f"runs/binary/w{WINDOW_SIZE}/{NUM_AGENTS}")

        # --- Model, Optimizer, Loss ---
        model = QNetwork(fov=FOV, USE_NEIGHCOORDS=USE_NEIGHCOORDS).to(DEVICE)
        # model = nn.DataParallel(model, device_ids=[1,2,3,4,5,6,7], output_device=1)
        # model = nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # weight = torch.tensor([1.0, 1.0, 0.5], device=DEVICE)
        # criterion = nn.CrossEntropyLoss(weight=weight)

        bce_loss   = nn.BCEWithLogitsLoss()                   # binary head
        dir_loss   = nn.CrossEntropyLoss()
        criterion = (bce_loss, dir_loss)

        best_acc = train_on_dataset(
            env, model, optimizer, criterion, BATCH_SIZE, EPOCHS, writer, model_file, sample_file=sample_file
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