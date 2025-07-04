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
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score
import torch.multiprocessing as mp

from Environment import Environment
from Model import QNetwork
from utils import Logger, step

# --- Dataset Generation ---
class PairwiseDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s
    
class PairDataset(Dataset):
    def __init__(self, data_txt, env):
        self.env   = env
        self.pairs = []      # will hold (line_idx, a, b, label)
        self.raw_data = []

        row, col = self.env.grid_map.shape
        row = row - 2
        col = col - 2
        assert (row, col) == (33, 46)

        with open(data_txt) as f:
            print("Reading from data.txt...")
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
        from collections import Counter as _Counter_pre
        pre_counts = _Counter_pre([lbl for (_, _, _, lbl) in self.pairs])
        print(f"PairDataset: class counts before undersampling: {{0}}={pre_counts[0]}, {{1}}={pre_counts[1]}, {{2}}={pre_counts[2]}")
        # --- undersample to smallest class count ---
        from collections import Counter
        # count occurrences per label
        label_counts = Counter([label for (_, _, _, label) in self.pairs])
        min_count = min(label_counts.values())
        # group indices by label
        indices_by_label = {lbl: [] for lbl in label_counts}
        for idx, (_, _, _, lbl) in enumerate(self.pairs):
            indices_by_label[lbl].append(idx)
        # sample min_count indices per label
        import random
        selected_indices = []
        for lbl, idxs in indices_by_label.items():
            selected_indices.extend(random.sample(idxs, min_count))
        # rebuild pairs to undersampled set
        self.pairs = [self.pairs[i] for i in selected_indices]
        # Print class counts after undersampling
        from collections import Counter as _Counter_post
        post_counts = _Counter_post([lbl for (_, _, _, lbl) in self.pairs])
        print(f"PairDataset: class counts after undersampling: {{0}}={post_counts[0]}, {{1}}={post_counts[1]}, {{2}}={post_counts[2]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        line_idx, a, b, label = self.pairs[idx]
        starts, goals, priorities = self.raw_data[line_idx]

        self.env.starts = starts
        self.env.goals = goals
        t0 = time.time()
        self.env.DHC_heur = self.env._get_DHC_heur()
        # print(f"_get_DHC_heur time: {time.time() - t0:.3f}s")

        t_obs = time.time()
        obs_fovs = self.env.get_obs()
        device = obs_fovs.device
        # print(f"get_obs time: {time.time() - t_obs:.3f}s")

        t_nf = time.time()
        neighbor_features_and_coord = self.env.get_neighbor_goal_heuristics_as_patches([a, b])
        # print(f"get_neighbor_goal_heuristics time: {time.time() - t_nf:.3f}s")

        obs = torch.stack([obs_fovs[a], obs_fovs[b]])  # shape (2, C, H, W)

        stacked_neigh = [
            torch.stack([n[0] for n in neigh]) if len(neigh) > 0 else None
            for neigh in neighbor_features_and_coord
        ]
        neigh = [stacked_neigh[0], stacked_neigh[1]]
        
        stacked_coords = [[n[1] for n in neigh] for neigh in neighbor_features_and_coord]
        neigh_coords = [torch.stack(stacked_coords[0]), torch.stack(stacked_coords[1])]

        label = torch.tensor(label).to(device)

        return obs, neigh, neigh_coords, label

def flatten_samples(newdata):
    samples = []
    for sample in newdata:
        obs_fovs, neighbor_features, close_pairs, labels = sample
        # Pre-stack neighbor features for all agents only once per sample
        stacked_neigh = [
            torch.stack([n[0] for n in neigh]) if len(neigh) > 0 else None
            for neigh in neighbor_features
        ]
        stacked_coords = [[n[1] for n in neigh] for neigh in neighbor_features]
        for i, label in enumerate(labels):
            a, b = close_pairs[i]
            entry = {
                'obs_fov': torch.stack([obs_fovs[a], obs_fovs[b]]),  # shape (2, C, H, W)
                'neighbor_features': [stacked_neigh[a], stacked_neigh[b]],
                'close_pair': (a, b),
                'label': label.item() if torch.is_tensor(label) else int(label),
                'neigh_coords': [stacked_coords[a], stacked_coords[b]],
            }
            samples.append(entry)
    return samples

def custom_collate(batch):
    # batch: list of (obs, neigh, neigh_coords, label)
    obs_list, neigh_list, neigh_coords_list, labels_list = zip(*batch)
    # Stack observations into single tensor: (batch_size, 2, C, H, W)
    obs_batch = torch.stack(obs_list)
    # Labels tensor
    labels_batch = torch.stack(labels_list)
    # neigh_list is a tuple of lists [nbatch of [tensor,...], ...], convert to list
    neigh_batch = list(neigh_list)
    # neigh_coords_batch = list(neigh_coords_list)
    neigh_coords_batch = [element for sublist in neigh_coords_list for element in sublist]
    return obs_batch, neigh_batch, neigh_coords_batch, labels_batch


def undersample_samples(samples):
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, entry in enumerate(samples):
        class_indices[entry['label']].append(idx)
    # Find the minority class count
    min_count = min(len(v) for v in class_indices.values())
    print(f"Undersampling to {min_count} samples per class")
    # Randomly select min_count indices per class
    undersampled_indices = []
    for indices in class_indices.values():
        undersampled_indices.extend(random.sample(indices, min_count))
    # Build the new undersampled samples list
    undersampled_samples = [samples[i] for i in undersampled_indices]
    random.shuffle(undersampled_samples)
    return undersampled_samples

def move_batch(batch):
    return batch
    obs, neigh, coords, labels = batch
    obs = obs.to(DEVICE, non_blocking=True) # torch.Size([64, 2, 8, 11, 11])
    if USE_NEIGHCOORDS:
        coords = [c.to(DEVICE, non_blocking=True) for c in coords] # (64, 2, tensor('num_neigh', 1, 11, 11))
    else:
        coords = None
    labels = labels.to(DEVICE, non_blocking=True) # labels_batch: (64)
    return obs, neigh, coords, labels

# --- Training Loop ---
def train_on_dataset(env, model, optimizer, criterion, BATCH_SIZE, train_epochs, writer, model_file, sample_file=None, train_set=None, test_set=None):

    # if os.path.exists(sample_file + 'samples.pkl'):
    #     print(f"Loading samples from {sample_file}...")
    #     with open(sample_file + 'samples.pkl', 'rb') as f:
    #         samples = pickle.load(f)
    #     print(f"Loaded {len(samples)} train samples.")

    #     with open(sample_file + 'test_samples.pkl', 'rb') as f:
    #         test_set_flatten = pickle.load(f)
    #     print(f"Loaded {len(test_set_flatten)} test samples.")
    # else:
    #     all_labels = [lab for sam in train_set for lab in sam[label_idx]]
    #     print("Train label 0:", all_labels.count(0), "1:", all_labels.count(1), "2:", all_labels.count(2))
    #     all_labels = [lab for sam in test_set for lab in sam[label_idx]]
    #     print("Test label 0:", all_labels.count(0), "1:", all_labels.count(1), "2:", all_labels.count(2))

    #     t0 = time.time()
    #     samples = flatten_samples(train_set)
    #     print(f"Flattened {len(samples)} samples in {time.time() - t0:.3f}s")

    #     # undersample
    #     samples = undersample_samples(samples)
    #     with open(sample_file + 'samples.pkl', 'wb') as f:
    #         pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f"Saved {len(samples)} samples to {sample_file + 'samples.pkl'}.")

    #     test_set_flatten = flatten_samples(test_set)
    #     with open(sample_file + 'test_samples.pkl', 'wb') as f:
    #         pickle.dump(test_set_flatten, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f"Saved {len(test_set_flatten)} samples to {sample_file + 'test_samples.pkl'}.")

    # labels = [entry['label'] for entry in samples]
    # print("Undersampled Train label 0:", labels.count(0), "1:", labels.count(1), "2:", labels.count(2))
    # labels = [entry['label'] for entry in test_set_flatten]
    # print("Test label 0:", labels.count(0), "1:", labels.count(1), "2:", labels.count(2))

    # pair_dataset = PairwiseDataset(samples)
    # balanced_loader = DataLoader(
    #     pair_dataset,
    #     batch_size=BATCH_SIZE,
    #     collate_fn=custom_collate,
    #     shuffle=True,x
    #     # num_workers=1,
    #     # pin_memory=True,
    #     # persistent_workers=True
    # )

    data = PairDataset(sample_file+'data.txt', env)

    balanced_loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        collate_fn=custom_collate,
        shuffle=True,
        # num_workers=1,               # ← dispatch 4 workers in parallel
        # prefetch_factor=1,           # ← each worker will pre‐fetch 2 samples into its buffer
        # persistent_workers=True,     # ← keep workers alive across epochs
        # pin_memory=True,             # ← stage CPU→GPU copies asynchronously
        # multiprocessing_context=mp.get_context('spawn'),
    )

    print(f"Number of batches: {len(balanced_loader)}")

    model.train()
    best_acc = 0
    for epoch in range(train_epochs):
        print(f"Training epoch {epoch + 1}/{train_epochs}...")
        total_loss = 0
        total_correct = 0
        total_pred = 0
        all_preds = []
        all_labels = []

        for batch_n, batch in tqdm(enumerate(balanced_loader)):
            t0 = time.time()

            obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, labels_batch = move_batch(batch)
            batch_size = len(labels_batch)

            t1 = time.time()

            _, batch_q_vals = model(obs_fovs_batch, [], neighbor_features_batch, neigh_coords_batch)
            batch_q_vals_flat = batch_q_vals

            print("here")
            print(labels_batch[0].device)
            loss = criterion(batch_q_vals_flat, labels_batch)
            print("her1")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t3 = time.time()

            pred = torch.argmax(batch_q_vals_flat, dim=1)
            correct = (pred == labels_batch).sum().item()
            total_correct += correct
            total_pred += batch_size
            total_loss += loss.item() * batch_size

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels_batch.cpu().tolist())

            print(f"Batch {batch_n}: load={t1-t0:.3f}s, forward+back={t3-t1:.3f}s")

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

        # test_loss, test_acc, per_class_acc = evaluate(test_set_flatten, model, BATCH_SIZE, epoch, criterion, writer)
        # print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        # writer.add_scalar('Loss/test', test_loss, epoch)
        # writer.add_scalar('Accuracy/test', test_acc, epoch)

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
def evaluate(dataset, model, batch_size, epoch, criterion, writer):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pred = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if not batch:
                continue
            obs_pair, neigh, neigh_coords, labels = move_batch(custom_collate(batch))

            # Forward pass
            logits = model(obs_pair, [], neigh, neigh_coords)
            # If model returns (encodings, logits), grab only logits
            if isinstance(logits, tuple):
                logits = logits[1] if len(logits) > 1 else logits[0]

            loss = criterion(logits, labels)

            pred = torch.argmax(logits, dim=1)
            correct = (pred == labels).sum().item()

            total_correct += correct
            total_pred += labels.size(0)
            total_loss += loss.item()

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataset)
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
        model_file = f"sup_pbs_{NUM_AGENTS}_w{WINDOW_SIZE}.pth"
        writer = SummaryWriter(log_dir=f"runs/with_neighcoords/w{WINDOW_SIZE}/{NUM_AGENTS}")

        # --- Model, Optimizer, Loss ---
        model = QNetwork(fov=FOV, USE_NEIGHCOORDS=USE_NEIGHCOORDS).to(DEVICE)
        # model = nn.DataParallel(model, device_ids=[0,1,2,3], output_device=0)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # ------------ read from data.txt generated in C++ -------------
        # logger = Logger()
        # dataset = []
        # data_ = []
        # if not os.path.exists(sample_file + 'samples.pkl'):
        #     with open(os.path.join(os.path.dirname(__file__), sample_file+'data.txt')) as f:
        #         print("Reading from data.txt...")
        #         for line in tqdm(f):
        #             starts_str, goals_str, priorities_str = line.split(';')
        #             starts = ast.literal_eval(starts_str)
        #             goals = ast.literal_eval(goals_str)
        #             priorities = ast.literal_eval(priorities_str.replace(': [', ':['))

        #             row, col = env.grid_map.shape
        #             row = row - 2
        #             col = col - 2
        #             assert (row, col) == (33, 46)
        #             # revert from idx to coord
        #             starts = [(start//col + 1, start%col + 1) for start in starts]
        #             goals = [[(g//col + 1, g%col + 1) for g in goal] for goal in goals]

        #             partial_prio = []
        #             for low, highs in priorities.items():
        #                 for high in highs:
        #                     partial_prio.append((high, low))
        #             data_.append(
        #                 {
        #                     'starts': starts,
        #                     'goals': goals,
        #                     'partial_prio': partial_prio
        #                 }
        #             )
        #         # Initialize label counts
        #         label_counts = Counter()

        #         for d in tqdm(data_):
        #             t0 = time.time()
        #             env.starts = d['starts']
        #             env.goals = d['goals']
        #             priority_order = d['partial_prio']
        #             env.DHC_heur = env._get_DHC_heur()
        #             print(f"_get_DHC_heur time: {time.time() - t0:.3f}s")

        #             t_cp = time.time()
        #             close_pairs = env.get_close_pairs()
        #             print(f"get_close_pairs time: {time.time() - t_cp:.3f}s")

        #             t_obs = time.time()
        #             obs_fovs = env.get_obs()
        #             print(f"get_obs time: {time.time() - t_obs:.3f}s")

        #             t_nf = time.time()
        #             neighbor_features_and_coord = env.get_neighbor_goal_heuristics_as_patches()
        #             print(f"get_neighbor_goal_heuristics time: {time.time() - t_nf:.3f}s")

        #             t_lbl = time.time()
        #             # Build a lookup for fast membership testing
        #             prio_set = set(priority_order)
        #             inv_prio_set = {(b, a) for (a, b) in priority_order}

        #             # Vectorized label assignment
        #             labels_list = [
        #                 0 if pair in prio_set
        #                 else 1 if pair in inv_prio_set
        #                 else 2
        #                 for pair in close_pairs
        #             ]

        #             # Update label counts in one go
        #             label_counts.update(labels_list)

        #             # Convert to tensor
        #             labels = torch.tensor(labels_list, dtype=torch.long)
        #             print(f"label construction time: {time.time() - t_lbl:.3f}s")
        #             dataset.append((obs_fovs, neighbor_features_and_coord, close_pairs, labels))
        #             print(f"Sample preprocessing time: {time.time() - t0:.3f}s")
        #         # Print total label distribution
        #         print(f"Total labels per class: 0={label_counts[0]}, 1={label_counts[1]}, 2={label_counts[2]}")
        #         print("Done reading from data.txt...")

        #     split = int(0.1 * len(dataset))
        #     indices = list(np.random.choice(len(dataset), size=split, replace=False))
        #     train_indices = list(set(range(len(dataset))) - set(indices))
        #     train_set = [dataset[i] for i in train_indices]
        #     test_set = [dataset[i] for i in indices]

        #     # 4. Train for a few epochs
        #     best_acc = train_on_dataset(
        #         model, optimizer, criterion, EPOCHS, sample_file=sample_file, train_set=train_set, test_set=test_set
        #     )
        # else:
        #     best_acc = train_on_dataset(
        #         model, optimizer, criterion, EPOCHS, sample_file=sample_file
        #     )

        best_acc = train_on_dataset(
            env, model, optimizer, criterion, BATCH_SIZE, EPOCHS, writer, model_file, sample_file=sample_file
        )

        writer.close()

        # --- Run Evaluation on Test Set ---
        model = QNetwork(fov=FOV, USE_NEIGHCOORDS=USE_NEIGHCOORDS).to(DEVICE)
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        with open(sample_file + 'test_samples.pkl', 'rb') as f:
            test_set_flatten = pickle.load(f)
        test_loss, test_acc, per_class_acc = evaluate(test_set_flatten, model, BATCH_SIZE, EPOCHS+1, criterion)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    except Exception as e:
        print(f"\nException caught: {e}\nStarting pdb...")
        pdb.post_mortem()
        sys.exit(1)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()