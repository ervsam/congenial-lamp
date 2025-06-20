import os
import sys
import pdb
import glob
import random
import pickle
import time
import yaml
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score

from Environment import Environment
from Model import QNetwork 
from utils import Logger, step, priority_based_search, topological_sort_pbs

# --- Dataset Generation ---
def generate_pbs_data_with_step(env, logger, num_samples, dataset, dataset_path):
    throughput = []
    old_start = env.starts.copy()
    old_goals = env.goals.copy()
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")
        # 1. Get obs_fovs and close_pairs for current env state
        obs_fovs = env.get_obs().cpu()
        neighbor_features = None
        if use_neighbor_features:
            neighbor_features = env.get_neighbor_goal_heuristics_as_patches()
        close_pairs = env.get_close_pairs()
        dummy_qvals = [torch.zeros(len(close_pairs), 2)]

        if not close_pairs:
            # still call step to advance environment even if not saving this instance
            _, _, _, new_start, new_goals, throughput = step(
                env, logger, throughput, q_vals=None, policy='random',
                obs_fovs=obs_fovs, old_start=old_start, old_goals=old_goals, neighbor_features=neighbor_features
            )
            old_start = new_start
            old_goals = new_goals
            continue

        # 2. Call step() to get PBS priorities and partial pairwise ordering (labels)
        #    Set q_vals=None and policy="PBS" to force PBS usage (per your code logic)
        priorities, partial_prio, _, new_start, new_goals, throughput = step(
            env, logger, throughput, q_vals=dummy_qvals, pbs_epsilon=1.0, obs_fovs=obs_fovs,
            old_start=old_start, old_goals=old_goals, neighbor_features=neighbor_features
        )

        # step() returns None if PBS fails; skip if so
        if priorities is None or partial_prio is None:
            old_start = new_start
            old_goals = new_goals
            continue

        plan, priority_order = priority_based_search(env.grid_map, old_start, old_goals, env.window_size)

        # 3. Generate labels for each close pair from PBS priorities
        #    label = 0 if a goes before b, 1 if b goes before a, 2 if they are equal
        labels = []
        for (a, b) in close_pairs:
            if (a, b) in priority_order:
                labels.append(0)
            elif (b, a) in priority_order:
                labels.append(1)
            else:
                labels.append(2)

        labels = torch.tensor(labels, dtype=torch.long)
        if use_neighbor_features:
            dataset.append((obs_fovs, neighbor_features, close_pairs, labels))
        else:
            dataset.append((obs_fovs, close_pairs, labels))

        # 4. Advance the environment
        old_start = new_start
        old_goals = new_goals

        if (i+1) % 500 == 0:
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Generated and saved {len(dataset)} samples to {dataset_path}")

    return dataset

class PairwiseDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        # Adjust as needed for your model's input
        return s

def flatten_samples(newdata, use_neighbor_features):
    samples = []
    for sample in newdata:
        if use_neighbor_features:
            obs_fovs, neighbor_features, close_pairs, labels = sample
        else:
            obs_fovs, close_pairs, labels = sample
            neighbor_features = None
        for i, label in enumerate(labels):
            a, b = close_pairs[i]
            entry = {
                'obs_fov': torch.stack([obs_fovs[a], obs_fovs[b]]),  # shape (2, C, H, W)
                'neighbor_features': [torch.stack(neighbor_features[a]), torch.stack(neighbor_features[b])] if neighbor_features is not None else None,
                'close_pair': (a, b),
                'label': label.item() if torch.is_tensor(label) else int(label)
            }
            samples.append(entry)
    return samples

def custom_collate(batch):
        obs_pair = torch.stack([item['obs_fov'] for item in batch])  # (batch, 2, C, H, W)
        if use_neighbor_features:
            neigh = [item['neighbor_features'] for item in batch] # (batch, 2, tensor('num_neigh', 1, F, F))
        else:
            neigh = None
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long) # batch
        return obs_pair, neigh, labels

# --- Training Loop ---
def train_on_dataset(train_set, test_set, model, optimizer, criterion, start_epoch, epochs_per_cycle, best_acc, best_f1=0):
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

    samples = flatten_samples(train_set, use_neighbor_features)

    # oversample
    label_counts = Counter([entry['label'] for entry in samples])
    class_weights = {label: 1.0/count for label, count in label_counts.items()}
    sample_weights = [class_weights[entry['label']] for entry in samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(samples),  # You can also try int(1.5 * len(samples)) for more epochs per epoch
        replacement=True
    )
    # undersample
    # samples = undersample_samples(samples)
    pair_dataset = PairwiseDataset(samples)
    
    balanced_loader = DataLoader(
        pair_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=custom_collate,
        # shuffle=True,
        # num_workers=4,
        # pin_memory=True
    )
    print(f"Number of batches: {len(balanced_loader)}")

    all_labels = [lab for sam in train_set for lab in sam[label_idx]]
    print("Train label 0:", all_labels.count(0), "1:", all_labels.count(1), "2:", all_labels.count(2))
    all_labels = [lab for sam in test_set for lab in sam[label_idx]]
    print("Test label 0:", all_labels.count(0), "1:", all_labels.count(1), "2:", all_labels.count(2))

    model.train()
    for epoch in range(start_epoch, start_epoch + epochs_per_cycle):
        print(f"Training epoch {epoch + 1}/{start_epoch + epochs_per_cycle}...")
        total_loss = 0
        total_correct = 0
        total_pred = 0
        all_preds = []
        all_labels = []

        for batch_n, batch in enumerate(balanced_loader):
            t0 = time.time()
            obs_fovs_batch, neighbor_features_batch, labels_batch = batch
            obs_fovs_batch = obs_fovs_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            # shapes
            # obs_fovs_batch: torch.Size([64, 2, 8, 11, 11])
            # neighbor_features_batch: (64, 2, tensor('num_neigh', 1, 11, 11))
            # labels_batch: (64)

            # label_counts = torch.bincount(torch.tensor(labels_batch))
            # label_ratios = label_counts.float() / label_counts.sum()
            # print("labels_batch count:", label_counts.tolist())
            # print("labels_batch ratio:", [round(r.item(), 3) for r in label_ratios])
            # print("labels_batch (non-tensor):", labels_batch)

            t1 = time.time()
            
            _, batch_q_vals = model(obs_fovs_batch, [], neighbor_features_batch)
            batch_q_vals_flat = batch_q_vals
            loss = criterion(batch_q_vals_flat, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t3 = time.time()

            pred = torch.argmax(batch_q_vals_flat, dim=1)
            correct = (pred == labels_batch).sum().item()
            total_correct += correct
            total_pred += labels_batch.size(0)
            total_loss += loss.item()

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels_batch.cpu().tolist())

            # print(f"Batch {batch_n}: load={t1-t0:.3f}s, forward+back={t3-t1:.3f}s")

        accuracy = total_correct / total_pred if total_pred > 0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_set):.4f} | Accuracy: {accuracy:.4f} | F1: {macro_f1:.4f}")

        test_set_flatten = flatten_samples(test_set, use_neighbor_features)
        test_loss, test_acc, macro_f1 = evaluate(test_set_flatten, model, BATCH_SIZE)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | F1: {macro_f1:.4f}")

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     print(f"New best accuracy: {best_acc:.4f}, saving model...")
        #     torch.save(model.state_dict(), model_file)
        #     print(f"Model saved as {model_file}")
        # else:
        #     print(f"Best accuracy is still {best_acc:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), model_file)
            print(f"New best macro F1: {best_f1:.4f}, saving model...")
        else:
            print(f"Best F1 is still {best_f1:.4f}")

        print()
    return best_acc

# --- Evaluation ---
def evaluate(dataset, model, batch_size=32):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pred = 0
    criterion = nn.CrossEntropyLoss()
    DEVICE = next(model.parameters()).device
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if not batch:
                continue
            obs_pair, neigh, labels = custom_collate(batch)
            obs_pair = obs_pair.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            logits = model(obs_pair, [], neigh)
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

    # all_labels and all_preds should be 1D arrays/lists of ints
    assert isinstance(all_labels, list) and isinstance(all_preds, list), "Should be lists"
    assert all(isinstance(x, int) for x in all_labels), "all_labels not all ints"
    assert all(isinstance(x, int) for x in all_preds), "all_preds not all ints"
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    print(f"Per-class F1 score: ", end='')
    for f1 in per_class_f1:
        print(f"{f1:.2f}, ", end='')
    print()

    return avg_loss, accuracy, macro_f1

def generate_new_samples(env, logger, n_samples, use_neighbor_features=True):
    print("Generating PBS-labeled dataset...")
    old_start = env.starts.copy()
    old_goals = env.goals.copy()
    dataset = []
    for i in range(n_samples):
        print(f"Generating sample {i+1}/{n_samples}...")
        t0 = time.time()

        # 1. Get obs_fovs and close_pairs for current env state
        obs_fovs = env.get_obs().cpu()
        t1 = time.time()
        neighbor_features = None
        if use_neighbor_features:
            neighbor_features = env.get_neighbor_goal_heuristics_as_patches()
        t2 = time.time()
        close_pairs = env.get_close_pairs()
        dummy_qvals = [torch.zeros(len(close_pairs), 2)]
        t3 = time.time()

        if not close_pairs:
            step_t0 = time.time()
            # still call step to advance environment even if not saving this instance
            _, _, _, new_start, new_goals, throughput = step(
                env, logger, [], q_vals=None, policy='random',
                obs_fovs=obs_fovs, old_start=old_start, old_goals=old_goals, neighbor_features=neighbor_features
            )
            step_t1 = time.time()
            print(f"no pairs: obs={t1-t0:.3f}s, neigh={t2-t1:.3f}s, close_pairs={t3-t2:.3f}s, step={step_t1-step_t0:.3f}s")
            old_start = new_start
            old_goals = new_goals
            continue

        # 2. Call step() to get PBS priorities and partial pairwise ordering (labels)
        #    Set q_vals=None and policy="PBS" to force PBS usage (per your code logic)
        step1_t0 = time.time()
        priorities, partial_prio, _, new_start, new_goals, throughput = step(
            env, logger, [], q_vals=dummy_qvals, pbs_epsilon=1.0, obs_fovs=obs_fovs,
            old_start=old_start, old_goals=old_goals, neighbor_features=neighbor_features
        )

        # step() returns None if PBS fails; skip if so
        if priorities is None or partial_prio is None:
            old_start = new_start
            old_goals = new_goals
            continue

        step1_t1 = time.time()
        plan_t0 = time.time()
        plan, priority_order = priority_based_search(env.grid_map, old_start, old_goals, env.window_size)
        plan_t1 = time.time()

        print(f"obs={t1-t0:.3f}s, neigh={t2-t1:.3f}s, close_pairs={t3-t2:.3f}s, "
            f"step={step1_t1-step1_t0:.3f}s, priority_search={plan_t1-plan_t0:.3f}s")

        # 3. Generate labels for each close pair from PBS priorities
        #    label = 0 if a goes before b, 1 if b goes before a, 2 if they are equal
        labels = []
        for (a, b) in close_pairs:
            if (a, b) in priority_order:
                labels.append(0)
            elif (b, a) in priority_order:
                labels.append(1)
            else:
                labels.append(2)
        labels = torch.tensor(labels, dtype=torch.long)
        if use_neighbor_features:
            dataset.append((obs_fovs, neighbor_features, close_pairs, labels))
        else:
            dataset.append((obs_fovs, close_pairs, labels))

        old_start, old_goals = new_start, new_goals
    return dataset

try:
    # --- Configurations ---
    CONFIG_NAME = "warehouse_2"
    CONFIG_FILE = "config.yaml"
    DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    LR = 1e-4
    FOV = 21

    EPOCHS = 100
    N_SAMPLES = 10

    use_neighbor_features = True
    model_file = "delete.pth"
    label_idx = 3

    # --- Load or Generate Dataset ---
    dataset = []
    dataset_path = 'delete.pkl'
    EPOCHS_PER_CYCLE = 3     # Train for 3 epochs each time you add new samples
    best_acc = 0
    epoch_counter = 0

    # --- Load Config and Initialize Environment ---
    with open(os.path.join(os.path.dirname(__file__), CONFIG_FILE), "r") as file:
        config_file = yaml.safe_load(file)
    config = config_file[CONFIG_NAME]
    env_config = config["environment"]

    env = Environment(
        env_config,
        logger=Logger(),  # Dummy logger
        grid_map_file=config["paths"]["map_file"],
        heuristic_map_file=config["paths"]["heur_file"]
    )

    # --- Model, Optimizer, Loss ---
    model = QNetwork(fov=FOV).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ------------ generate data -------------
    logger = Logger()
    new_samples = generate_new_samples(env, logger, n_samples=N_SAMPLES, use_neighbor_features=use_neighbor_features)
    dataset.extend(new_samples)

    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)

    split = int(0.2 * len(dataset))
    indices = np.random.choice(len(dataset), size=split, replace=False)
    train_indices = list(set(range(len(dataset))) - set(indices))
    train_set = [dataset[i] for i in train_indices]
    test_set = [dataset[i] for i in indices]

    # 4. Train for a few epochs
    best_acc = train_on_dataset(
        train_set, test_set, model, optimizer, criterion, epoch_counter, EPOCHS, best_acc
    )
    
    # NUM_CYCLES = 0
    # for cycle in range(NUM_CYCLES):
    #     print(f"--- Cycle {cycle + 1} ---")
    #     # 1. Generate new data
    #     new_samples = generate_new_samples(env, logger, n_samples=N_SAMPLES, use_neighbor_features=use_neighbor_features)
    #     dataset.extend(new_samples)   # or save to disk

    #     # 2. (Optional) Save dataset
    #     with open(dataset_path, 'wb') as f:
    #         pickle.dump(dataset, f)

    #     # 3. Re-split train/test if needed
    #     split = int(0.2 * len(dataset))
    #     indices = np.random.choice(len(dataset), size=split, replace=False)
    #     train_indices = list(set(range(len(dataset))) - set(indices))
    #     train_set = [dataset[i] for i in train_indices]
    #     test_set = [dataset[i] for i in indices]

    #     # 4. Train for a few epochs
    #     best_acc = train_on_dataset(
    #         train_set, test_set, model, optimizer, criterion, epoch_counter, EPOCHS_PER_CYCLE, best_acc
    #     )
    #     epoch_counter += EPOCHS_PER_CYCLE

    # ------------- Training on each batch ------------------
    # BATCHED_DATASET_DIR = "./batched_dataset_1000/" 
    # BATCH_PATTERN = os.path.join(os.path.dirname(__file__), os.path.join(BATCHED_DATASET_DIR, "batch_*.pkl"))
    # batch_files = sorted(glob.glob(BATCH_PATTERN))
    # print(f"Found {len(batch_files)} batch files.")

    # for batch_file in batch_files[:1]:
    #     batch_file = "delete.pkl"
    #     print(f"Loading batch: {batch_file}")
    #     with open(os.path.join(os.path.dirname(__file__), batch_file), "rb") as f:
    #         batch_data = pickle.load(f)
    #     print(f"Loaded {len(batch_data)} samples from {batch_file}")

    #     # Optionally: split within each batch
    #     split = int(0.2 * len(batch_data))
    #     indices = np.random.choice(len(batch_data), size=split, replace=False)
    #     train_indices = list(set(range(len(batch_data))) - set(indices))
    #     train_set = [batch_data[i] for i in train_indices]
    #     test_set = [batch_data[i] for i in indices]

    #     best_acc = train_on_dataset(
    #         train_set, test_set, model, optimizer, criterion, epoch_counter, EPOCHS, best_acc
    #     )
    #     epoch_counter += EPOCHS_PER_CYCLE


    print(f"Loading batch: {dataset_path}")
    with open(os.path.join(os.path.dirname(__file__), dataset_path), "rb") as f:
        dataset = pickle.load(f)
    print(f"Loaded {len(dataset)} samples from {dataset_path}")

    # split = int(0.2 * len(dataset))
    # indices = np.random.choice(len(dataset), size=split, replace=False)
    # test_set = [dataset[i] for i in indices]

    # --- Load Trained Model ---
    model = QNetwork(fov=FOV).to(DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    # --- Run Evaluation on Test Set ---
    test_set_flatten = flatten_samples(test_set, use_neighbor_features)
    test_loss, test_acc, macro_f1 = evaluate(test_set_flatten, model, BATCH_SIZE)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | F1: {macro_f1:.4f}")

except Exception as e:
    print(f"\nException caught: {e}\nStarting pdb...")
    pdb.post_mortem()
    sys.exit(1)