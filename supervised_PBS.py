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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score

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
        # Adjust as needed for your model's input
        return s

def flatten_samples(newdata, use_neighbor_features):
    samples = []
    for sample in newdata:
        if use_neighbor_features:
            obs_fovs, neighbor_features, close_pairs, labels = sample
            # Pre-stack neighbor features for all agents only once per sample
            stacked_neigh = [
                torch.stack([n[0] for n in neigh]) if len(neigh) > 0 else None
                for neigh in neighbor_features
            ]
            stacked_coords = [[n[1] for n in neigh] for neigh in neighbor_features]
        else:
            obs_fovs, close_pairs, labels = sample
            neighbor_features = None
            stacked_neigh = None
            stacked_coords = None

        for i, label in enumerate(labels):
            a, b = close_pairs[i]
            entry = {
                'obs_fov': torch.stack([obs_fovs[a], obs_fovs[b]]),  # shape (2, C, H, W)
                'neighbor_features': [stacked_neigh[a], stacked_neigh[b]] if use_neighbor_features else None,
                'close_pair': (a, b),
                'label': label.item() if torch.is_tensor(label) else int(label),
                'neigh_coords': [stacked_coords[a], stacked_coords[b]] if use_neighbor_features else None,
            }
            samples.append(entry)
    return samples
# OLD (DELETE)
# def flatten_samples(newdata, use_neighbor_features):
#     samples = []
#     for sample in newdata:
#         if use_neighbor_features:
#             obs_fovs, neighbor_features, close_pairs, labels = sample
#         else:
#             obs_fovs, close_pairs, labels = sample
#             neighbor_features = None
#         for i, label in enumerate(labels):
#             a, b = close_pairs[i]
#             entry = {
#                 'obs_fov': torch.stack([obs_fovs[a], obs_fovs[b]]),  # shape (2, C, H, W)
#                 'neighbor_features': [torch.stack([n[0] for n in neighbor_features[a]]), torch.stack([n[0] for n in neighbor_features[b]])],
#                 'close_pair': (a, b),
#                 'label': label.item() if torch.is_tensor(label) else int(label),
#                 'neigh_coords': [[n[1] for n in neighbor_features[a]], [n[1] for n in neighbor_features[b]]],
#             }
#             samples.append(entry)
#     return samples

def custom_collate(batch):
        obs_pair = torch.stack([item['obs_fov'] for item in batch])  # (batch, 2, C, H, W)
        if use_neighbor_features:
            neigh = [item['neighbor_features'] for item in batch] # (batch, 2, tensor('num_neigh', 1, F, F))
            neigh_coords = [torch.stack(item).to(DEVICE) for sublist in batch for item in sublist['neigh_coords']] # list[batch*2, tensor('num_neigh', 2)]
        else:
            neigh = None
        
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long) # batch
        return obs_pair, neigh, neigh_coords, labels

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


    sample_file = os.path.join(os.path.dirname(__file__), 'data_gen/70/w20/samples.pkl')

    if os.path.exists(sample_file):
        print(f"Loading samples from {sample_file}...")
        with open(sample_file, 'rb') as f:
            samples = pickle.load(f)
        print(f"Loaded {len(samples)} samples.")
    else:
        t0 = time.time()
        samples = flatten_samples(train_set, use_neighbor_features)
        print(f"Flattened {len(samples)} samples in {time.time() - t0:.3f}s")

        # undersample
        samples = undersample_samples(samples)
        with open(sample_file, 'wb') as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(samples)} samples to {sample_file}.")

    pair_dataset = PairwiseDataset(samples)
    
    balanced_loader = DataLoader(
        pair_dataset,
        batch_size=BATCH_SIZE,
        # sampler=sampler,
        collate_fn=custom_collate,
        shuffle=True,
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
            obs_fovs_batch, neighbor_features_batch, neigh_coords_batch, labels_batch = batch
            obs_fovs_batch = obs_fovs_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            # shapes
            # obs_fovs_batch: torch.Size([64, 2, 8, 11, 11])
            # neighbor_features_batch: (64, 2, tensor('num_neigh', 1, 11, 11))
            # labels_batch: (64)

            t1 = time.time()
            
            _, batch_q_vals = model(obs_fovs_batch, [], neighbor_features_batch, neigh_coords_batch)
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

            print(f"Batch {batch_n}: load={t1-t0:.3f}s, forward+back={t3-t1:.3f}s")

        accuracy = total_correct / total_pred if total_pred > 0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        num_classes = len(set(all_labels_np))

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_set):.4f} | Accuracy: {accuracy:.4f} | F1: {macro_f1:.4f}")
        writer.add_scalar('Loss/train', total_loss / len(train_set), epoch)
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

        test_set_flatten = flatten_samples(test_set, use_neighbor_features)
        test_loss, test_acc, per_class_acc = evaluate(test_set_flatten, model, BATCH_SIZE, epoch)
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

        # if macro_f1 > best_f1:
        #     best_f1 = macro_f1
        #     torch.save(model.state_dict(), model_file)
        #     print(f"New best macro F1: {best_f1:.4f}, saving model...")
        # else:
        #     print(f"Best F1 is still {best_f1:.4f}")

        print()
    return best_acc

# --- Evaluation ---
def evaluate(dataset, model, batch_size, epoch):
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
            obs_pair, neigh, neigh_coords, labels = custom_collate(batch)
            obs_pair = obs_pair.to(DEVICE)
            labels = labels.to(DEVICE)

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

def generate_new_samples(env, logger, n_samples, use_neighbor_features=True):
    print("Generating PBS-labeled dataset...")
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
        t3 = time.time()

        if not close_pairs:
            step_t0 = time.time()
            # still call step to advance environment even if not saving this instance
            priorities, priority_order = step(env, logger)
            step_t1 = time.time()
            print(f"no pairs: obs={t1-t0:.3f}s, neigh={t2-t1:.3f}s, close_pairs={t3-t2:.3f}s, step={step_t1-step_t0:.3f}s")
            continue

        step1_t0 = time.time()
        priorities, priority_order = step(env, logger)
        step1_t1 = time.time()
        if priorities is None:
            continue

        print(f"obs={t1-t0:.3f}s, neigh={t2-t1:.3f}s, close_pairs={t3-t2:.3f}s, step={step1_t1-step1_t0:.3f}s")

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

        print()

    return dataset

try:
    # --- Configurations ---
    CONFIG_NAME = "warehouse_2"
    CONFIG_FILE = "config.yaml"
    DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    LR = 1e-4
    WINDOW_SIZE = 20
    FOV = WINDOW_SIZE * 4 + 1

    EPOCHS = 100
    N_SAMPLES = 0

    use_neighbor_features = True
    model_file = "sup_pbs_neighcoord_70_w20.pth"
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

    gen_data = False
    if gen_data:
        BATCHED_DATASET_DIR = "./dataset/" 
        BATCH_PATTERN = os.path.join(os.path.dirname(__file__), os.path.join(BATCHED_DATASET_DIR, "data_batch_*.pkl"))
        batch_files = sorted(glob.glob(BATCH_PATTERN))
        print(f"Found {len(batch_files)} batch files.")

        for batch_file in batch_files:
            print(f"Loading batch: {batch_file}")
            with open(os.path.join(os.path.dirname(__file__), batch_file), "rb") as f:
                batch_data = pickle.load(f)
            print(f"Loaded {len(batch_data)} samples from {batch_file}")
            dataset.extend(batch_data)

        os.makedirs('dataset', exist_ok=True)
        samples_per_batch = 500
        total_samples = len(dataset)
        batch_idx = 0
        while total_samples < N_SAMPLES:
            this_batch = min(samples_per_batch, N_SAMPLES - total_samples)
            total_samples += this_batch
            batch_idx += 1
            print(f"Generating batch {batch_idx}, {this_batch} samples...")
            t_gensamples = time.time()
            new_samples = generate_new_samples(env, logger, n_samples=this_batch, use_neighbor_features=use_neighbor_features)
            print(f"Time to generate {this_batch}: {time.time() - t_gensamples:.3f}s")

            # Save the current chunk to a separate file
            batch_path = os.path.join('dataset', f'data_batch_{total_samples:05d}.pkl')
            with open(batch_path, 'wb') as f:
                pickle.dump(new_samples, f)
            print(f"Saved batch {batch_idx} to {batch_path}")

            dataset.extend(new_samples)
    else:
        # read from data.txt generated in C++
        data_ = []
        with open(os.path.join(os.path.dirname(__file__), 'data_gen/70/w20/data.txt')) as f:
            print("Reading from data.txt...")
            for line in tqdm(f):
                starts_str, goals_str, priorities_str = line.split(';')
                starts = ast.literal_eval(starts_str)
                goals = ast.literal_eval(goals_str)
                priorities = ast.literal_eval(priorities_str.replace(': [', ':['))  # handle optional space

                row, col = env.grid_map.shape
                row = row - 2
                col = col - 2
                assert (row, col) == (33, 46)

                # revert from idx to coord
                starts = [(start//col + 1, start%col + 1) for start in starts]

                # ONLY SAVE THE FIRST GOAL FOR NOW, because we dont use the following goals in our input
                goals = [[(g//col + 1, g%col + 1) for g in goal[:1]] for goal in goals]

                partial_prio = []
                for low, highs in priorities.items():
                    for high in highs:
                        partial_prio.append((high, low))

                data_.append(
                    {
                        'starts': starts,
                        'goals': goals,
                        'partial_prio': partial_prio
                    }
                )
            
            for d in tqdm(data_):
                env.starts = d['starts']
                env.goals = d['goals']
                priority_order = d['partial_prio']
                env.DHC_heur = env._get_DHC_heur()

                close_pairs = env.get_close_pairs()
                obs_fovs = env.get_obs()
                neighbor_features_and_coord = env.get_neighbor_goal_heuristics_as_patches()

                labels = []
                for (a, b) in close_pairs:
                    if (a, b) in priority_order:
                        labels.append(0)
                    elif (b, a) in priority_order:
                        labels.append(1)
                    else:
                        labels.append(2)
                labels = torch.tensor(labels, dtype=torch.long)
                dataset.append((obs_fovs, neighbor_features_and_coord, close_pairs, labels))

            print("Done reading from data.txt...")
            

    split = int(0.2 * len(dataset))
    indices = list(np.random.choice(len(dataset), size=split, replace=False))
    train_indices = list(set(range(len(dataset))) - set(indices))
    train_set = [dataset[i] for i in train_indices]
    test_set = [dataset[i] for i in indices]

    # 4. Train for a few epochs
    writer = SummaryWriter(log_dir="runs/with_neighcoords/w20/70")
    best_acc = train_on_dataset(
        train_set, test_set, model, optimizer, criterion, epoch_counter, EPOCHS, best_acc
    )
    writer.close()

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
    test_loss, test_acc, per_class_acc = evaluate(test_set_flatten, model, BATCH_SIZE, EPOCHS+1)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

except Exception as e:
    print(f"\nException caught: {e}\nStarting pdb...")
    pdb.post_mortem()
    sys.exit(1)