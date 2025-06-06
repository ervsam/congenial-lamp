import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from Environment import Environment
from Model import QNetwork  # Replace with your model class if named differently
from utils import Logger
from utils import step, priority_based_search, topological_sort_pbs
import pickle
import os
import pdb

# --- Configurations ---
CONFIG_NAME = "warehouse_2"
CONFIG_FILE = "config.yaml"
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 30
N_SAMPLES = 100000

use_neighbor_features = True
model_file = "supervised_pbs_model_best_partialorderlabels.pth"
label_idx = 3

# --- Load or Generate Dataset ---
dataset = []
dataset_path = 'supervised_pbs_dataset_partialorderlabels.pkl'
EPOCHS_PER_CYCLE = 3     # Train for 3 epochs each time you add new samples
CYCLE_SIZE = 100
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
model = QNetwork().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()  # 2-class per pair

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

        # 3. Generate labels for each close pair from PBS priorities
        #    label = 0 if a goes before b, 1 otherwise
        labels = []
        for (a, b) in close_pairs:
            if priorities.index(a) < priorities.index(b):
                labels.append(0)
            else:
                labels.append(1)
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

# --- Training Loop ---
def train_on_dataset(train_set, test_set, model, optimizer, criterion, start_epoch, epochs_per_cycle, best_acc):
    model.train()
    for epoch in range(start_epoch, start_epoch + epochs_per_cycle):
        print(f"Training epoch {epoch + 1}/{start_epoch + epochs_per_cycle}...")
        total_loss = 0
        total_correct = 0
        total_pred = 0
        for i in range(0, len(train_set), BATCH_SIZE):
            batch = train_set[i:i+BATCH_SIZE]
            if not batch:
                continue
            if use_neighbor_features:
                obs_fovs_batch, neighbor_features_batch, close_pairs_batch, labels_batch = zip(*batch)
            else:
                obs_fovs_batch, close_pairs_batch, labels_batch = zip(*batch)
                neighbor_features_batch = None
            obs_fovs_batch = torch.stack(obs_fovs_batch).to(DEVICE)

            if use_neighbor_features:
                flat = [n for neigh_f in neighbor_features_batch for n_i in neigh_f for n in n_i]
                if flat:
                    stacked = torch.stack(flat).to(DEVICE)
                    idx = 0
                    rebuilt = []
                    for batch in neighbor_features_batch:
                        batch_rebuilt = []
                        for agent in batch:
                            n = len(agent)
                            agent_rebuilt = [stacked[idx + i] for i in range(n)]
                            idx += n
                            if n == 0:
                                # Replace (1, C, H, W) with your patch shape, e.g., (1, 1, 11, 11)
                                batch_rebuilt.append(torch.empty(0, *stacked.shape[1:], device=DEVICE))
                            else:
                                batch_rebuilt.append(torch.stack(agent_rebuilt))
                        rebuilt.append(batch_rebuilt)
                    neighbor_features_batch = rebuilt

            labels_batch = [lab for labs in labels_batch for lab in labs]
            labels_batch = torch.tensor(labels_batch, dtype=torch.long, device=DEVICE)

            _, batch_q_vals = model(obs_fovs_batch, close_pairs_batch, neighbor_features_batch)
            batch_q_vals_flat = torch.cat(batch_q_vals, dim=0)
            loss = criterion(batch_q_vals_flat, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(batch_q_vals_flat, dim=1)
            correct = (pred == labels_batch).sum().item()
            total_correct += correct
            total_pred += labels_batch.size(0)
            total_loss += loss.item()

        accuracy = total_correct / total_pred if total_pred > 0 else 0.0
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_set):.4f} | Accuracy: {accuracy:.4f}")

        test_loss, test_acc = evaluate(test_set, model, BATCH_SIZE)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            print(f"New best accuracy: {best_acc:.4f}, saving model...")
            torch.save(model.state_dict(), model_file)
            print(f"Model saved as {model_file}")
    return best_acc

# --- Evaluation ---
def evaluate(dataset, model, batch_size=32):
    total_loss = 0
    total_correct = 0
    total_pred = 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if not batch:
                continue
            if use_neighbor_features:
                obs_fovs_batch, neighbor_features_batch, close_pairs_batch, labels_batch = zip(*batch)
            else:
                obs_fovs_batch, close_pairs_batch, labels_batch = zip(*batch)
                neighbor_features_batch = None
            obs_fovs_batch = torch.stack(obs_fovs_batch).to(DEVICE)

            if use_neighbor_features:
                # Fast device transfer for neighbor_features_batch (list of lists of tensors)
                flat = [n for neigh_f in neighbor_features_batch for n_i in neigh_f for n in n_i]
                if flat:
                    stacked = torch.stack(flat).to(DEVICE)
                    idx = 0
                    rebuilt = []
                    for batch in neighbor_features_batch:
                        batch_rebuilt = []
                        for agent in batch:
                            n = len(agent)
                            agent_rebuilt = [stacked[idx + i] for i in range(n)]
                            idx += n
                            if n == 0:
                                # Replace (1, C, H, W) with your patch shape, e.g., (1, 1, 11, 11)
                                batch_rebuilt.append(torch.empty(0, *stacked.shape[1:], device=DEVICE))
                            else:
                                batch_rebuilt.append(torch.stack(agent_rebuilt))
                        rebuilt.append(batch_rebuilt)
                    neighbor_features_batch = rebuilt

            all_close_pairs = [list(pairs) for pairs in close_pairs_batch]
            labels_batch = [lab for labs in labels_batch for lab in labs]
            labels_batch = torch.tensor(labels_batch, dtype=torch.long, device=DEVICE)

            # Forward Pass
            _, batch_q_vals = model(obs_fovs_batch, all_close_pairs, neighbor_features_batch)
            batch_q_vals_flat = torch.cat(batch_q_vals, dim=0)

            loss = criterion(batch_q_vals_flat, labels_batch)

            pred = torch.argmax(batch_q_vals_flat, dim=1)
            correct = (pred == labels_batch).sum().item()
            total_correct += correct
            total_pred += labels_batch.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    accuracy = total_correct / total_pred if total_pred > 0 else 0.0
    return avg_loss, accuracy


if os.path.exists(os.path.join(os.path.dirname(__file__), dataset_path)):
    with open(os.path.join(os.path.dirname(__file__), dataset_path), 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded {len(dataset)} samples from {dataset_path}")

print("Generating PBS-labeled dataset...")
old_len = len(dataset)

split = int(0.2 * len(dataset))
indices = np.random.choice(len(dataset), size=split, replace=False)
train_indices = list(set(range(len(dataset))) - set(indices))
train_set = [dataset[i] for i in train_indices]
test_set = [dataset[i] for i in indices]

logger = Logger()
throughput = []
old_start = env.starts.copy()
old_goals = env.goals.copy()
for i in range(old_len, N_SAMPLES):
    print(f"Generating sample {i+1}/{N_SAMPLES}...")

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

    plan, priority_order = priority_based_search(env.grid_map, old_start, old_goals, env.window_size)

    # step() returns None if PBS fails; skip if so
    if priorities is None or partial_prio is None:
        old_start = new_start
        old_goals = new_goals
        continue

    # 3. Generate labels for each close pair from PBS priorities
    #    label = 0 if a goes before b, 1 otherwise
    labels = []
    for (a, b) in close_pairs:
        if priorities.index(a) < priorities.index(b):
            labels.append(0)
        else:
            labels.append(1)
    labels = torch.tensor(labels, dtype=torch.long)
    if use_neighbor_features:
        dataset.append((obs_fovs, neighbor_features, close_pairs, labels))
    else:
        dataset.append((obs_fovs, close_pairs, labels))

    # After each new sample, periodically save and train
    if (i + 1) % CYCLE_SIZE == 0 or (i + 1) == N_SAMPLES:
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Generated and saved {len(dataset)} samples to {dataset_path}")

        split = int(0.2 * len(dataset))
        existing = set(indices)
        needed = split - len(indices)
        if needed > 0:
            available = list(set(range(len(dataset))) - existing)
            indices = np.concatenate([indices, np.random.choice(available, size=needed, replace=False)])

        train_indices = list(set(range(len(dataset))) - set(indices))
        train_set = [dataset[i] for i in train_indices]
        test_set = [dataset[i] for i in indices]

        all_labels = [lab for sample in test_set for lab in sample[label_idx]]
        print("Test label 0:", all_labels.count(0), "1:", all_labels.count(1))
        print(f"Train set size: {len(train_set)}. Test set size: {len(test_set)}")

        # Train for a few epochs
        best_acc = train_on_dataset(
            train_set, test_set, model, optimizer, criterion, epoch_counter, EPOCHS_PER_CYCLE, best_acc
        )
        epoch_counter += EPOCHS_PER_CYCLE

split = int(0.2 * len(dataset))
indices = np.random.choice(len(dataset), size=split, replace=False)
train_set = dataset[~np.isin(np.arange(len(dataset)), indices)]
test_set = dataset[indices]

best_acc = train_on_dataset(
    train_set, test_set, model, optimizer, criterion, epoch_counter, EPOCHS, best_acc
)

# --- Load Trained Model ---
model = QNetwork().to(DEVICE)
model.load_state_dict(torch.load(model_file, map_location=DEVICE))
model.eval()

criterion = nn.CrossEntropyLoss()

# --- Run Evaluation on Test Set ---
test_loss, test_acc = evaluate(test_set, model, BATCH_SIZE)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")