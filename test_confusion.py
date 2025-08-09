# %%
# test_confusion.py
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from utils import Logger
from tqdm import tqdm
import time

from supervised_PBS import PairDataset, custom_collate, QNetwork
from Environment import Environment

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
# 

# --- Config & paths (edit as needed) ---
CONFIG_NAME   = "warehouse_2"
CONFIG_FILE   = "config.yaml"
MODEL_PATH    = "sup_pbs_140_w20_binary.pth"  # path to your saved model
TEST_DATA_DIR = os.path.join("data_gen", "140", "w20")  # your sample_file root
TEST_FILE     = os.path.join(TEST_DATA_DIR, "test_data.txt")
BATCH_SIZE    = 64

# --- Load config & env ---
cfg = yaml.safe_load(open(CONFIG_FILE))[CONFIG_NAME]
env_cfg = cfg["environment"]
env = Environment(
    env_cfg,
    logger=Logger(),
    grid_map_file=cfg["paths"]["map_file"],
    heuristic_map_file=cfg["paths"]["heur_file"],
    device=device
)

# --- Prepare test DataLoader ---
test_ds = PairDataset(TEST_FILE, env, undersample=True)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda batch, env=env: custom_collate(batch, env),
    # pin_memory=True,
)

# --- Build & load model ---
model = QNetwork(fov=env_cfg["FOV"], USE_NEIGHCOORDS=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Run inference ---
all_trues = []
all_preds = []
with torch.no_grad():
    for obs, neigh_feats, neigh_coords, labels, mask in tqdm(test_loader):
        t0 = time.perf_counter()
        obs = obs.to(device, non_blocking=True)
        # print(f"obs.to: {time.perf_counter()-t0:.6f}s")

        t1 = time.perf_counter()
        nf = neigh_feats.to(device, non_blocking=True)
        # print(f"neigh_feats.to: {time.perf_counter()-t1:.6f}s")

        t2 = time.perf_counter()
        nc = neigh_coords.to(device, non_blocking=True)
        # print(f"neigh_coords.to: {time.perf_counter()-t2:.6f}s")

        t3 = time.perf_counter()
        m = mask.to(device, non_blocking=True)
        # print(f"mask.to: {time.perf_counter()-t3:.6f}s")

        t4 = time.perf_counter()
        lbls = labels.to(device, non_blocking=True)
        # print(f"labels.to: {time.perf_counter()-t4:.6f}s")

        t5 = time.perf_counter()
        outputs = model(obs, nf, nc, m)
        # print(f"model forward: {time.perf_counter()-t5:.6f}s")

        if isinstance(outputs, tuple):
            qvals = outputs[-1]
        else:
            qvals = outputs

        t6 = time.perf_counter()
        preds = torch.argmax(qvals, dim=1)
        # print(f"argmax: {time.perf_counter()-t6:.6f}s")

        t7 = time.perf_counter()
        all_trues.extend(lbls.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        # print(f"cpu & extend: {time.perf_counter()-t7:.6f}s")

# --- Confusion matrix ---
cm = confusion_matrix(all_trues, all_preds, labels=[0,1,2])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["pred 0","pred 1","pred 2"],
            yticklabels=["true 0","true 1","true 2"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
# plt.show()

# --- Classification report ---
print("\nClassification Report:\n")
print(classification_report(
    all_trues,
    all_preds,
    labels=[0,1,2],
    target_names=["0: A→B", "1: B→A", "2: no-priority"]
))

# %%
