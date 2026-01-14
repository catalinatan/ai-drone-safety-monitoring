import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import timm
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets

# --- Model ---
model = timm.create_model(
    'mobilenetv3_large_100',
    pretrained=True,
    num_classes=3
)

model.train()

# --- Transforms --- 
cfg = timm.data.resolve_data_config(model.pretrained_cfg)
train_tfms = timm.data.create_transform(**cfg, is_training=True)
eval_tfms  = timm.data.create_transform(**cfg, is_training=False)

# --- Paths ---
grandparent_dir = Path(__file__).resolve().parents[2]
data_dir = os.path.join(grandparent_dir, "data")

# --- Dataset Filtering ---
base = datasets.ImageFolder(root=str(data_dir))
keep_classes = ["ships", "bridges", "railways"]
keep_ids = [base.class_to_idx[c] for c in keep_classes if c in base.class_to_idx]
kept_indices = [i for i, y in enumerate(base.targets) if y in keep_ids]

# --- Stratified train/val/test split ---
id_map = {orig: new for new, orig in enumerate(keep_ids)}
kept_labels = np.array([id_map[base.targets[i]] for i in kept_indices], dtype=np.int64)

rng = np.random.default_rng(42)
kept_indices = np.array(kept_indices, dtype=int)

train_idx, val_idx, test_idx = [], [], []
train_ratio, val_ratio = 0.8, 0.1  # test = remainder

for c in np.unique(kept_labels):
    cls_idx = kept_indices[kept_labels == c].copy()
    rng.shuffle(cls_idx)

    n = len(cls_idx)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train_idx.append(cls_idx[:n_train])
    val_idx.append(cls_idx[n_train:n_train + n_val])
    test_idx.append(cls_idx[n_train + n_val:])

train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
val_idx   = np.concatenate(val_idx)   if val_idx   else np.array([], dtype=int)
test_idx  = np.concatenate(test_idx)  if test_idx  else np.array([], dtype=int)

rng.shuffle(train_idx)
rng.shuffle(val_idx)
rng.shuffle(test_idx)

# --- Datasets with timm transforms ---
train_base = datasets.ImageFolder(root=str(data_dir), transform=train_tfms)
eval_base  = datasets.ImageFolder(root=str(data_dir), transform=eval_tfms)

# --- Remap labels → {0,1,2} ---
class RemapSubset(Dataset):
    def __init__(self, subset, id_map):
        self.subset = subset
        self.id_map = id_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.id_map[y]

# Train/val/test datasets with remapped labels
train_ds = RemapSubset(Subset(train_base, train_idx.tolist()), id_map)
val_ds   = RemapSubset(Subset(eval_base,  val_idx.tolist()),   id_map)
test_ds  = RemapSubset(Subset(eval_base,  test_idx.tolist()),  id_map)

# --- Oversampling minority classes ---
train_targets = np.array([id_map[base.targets[i]] for i in train_idx], dtype=np.int64)

class_counts = np.bincount(train_targets, minlength=3)
class_weights = 1.0 / np.maximum(class_counts, 1)
sample_weights = class_weights[train_targets]

# --- DataLoaders ---
random_seed = 42
g = torch.Generator().manual_seed(random_seed)

sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
    generator=g
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# --- Sanity checks ---
x, y = train_ds[0]
print("Train sample shape:", x.shape)
print("Train sample label:", y)  # must be 0, 1, or 2

logging.info("Class mapping: %s", base.class_to_idx)
logging.info("Kept class ids: %s", keep_ids)
logging.info("Split sizes: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))
logging.info("Train batches: %d Val batches: %d Test batches: %d",
             len(train_loader), len(val_loader), len(test_loader))
