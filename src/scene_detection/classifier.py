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

id_map = {orig: new for new, orig in enumerate(keep_ids)}

filtered_train = RemapSubset(Subset(train_base, kept_indices), id_map)
filtered_eval  = RemapSubset(Subset(eval_base,  kept_indices), id_map)

# --- Oversampling minority classes ---
filtered_targets = np.array([id_map[base.targets[i]] for i in kept_indices], dtype=np.int64)

class_counts = np.bincount(filtered_targets)    
class_weights = 1.0 / class_counts
sample_weights = class_weights[filtered_targets]

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
    filtered_train,
    batch_size=32,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)

eval_loader = DataLoader(
    filtered_eval,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# --- Sanity checks ---
x, y = filtered_train[0]
print("Sample shape:", x.shape)
print("Sample label:", y)  # must be 0, 1, or 2

logging.info("Class mapping: %s", base.class_to_idx)
logging.info("Kept class ids: %s", keep_ids)
logging.info("Train batches: %d Eval batches: %d",
             len(train_loader), len(eval_loader))
