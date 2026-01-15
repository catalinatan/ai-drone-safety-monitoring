# data.py
import os
from pathlib import Path

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets


class RemapSubset(Dataset):
    def __init__(self, subset, id_map):
        self.subset = subset
        self.id_map = id_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.id_map[int(y)]


def make_transforms(model_name: str = "mobilenetv3_large_100", input_size=None):
    """
    Create timm official train/eval transforms based on pretrained cfg for model_name.
    """
    # Create a temporary model only to read its pretrained_cfg (no training here)
    tmp_model = timm.create_model(model_name, pretrained=True)
    cfg = timm.data.resolve_data_config(tmp_model.pretrained_cfg) if input_size is None else {"input_size": input_size}

    # If input_size override is used, you should also pass mean/std/interp; default behavior is cfg from pretrained_cfg.
    if input_size is not None:
        cfg = timm.data.resolve_data_config(tmp_model.pretrained_cfg, {"input_size": input_size})

    train_tfms = timm.data.create_transform(**cfg, is_training=True)
    eval_tfms = timm.data.create_transform(**cfg, is_training=False)
    return cfg, train_tfms, eval_tfms


def build_loaders(
    data_dir: str | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    model_name: str = "mobilenetv3_large_100",
):
    """
    Builds stratified train/val/test DataLoaders from a single ImageFolder root.
    Keeps all classes found in the data directory.
    Returns: train_loader, val_loader, test_loader, class_to_idx, cfg
    """
    if data_dir is None:
        grandparent_dir = Path(__file__).resolve().parents[2]
        data_dir = os.path.join(grandparent_dir, "data")

    # Transforms (official timm)
    cfg, train_tfms, eval_tfms = make_transforms(model_name=model_name)

    # Base dataset (no transforms) just to read targets and class_to_idx
    base = datasets.ImageFolder(root=str(data_dir))
    
    all_indices = np.arange(len(base.targets), dtype=int)
    all_labels = np.array(base.targets, dtype=np.int64)
    num_classes = len(base.classes)

    # Stratified split
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in range(num_classes):
        cls_idx = all_indices[all_labels == c].copy()
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

    # Datasets with transforms
    train_base = datasets.ImageFolder(root=str(data_dir), transform=train_tfms)
    eval_base  = datasets.ImageFolder(root=str(data_dir), transform=eval_tfms)

    train_ds = Subset(train_base, train_idx.tolist())
    val_ds   = Subset(eval_base,  val_idx.tolist())
    test_ds  = Subset(eval_base,  test_idx.tolist())

    # Oversampling on train only
    train_targets = np.array([base.targets[i] for i in train_idx], dtype=np.int64)
    class_counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_targets]

    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, base.class_to_idx, cfg


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_to_idx, cfg = build_loaders()
    x, y = next(iter(train_loader))
    print("Batch:", x.shape, y.min().item(), y.max().item())
    print("class_to_idx:", class_to_idx)
    print("cfg:", cfg)
