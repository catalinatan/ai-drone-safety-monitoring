# data.py
import os
from pathlib import Path

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets


# class RemapSubset(Dataset):
#     def __init__(self, subset, id_map):
#         self.subset = subset
#         self.id_map = id_map

#     def __len__(self):
#         return len(self.subset)

#     def __getitem__(self, idx):
#         x, y = self.subset[idx]
#         return x, self.id_map[int(y)]


def get_timm_transforms(model_name: str,
                    input_size=None,
                    ):
    """Transform images into format compatible for finetuning specified model.

    Args:
        model_name (str): Pretrained model to load.
        input_size (tuple, optional): Input size for the model.

    Returns:
        tuple: Configuration dictionary, training transforms, and evaluation transforms.
    """
    # Retrieve required transformation to match model's expected input (NO TRAINING)
    tmp_model = timm.create_model(model_name, pretrained=True)
    cfg = timm.data.resolve_data_config(tmp_model.pretrained_cfg) if input_size is None else {"input_size": input_size}

    # If input_size override is used, you should also pass mean/std/interp; default behavior is cfg from pretrained_cfg.
    if input_size is not None:
        cfg = timm.data.resolve_data_config(tmp_model.pretrained_cfg, {"input_size": input_size})

    train_transforms = timm.data.create_transform(**cfg, is_training=True)
    eval_transforms = timm.data.create_transform(**cfg, is_training=False)
    return cfg, train_transforms, eval_transforms


def get_dataloaders(
        data_dir: str | None = None,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        model_name: str = "mobilenetv3_large_100",
        ):
    """Create stratified train-val-test DataLoaders from a specified root directory.
    Assumes data is organized in subdirectories per class (ImageFolder format).
    Args:
        data_dir (str): Root directory of the dataset.
        batch_size (int, optional): Batch size for DataLoaders.
        num_workers (int, optional): Number of worker processes for data loading.
        pin_memory (bool, optional): Whether to use pinned memory for DataLoaders.
        seed (int, optional): Random seed for reproducibility.
        train_ratio (float, optional): Proportion of data for training set.
        val_ratio (float, optional): Proportion of data for validation set.
        model_name (str, optional): Pretrained model name for input transforms.

    Returns:
        tuple: Train, validation, and test DataLoaders, class_to_idx mapping, and config dictionary.
    """
    if data_dir is None:
        grandparent_dir = Path(__file__).resolve().parents[2]
        data_dir = os.path.join(grandparent_dir, "data/scene_detection")

    # Create TIMM specified transforms
    cfg, train_transforms, eval_transforms = get_timm_transforms(model_name=model_name)

    # Create dataset object that assigns labels based on subdirectory names
    base_dataset = datasets.ImageFolder(root=data_dir)
    
    # Get data indices and labels
    all_indices = np.arange(len(base_dataset.targets), dtype=int)
    all_labels = np.array(base_dataset.targets, dtype=np.int64)
    num_classes = len(base_dataset.classes)

    # Stratified split
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for class_label in range(num_classes):
        class_idx = all_indices[all_labels == class_label].copy()
        rng.shuffle(class_idx)

        n = len(class_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_idx.append(class_idx[:n_train])
        val_idx.append(class_idx[n_train:n_train + n_val])
        test_idx.append(class_idx[n_train + n_val:])

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    val_idx   = np.concatenate(val_idx)   if val_idx   else np.array([], dtype=int)
    test_idx  = np.concatenate(test_idx)  if test_idx  else np.array([], dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Execute TIMM transforms on Dataset and create Subsets by indices
    train_base_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    eval_base_dataset  = datasets.ImageFolder(root=data_dir, transform=eval_transforms)

    train_dataset = Subset(train_base_dataset, train_idx.tolist())
    val_dataset   = Subset(eval_base_dataset,  val_idx.tolist())
    test_dataset  = Subset(eval_base_dataset,  test_idx.tolist())

    # Oversample minority classes in training set
    train_targets = np.array([base_dataset.targets[i] for i in train_idx], dtype=np.int64)
    class_counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_targets]

    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, base_dataset.class_to_idx, cfg


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    train_loader, val_loader, test_loader, class_to_idx, cfg = get_dataloaders()
    x, y = next(iter(train_loader))
    print("Batch:", x.shape, y.min().item(), y.max().item())
    print("class_to_idx:", class_to_idx)
    print("cfg:", cfg)
