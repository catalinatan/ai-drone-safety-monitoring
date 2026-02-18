import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets

from src.utils import find_project_root
from src.logger import setup_logger

# Initialize logger with custom prefix and file output
logger = setup_logger(__name__, log_prefix="preprocess", file_output=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class DataConfig:
    """Configuration for Torch Dataset and DataLoaders"""
    data_dir: str | None = None
    data_dirs: List[str] | None = None
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    model_name: str = "mobilenetv3_large_100"
    input_size: Tuple[int, int, int] | None = None

class SceneDataset(Dataset):
    """Dataset that loads images from multiple YOLO-style dataset directories.

    Each directory is treated as a separate class. The class name is inferred
    by stripping the ``_dataset`` suffix from the directory name.

    Expected layout per directory::

        <name>_dataset/
            images/
                train/
                    img1.jpg
                val/
                    img2.jpg
    """

    def __init__(self, data_dirs: List[str], transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.targets: List[int] = []

        # Derive sorted class names for deterministic label assignment
        class_names = []
        for d in data_dirs:
            name = Path(d).name
            class_name = name.replace("_dataset", "") if name.endswith("_dataset") else name
            class_names.append((class_name, d))
        class_names.sort(key=lambda x: x[0])

        self.class_to_idx = {name: idx for idx, (name, _) in enumerate(class_names)}
        self.classes = list(self.class_to_idx.keys())

        for class_name, dir_path in class_names:
            label = self.class_to_idx[class_name]
            images_dir = Path(dir_path) / "images"
            if not images_dir.exists():
                logger.warning(f"No images/ directory in {dir_path}, skipping")
                continue
            for split in ("train", "val"):
                split_dir = images_dir / split
                if not split_dir.exists():
                    continue
                for img_path in sorted(split_dir.iterdir()):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS and img_path.is_file():
                        self.samples.append((str(img_path), label))
                        self.targets.append(label)

        logger.info(
            f"SceneDataset: {len(self.samples)} images across "
            f"{len(self.classes)} classes: {self.class_to_idx}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TransformFactory:
    """Factory to create model-specific transforms using TIMM"""

    @staticmethod
    def create_transforms(model_name: str, input_size=None):
        """Setup data transforms for a given model using TIMM.

        Args:
            model_name (str): Pretrained model name to load.
            input_size (tuple, optional): Override input size.

        Returns:
            tuple: (cfg, train_transforms, eval_transforms)
        """
        logger.info(f"Creating transforms for model: {model_name}")
        logger.debug(f"Input size override: {input_size}")
        
        tmp_model = timm.create_model(model_name, pretrained=True)

        if input_size is None:
            cfg = timm.data.resolve_data_config(tmp_model.pretrained_cfg)
        else:
            # If input_size override is used, you should also pass mean/std/interp; default behavior is cfg from pretrained_cfg.
            cfg = timm.data.resolve_data_config(tmp_model.pretrained_cfg, {"input_size": input_size})

        train_transforms = timm.data.create_transform(**cfg, is_training=True)
        eval_transforms = timm.data.create_transform(**cfg, is_training=False)
        
        logger.info(f"Transforms created successfully. Config: {cfg}")
        return cfg, train_transforms, eval_transforms

class StratifiedSplitter:
    """Handles stratified splitting of dataset indices."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        logger.debug(f"StratifiedSplitter initialized with seed: {seed}")
    
    def split(
        self, 
        targets: np.ndarray, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1,
        ):
        """Perform stratified train/val/test split.
        
        Args:
            targets (np.ndarray): Array of class labels for the samples.
            train_ratio (float): Proportion for training set.
            val_ratio (float): Proportion for validation set.
            
        Returns:
            tuple: (train_indices, val_indices, test_indices)
        """
        logger.info(f"Starting stratified split with ratios - train: {train_ratio}, val: {val_ratio}, test: {1-train_ratio-val_ratio:.2f}")
        
        all_indices = np.arange(len(targets), dtype=int)
        num_classes = int(targets.max()) + 1
        
        logger.info(f"Total samples: {len(targets)}, Number of classes: {num_classes}")
        
        train_idx, val_idx, test_idx = [], [], []
        
        for class_label in range(num_classes):
            class_idx = all_indices[targets == class_label].copy()
            self.rng.shuffle(class_idx)
            
            n = len(class_idx)
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))
            
            logger.debug(f"Class {class_label}: {n} samples -> train: {n_train}, val: {n_val}, test: {n - n_train - n_val}")
            
            train_idx.append(class_idx[:n_train])
            val_idx.append(class_idx[n_train:n_train + n_val])
            test_idx.append(class_idx[n_train + n_val:])
        
        train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
        val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
        test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)
        
        self.rng.shuffle(train_idx)
        self.rng.shuffle(val_idx)
        self.rng.shuffle(test_idx)
        
        logger.info(f"Split completed - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        return train_idx, val_idx, test_idx

class WeightedSamplerFactory:
    """Factory for creating weighted samplers for imbalanced datasets."""
    
    @staticmethod
    def create_sampler(
        targets: np.ndarray, 
        num_classes: int, 
        seed: int = 42,
        ):
        """Create a weighted random sampler to handle class imbalance.
        
        Args:
            targets (np.ndarray): Array of class labels for the samples.
            num_classes (int): Total number of classes.
            seed (int): Random seed for reproducibility.
            
        Returns:
            WeightedRandomSampler: Sampler for DataLoader.
        """
        logger.info("Creating weighted sampler for class imbalance")
        
        class_counts = np.bincount(targets, minlength=num_classes)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[targets]
        
        logger.info(f"Class distribution: {dict(enumerate(class_counts))}")
        logger.debug(f"Class weights: {dict(enumerate(class_weights))}")
        
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        
        logger.info("Weighted sampler created successfully")
        return sampler

class SceneDatasetBuilder:
    """Builds dataloaders for scene detection datasets."""
    
    def __init__(self, config: DataConfig):
        logger.info("Initializing SceneDatasetBuilder")
        logger.debug(f"Config: {config}")
        
        self.config = config
        self.splitter = StratifiedSplitter(seed=config.seed)
        self.transform_factory = TransformFactory()
    
    def _create_imagefolder_datasets(
        self,
        data_dir: str,
        train_transforms,
        eval_transforms,
        ):
        """Create base datasets from an ImageFolder directory.

        Args:
            data_dir (str): Root directory with class subfolders.
            train_transforms: TIMM transforms for training data.
            eval_transforms: TIMM transforms for evaluation data.

        Returns:
            tuple: (train_dataset, eval_dataset)
        """
        logger.debug(f"Creating ImageFolder datasets from: {data_dir}")

        train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
        eval_dataset = datasets.ImageFolder(root=data_dir, transform=eval_transforms)

        logger.info(f"Datasets created - Found {len(train_dataset.classes)} classes: {train_dataset.classes}")
        return train_dataset, eval_dataset

    def _create_scene_datasets(
        self,
        data_dirs: List[str],
        train_transforms,
        eval_transforms,
        ):
        """Create base datasets from multiple YOLO-style dataset directories.

        Args:
            data_dirs: List of paths to YOLO-style dataset directories.
            train_transforms: TIMM transforms for training data.
            eval_transforms: TIMM transforms for evaluation data.

        Returns:
            tuple: (train_dataset, eval_dataset)
        """
        logger.debug(f"Creating SceneDatasets from: {data_dirs}")

        train_dataset = SceneDataset(data_dirs, transform=train_transforms)
        eval_dataset = SceneDataset(data_dirs, transform=eval_transforms)

        return train_dataset, eval_dataset

    def build_dataloaders(self):
        """Build train, validation, and test dataloaders.

        Returns:
            tuple: (train_loader, val_loader, test_loader, class_to_idx, config_dict)
        """
        logger.info("=" * 60)
        logger.info("Building dataloaders")
        logger.info("=" * 60)

        # Get transforms
        cfg, train_transforms, eval_transforms = self.transform_factory.create_transforms(
            model_name=self.config.model_name,
            input_size=self.config.input_size,
        )

        # Build base datasets depending on config
        if self.config.data_dirs is not None:
            # YOLO-style multi-directory mode
            logger.info(f"Using YOLO-style data directories: {self.config.data_dirs}")
            for d in self.config.data_dirs:
                if not os.path.exists(d):
                    logger.error(f"Data directory does not exist: {d}")
                    raise FileNotFoundError(f"Data directory not found: {d}")

            base_dataset = SceneDataset(self.config.data_dirs)
            train_base_dataset, eval_base_dataset = self._create_scene_datasets(
                self.config.data_dirs, train_transforms, eval_transforms,
            )
        else:
            # ImageFolder mode (original behaviour)
            root_dir = find_project_root()
            if self.config.data_dir is None:
                data_dir = os.path.join(root_dir, "data/scene_detection")
                logger.info(f"Using default data directory: {data_dir}")
            else:
                data_dir = self.config.data_dir
                logger.info(f"Using custom data directory: {data_dir}")

            if not os.path.exists(data_dir):
                logger.error(f"Data directory does not exist: {data_dir}")
                raise FileNotFoundError(f"Data directory not found: {data_dir}")

            logger.info("Loading base dataset to extract class information")
            base_dataset = datasets.ImageFolder(root=data_dir)
            train_base_dataset, eval_base_dataset = self._create_imagefolder_datasets(
                data_dir, train_transforms, eval_transforms,
            )

        img_class_array = np.array(base_dataset.targets, dtype=np.int64)
        num_classes = len(base_dataset.classes)

        # Perform stratified split
        train_idx, val_idx, test_idx = self.splitter.split(
            targets=img_class_array,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
        )

        # Create subsets
        logger.info("Creating dataset subsets")
        train_dataset = Subset(train_base_dataset, train_idx.tolist())
        val_dataset = Subset(eval_base_dataset, val_idx.tolist())
        test_dataset = Subset(eval_base_dataset, test_idx.tolist())

        # Create weighted sampler for training
        train_targets = img_class_array[train_idx]
        sampler = WeightedSamplerFactory.create_sampler(
            targets=train_targets,
            num_classes=num_classes,
            seed=self.config.seed,
        )

        # Create dataloaders
        logger.info("Creating DataLoaders")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        logger.info(f"DataLoaders created successfully:")
        logger.info(f"  - Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
        logger.info(f"  - Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
        logger.info(f"  - Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
        logger.info(f"  - Batch size: {self.config.batch_size}")
        logger.info(f"  - Class mapping: {base_dataset.class_to_idx}")
        logger.info("=" * 60)

        return train_loader, val_loader, test_loader, base_dataset.class_to_idx, cfg


if __name__ == "__main__":
    logger.info("Running preprocess.py as main script")
    
    config = DataConfig(
        batch_size=64,
        model_name="mobilenetv3_large_100",
        seed=123,
    )
    try:
        builder = SceneDatasetBuilder(config)
        train_loader, val_loader, test_loader, class_to_idx, cfg = builder.build_dataloaders()
        
        # Test loading a batch
        logger.info("Testing batch loading...")
        x, y = next(iter(train_loader))
        logger.info(f"Batch shape: {x.shape}, Labels range: [{y.min().item()}, {y.max().item()}]")
        logger.info(f"Class to index mapping: {class_to_idx}")
        logger.info(f"Transform config: {cfg}")
        logger.info("✓ Preprocessing pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        raise
