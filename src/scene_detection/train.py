# training.py
import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import timm

from src.logger import setup_logger
from src.scene_detection.preprocess import DataConfig, SceneDatasetBuilder
from src.utils import get_device, load_checkpoint

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Model settings
    model_name: str = "mobilenetv3_large_100"
    num_classes: int = 4
    
    # Training hyperparameters
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    
    # Training options
    save_path: str = "best_scene_model.pth"
    no_amp: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_args(cls, args):
        """Create config from argparse arguments"""
        return cls(
            model_name=args.model_name,
            num_classes=args.num_classes,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_path=args.save_path,
            no_amp=args.no_amp,
            log_level=args.log_level,
        )

class SceneTrainer:
    """Handles training loop for scene detection model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: TrainingConfig,
        device: torch.device,
        use_amp: bool,
        class_to_idx: dict,
        data_cfg: dict,
        logger,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.use_amp = use_amp
        self.class_to_idx = class_to_idx
        self.data_cfg = data_cfg
        self.logger = logger
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
        
        self.best_val_acc = -1.0
    
    def train_one_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        
        return running_loss / max(total, 1), correct / max(total, 1)
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate model on given dataloader"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            logits = self.model(x)
            loss = self.criterion(logits, y)
            
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        
        return running_loss / max(total, 1), correct / max(total, 1)
    
    def save_checkpoint(self, val_acc: float):
        """Save model checkpoint with training metadata.
        
        Args:
            val_acc: Validation accuracy for this checkpoint
        """
        checkpoint = {
            "model_state": self.model.state_dict(),
            "model_name": self.config.model_name,
            "num_classes": self.config.num_classes,
            "class_to_idx": self.class_to_idx,
            "data_cfg": self.data_cfg,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, self.config.save_path)
        self.logger.info(f"Saved checkpoint to {self.config.save_path} (val_acc={val_acc:.3f})")
    
    def load_best_checkpoint(self):
        """Load the best saved checkpoint into the model"""
        self.logger.info(f"Loading best checkpoint from {self.config.save_path}")
        ckpt = load_checkpoint(self.config.save_path, self.device)
        self.model.load_state_dict(ckpt["model_state"])
        
        if "val_acc" in ckpt:
            self.logger.info(f"Loaded checkpoint with val_acc={ckpt['val_acc']:.3f}")
    
    def train(self):
        """Run full training loop"""
        self.logger.info("=" * 60)
        self.logger.info("Starting training")
        self.logger.info("=" * 60)
        
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader)
            self.scheduler.step()
            
            elapsed = time.time() - t0
            
            self.logger.info(
                f"Epoch {epoch:02d}/{self.config.epochs:02d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
                f"{elapsed:.1f}s"
            )
            
            # Save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(val_acc)
        
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.3f}")
        return self.best_val_acc
    
    def test(self):
        """Evaluate on test set using best checkpoint"""
        self.logger.info("=" * 60)
        self.logger.info("Testing on best checkpoint")
        
        # Load best checkpoint
        self.load_best_checkpoint()
        
        # Evaluate
        test_loss, test_acc = self.evaluate(self.test_loader)
        self.logger.info(f"TEST | loss {test_loss:.4f} acc {test_acc:.3f}")
        self.logger.info("=" * 60)
        
        return test_loss, test_acc

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train scene detection model")
    
    # Data arguments (--data-dir and --data-dirs are mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--data-dir", type=str, default=None,
                       help="ImageFolder root (contains class folders)")
    data_group.add_argument("--data-dirs", type=str, nargs="+", default=None,
                       help="YOLO-style dataset directories (e.g. data/bridge_dataset data/ship_dataset)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="mobilenetv3_large_100")
    parser.add_argument("--num-classes", type=int, default=4)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Other arguments
    parser.add_argument("--save-path", type=str, default="best_scene_model.pth")
    parser.add_argument("--no-amp", action="store_true",
                       help="Disable mixed precision even on CUDA")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(
        __name__,
        level=getattr(__import__('logging'), args.log_level.upper()),
        log_prefix="train",
        file_output=True,
    )
    
    # Create configurations
    data_config = DataConfig(
        data_dir=args.data_dir,
        data_dirs=args.data_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        model_name=args.model_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    
    train_config = TrainingConfig.from_args(args)
    
    # Get device
    device, use_amp = get_device(use_amp=not args.no_amp)
    logger.info(f"Device: {device} | AMP: {use_amp}")
    logger.info(f"Model: {args.model_name} | Epochs: {args.epochs} | "
               f"Batch size: {args.batch_size} | LR: {args.lr} | WD: {args.weight_decay}")
    
    # Build dataloaders
    logger.info("Building dataloaders...")
    builder = SceneDatasetBuilder(data_config)
    train_loader, val_loader, test_loader, class_to_idx, data_cfg = builder.build_dataloaders()
    
    logger.info(f"Class mapping: {class_to_idx}")
    logger.info(f"Data config: {data_cfg}")
    logger.info(f"Batches | train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")
    
    # Create model
    logger.info(f"Creating model: {args.model_name}")
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=args.num_classes,
    )
    
    # Create trainer and train
    trainer = SceneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_config,
        device=device,
        use_amp=use_amp,
        class_to_idx=class_to_idx,
        data_cfg=data_cfg,
        logger=logger,
    )
    
    # Train and test
    best_val_acc = trainer.train()
    test_loss, test_acc = trainer.test()
    
    logger.info(f"Training complete | Best val acc: {best_val_acc:.3f} | Test acc: {test_acc:.3f}")


if __name__ == "__main__":
    main()
