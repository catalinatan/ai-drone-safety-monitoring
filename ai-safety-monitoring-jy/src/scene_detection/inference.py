import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import timm
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import get_device, load_checkpoint, find_project_root
from logger import setup_logger


# Default configuration
DEFAULT_CLASS_NAMES = ["bridge", "others", "railway", "ship"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class SceneDetector:
    """Scene classification inference engine.
    
    This class encapsulates a trained scene classification model and provides
    methods for single image and batch inference.
    
    Example:
        detector = SceneDetector("best_scene_model.pth")
        result = detector.predict("image.jpg", topk=3)
        print(result)
        [('bridge', 0.8234), ('railway', 0.1234), ('others', 0.0432)]
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the scene detector.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            logger: Optional logger instance (creates one if not provided)
        """
        self.logger = logger or setup_logger(
            __name__,
            log_prefix="scene_detector",
            file_output=False,
            console_output=False
        )
        
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.device = self._setup_device(device)
        self.model, self.checkpoint = self._load_model()
        self.class_names = self._get_class_names()
        self.transforms = self._get_transforms()
        
        self.logger.info(f"SceneDetector initialized with {len(self.class_names)} classes")
    
    def _resolve_checkpoint_path(self, checkpoint_path: str):
        """Resolve checkpoint path to absolute path."""
        path = Path(checkpoint_path)
        if not path.is_absolute():
            project_root = find_project_root()
            path = project_root / checkpoint_path
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        return path
    
    def _setup_device(self, device: str):
        """Setup computation device."""
        if device == "auto":
            dev, _ = get_device(use_amp=False)
            self.logger.info(f"Auto-selected device: {dev}")
            return dev
        else:
            dev = torch.device(device)
            self.logger.info(f"Using specified device: {dev}")
            return dev
    
    def _load_model(self):
        """Load model from checkpoint."""
        self.logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = load_checkpoint(str(self.checkpoint_path), self.device)
        
        model_name = checkpoint.get("model_name", "mobilenetv3_large_100")
        num_classes = checkpoint.get("num_classes", 4)
        
        self.logger.info(f"Model architecture: {model_name}")
        self.logger.info(f"Number of classes: {num_classes}")
        
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model.to(self.device)
        
        if "val_acc" in checkpoint:
            self.logger.info(f"Model validation accuracy: {checkpoint['val_acc']:.4f}")
        
        return model, checkpoint
    
    def _get_class_names(self):
        """Extract class names from checkpoint or use defaults."""
        if "class_to_idx" in self.checkpoint:
            idx_to_class = {v: k for k, v in self.checkpoint["class_to_idx"].items()}
            num_classes = self.checkpoint.get("num_classes", len(idx_to_class))
            class_names = [idx_to_class.get(i, f"class_{i}") for i in range(num_classes)]
        else:
            class_names = DEFAULT_CLASS_NAMES
            self.logger.warning("No class_to_idx in checkpoint, using defaults")
        
        self.logger.info(f"Classes: {class_names}")
        return class_names
    
    def _get_transforms(self):
        """Get image transforms from checkpoint or create from pretrained config."""
        if "data_cfg" in self.checkpoint:
            self.logger.debug("Using transforms from checkpoint data_cfg")
            return timm.data.create_transform(**self.checkpoint["data_cfg"], is_training=False)
        else:
            # Fallback: use pretrained config
            model_name = self.checkpoint.get("model_name", "mobilenetv3_large_100")
            self.logger.debug(f"Creating transforms from pretrained config: {model_name}")
            tmp_model = timm.create_model(model_name, pretrained=True)
            cfg = timm.data.resolve_data_config(tmp_model.pretrained_cfg)
            return timm.data.create_transform(**cfg, is_training=False)
    
    @torch.no_grad()
    def predict(
        self,
        img_path: Union[str, Path],
        topk: int = 5,
    ):
        """Run inference on a single image.
        
        Args:
            img_path: Path to image file
            topk: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples, sorted by probability
            
        Example:
            detector.predict("bridge.jpg", topk=3)
            [('bridge', 0.95), ('railway', 0.03), ('others', 0.02)]
        """
        img_path = Path(img_path)
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        tensor = self.transforms(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Get top-k predictions
        topk_probs, topk_indices = probs.topk(min(topk, len(self.class_names)), dim=1)
        topk_probs = topk_probs[0].cpu().tolist()
        topk_indices = topk_indices[0].cpu().tolist()
        
        results = []
        for prob, idx in zip(topk_probs, topk_indices):
            label = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            results.append((label, prob))
        
        self.logger.debug(f"Predicted {img_path.name}: {results[0][0]} ({results[0][1]:.4f})")
        
        return results
    
    def predict_batch(
        self,
        img_paths: List[Union[str, Path]],
        topk: int = 3,
        show_progress: bool = True
    ):
        """Run inference on multiple images.
        
        Args:
            img_paths: List of image file paths
            topk: Number of top predictions to return
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping filenames to prediction lists
            
        Example:
            results = detector.predict_batch(["img1.jpg", "img2.jpg"])
            results["img1.jpg"]
            [('bridge', 0.95), ('railway', 0.03)]
        """
        results = {}
        errors = []
        
        iterator = tqdm(img_paths, desc="Inference") if show_progress else img_paths
        
        for img_path in iterator:
            img_path = Path(img_path)
            try:
                preds = self.predict(img_path, topk=topk)
                results[img_path.name] = preds
            except Exception as e:
                error_msg = f"Error processing {img_path.name}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        self.logger.info(f"Processed {len(results)}/{len(img_paths)} images, {len(errors)} errors")
        
        return results
    
    def process_directory(
        self,
        data_dir: Union[str, Path],
        topk: int = 3,
        output_file: Optional[str] = None,
        verbose: bool = True,
        recursive: bool = False
    ):
        """Process all images in a directory.
        
        Args:
            data_dir: Directory containing images
            topk: Number of top predictions to return
            output_file: Optional path to save results as JSON
            verbose: Whether to print per-image results
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary mapping filenames to prediction lists
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            self.logger.error(f"Directory not found: {data_dir}")
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        # Get all image files
        if recursive:
            image_files = [
                f for ext in IMAGE_EXTENSIONS
                for f in data_path.rglob(f"*{ext}")
            ]
        else:
            image_files = [
                f for f in data_path.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]
        
        if not image_files:
            self.logger.warning(f"No images found in {data_dir}")
            print(f"No images found in {data_dir}")
            return {}
        
        self.logger.info(f"Processing {len(image_files)} images from {data_dir}")
        print(f"Processing {len(image_files)} images...")
        
        all_results = {}
        errors = []
        
        for img_file in tqdm(image_files, desc="Inference"):
            try:
                preds = self.predict(img_file, topk=topk)
                all_results[img_file.name] = preds
                
                if verbose:
                    print(f"\n{img_file.name}:")
                    for label, prob in preds:
                        print(f"  {label}: {prob:.4f}")
                        
            except Exception as e:
                error_msg = f"Error processing {img_file.name}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                if verbose:
                    print(f"\n{error_msg}")
        
        # Summary
        summary = f"Processed: {len(all_results)}/{len(image_files)} images"
        if errors:
            summary += f", Errors: {len(errors)}"
        
        self.logger.info(summary)
        print(f"\n{'='*50}")
        print(summary)
        
        # Save results to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            self.logger.info(f"Results saved to: {output_file}")
            print(f"Results saved to: {output_file}")
        
        return all_results
    
    def __call__(
        self,
        img_path: Union[str, Path],
        topk: int = 3
    ):
        """Make detector callable for convenience.
        
        Example:
            >>> detector = SceneDetector("model.pth")
            >>> result = detector("image.jpg")  # Same as detector.predict()
        """
        return self.predict(img_path, topk=topk)
    
    def get_class_distribution(
        self,
        results: Dict[str, List[Tuple[str, float]]]
    ):
        """Get distribution of predicted classes (top prediction only).
        
        Args:
            results: Results from predict_batch or process_directory
            
        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {name: 0 for name in self.class_names}
        
        for preds in results.values():
            if preds:
                top_class = preds[0][0]
                if top_class in distribution:
                    distribution[top_class] += 1
        
        return distribution


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run scene classification inference on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory
  python -m src.scene_detection.inference
  
  # Single image
  python -m src.scene_detection.inference --image test.jpg --topk 5
  
  # Custom checkpoint and output
  python -m src.scene_detection.inference --checkpoint models/scene.pth --output results.json
  
  # Quiet mode with debug logging
  python -m src.scene_detection.inference --quiet --log-level DEBUG
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_scene_model.pth",
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing images (default: data/scene_detection_test)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image file to process"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top predictions to show (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image output (only show summary)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for images"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for inference (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(
        name=__name__,
        level=log_level,
        log_prefix="scene_inference",
        file_output=not args.no_log_file
    )
    
    logger.info("="*50)
    logger.info("Starting Scene Detection Inference")
    logger.info("="*50)
    
    try:
        # Initialize detector
        detector = SceneDetector(
            checkpoint_path=args.checkpoint,
            device=args.device,
            logger=logger
        )
        
        print(f"Device: {detector.device}")
        print(f"Classes: {detector.class_names}")
        
        # Process single image or directory
        if args.image:
            # Single image inference
            logger.info(f"Processing single image: {args.image}")
            print(f"\nProcessing: {args.image}")
            
            results = detector.predict(args.image, topk=args.topk)
            
            print(f"\nPredictions:")
            for label, prob in results:
                print(f"  {label}: {prob:.4f}")
                logger.info(f"  {label}: {prob:.4f}")
            
            # Save to JSON if requested
            if args.output:
                output_data = {Path(args.image).name: results}
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to: {args.output}")
                logger.info(f"Results saved to: {args.output}")
                
        else:
            # Directory inference
            if args.data_dir:
                data_dir = args.data_dir
            else:
                project_root = find_project_root()
                data_dir = str(project_root / "data" / "scene_detection_test")
            
            logger.info(f"Processing directory: {data_dir}")
            results = detector.process_directory(
                data_dir,
                topk=args.topk,
                output_file=args.output,
                verbose=not args.quiet,
                recursive=args.recursive
            )
            
            # Print class distribution
            if results:
                distribution = detector.get_class_distribution(results)
                print(f"\nClass Distribution:")
                for class_name, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {class_name}: {count}")
                    logger.info(f"  {class_name}: {count}")
        
        logger.info("Inference completed successfully")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()