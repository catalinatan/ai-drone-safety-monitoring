from pathlib import Path

import torch

def find_project_root(start: Path | None = None):
    """Based on config files, derive root path of project folder.

    Args:
        start (Path | None, optional): Starting path to search from.

    Returns:
        Path: Root path of the project
    """
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback: topmost directory we reached
    return start.parents[-1]

def get_device(use_amp: bool = True):
    """Determine the best available device and whether to use AMP.
    
    Args:
        use_amp (bool): Whether to enable automatic mixed precision (only works on CUDA)
    
    Returns:
        tuple: (device, use_amp_enabled)
            device (torch.device): The selected device
            use_amp_enabled (bool): Whether AMP is enabled
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp_enabled = use_amp
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        amp_enabled = False  # MPS doesn't support AMP yet
    else:
        device = torch.device("cpu")
        amp_enabled = False
    
    return device, amp_enabled

def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Load model checkpoint from disk.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file (.pth)
        device (torch.device): Device to map the checkpoint tensors to
        
    Returns:
        Dictionary containing:
            - model_state (dict): Model state dict
            - model_name (str): Name of the architecture
            - num_classes (int): Number of output classes
            - class_to_idx (dict): Class name to index mapping
            - data_cfg (dict): Data configuration used during training
            - val_acc (float): Best validation accuracy (if available)
    """
    return torch.load(checkpoint_path, map_location=device, weights_only=False)