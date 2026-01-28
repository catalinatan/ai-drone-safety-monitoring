import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.utils import find_project_root

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    log_prefix: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    format_str: Optional[str] = None,
    ):
    """Set up a logger with console and timestamped file output.
    
    Args:
        name (str): Logger name (typically __name__ from calling module).
        level (int): Logging level (default: logging.INFO).
        log_dir (str, optional): Directory to save log file (default: logs/ in project root).
        log_prefix (str, optional): Prefix for log filename (default: "log_").
        console_output (bool): Whether to log to console.
        file_output (bool): Whether to log to timestamped file.
        format_str (str, optional): Log message format.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_str)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with timestamp
    if file_output:
        # Set default log directory if not provided
        if log_dir is None:
            project_root = find_project_root()
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir)
        
        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with YYYYMMDD prefix
        date_prefix = datetime.now().strftime("%Y%m%d")
        if log_prefix:
            log_filename = f"{date_prefix}_{log_prefix}.log"
        else:
            log_filename = f"{date_prefix}.log"
        
        log_path = log_dir / log_filename
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(
    name: str, 
    level: int = logging.INFO,
    log_prefix: Optional[str] = None,
    file_output: bool = False,
    ):
    """Get an existing logger or create a basic one.
    
    Args:
        name (str): Logger name.
        level (int): Logging level.
        log_prefix (str, optional): Prefix for log filename.
        file_output (bool): Whether to log to file.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    
    # If no handlers, set up logger
    if not logger.handlers:
        logger = setup_logger(name, level=level, log_prefix=log_prefix, file_output=file_output)
    
    return logger