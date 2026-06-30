"""
Device management utility for GPU/CPU selection across ML models.
Centralizes device detection and configuration for consistent GPU handling.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def get_device():
    """
    Get the appropriate device (cuda or cpu) based on availability and environment configuration.
    
    Returns:
        torch.device: Device object for use with PyTorch models
    """
    # Check environment configuration
    gpu_enabled = os.getenv("GPU_ENABLED", "true").lower() == "true"
    
    if not gpu_enabled:
        logger.info("GPU disabled via GPU_ENABLED=false, using CPU")
        return torch.device("cpu")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    
    # Get CUDA device ID from environment (default: 0)
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    try:
        device_id = int(cuda_visible_devices.split(",")[0])
    except (ValueError, IndexError):
        device_id = 0
    
    device = torch.device(f"cuda:{device_id}")
    cuda_name = torch.cuda.get_device_name(device_id)
    cuda_version = torch.version.cuda
    logger.info(f"Using GPU device: {device} ({cuda_name}), CUDA version: {cuda_version}")
    
    return device


def get_device_string():
    """
    Get device as string for models that use string-based device specification.
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    return "cuda" if torch.cuda.is_available() and os.getenv("GPU_ENABLED", "true").lower() == "true" else "cpu"


def log_device_info():
    """
    Log comprehensive device and GPU information for debugging.
    """
    if torch.cuda.is_available():
        logger.info("CUDA is available")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        logger.info("CUDA is not available, using CPU")
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"GPU_ENABLED: {os.getenv('GPU_ENABLED', 'true')}")
