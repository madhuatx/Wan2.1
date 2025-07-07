#!/usr/bin/env python3
"""
Test script for WanVAE model to debug encoding/decoding process.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wan.modules.vae import WanVAE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load video frames using OpenCV.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to load (None for all frames)
    
    Returns:
        frames: numpy array of shape (T, H, W, C) in range [0, 255]
        fps: frames per second
    """
    logger.info(f"Loading video from: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        frame_count = min(frame_count, max_frames)
    
    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        if i % 10 == 0:
            logger.info(f"Loaded frame {i+1}/{frame_count}")
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames were loaded from the video")
    
    frames = np.array(frames)  # Shape: (T, H, W, C)
    logger.info(f"Loaded {len(frames)} frames with shape: {frames.shape}, FPS: {fps}")
    
    return frames, fps


def preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    """
    Preprocess frames for VAE input.
    
    Args:
        frames: numpy array of shape (T, H, W, C) in range [0, 255]
    
    Returns:
        torch.Tensor of shape (C, T, H, W) in range [-1, 1]
    """
    logger.info("Preprocessing frames...")
    
    # Convert to float and normalize to [-1, 1]
    frames_float = frames.astype(np.float32) / 255.0
    frames_normalized = frames_float * 2.0 - 1.0
    
    # Convert to torch tensor and rearrange dimensions: (T, H, W, C) -> (C, T, H, W)
    frames_tensor = torch.from_numpy(frames_normalized).permute(3, 0, 1, 2)
    
    logger.info(f"Preprocessed tensor shape: {frames_tensor.shape}, range: [{frames_tensor.min():.3f}, {frames_tensor.max():.3f}]")
    
    return frames_tensor


def postprocess_frames(frames_tensor: torch.Tensor) -> np.ndarray:
    """
    Postprocess frames from VAE output.
    
    Args:
        frames_tensor: torch.Tensor of shape (C, T, H, W) in range [-1, 1]
    
    Returns:
        numpy array of shape (T, H, W, C) in range [0, 255]
    """
    logger.info("Postprocessing frames...")
    
    # Convert from (C, T, H, W) to (T, H, W, C)
    frames = frames_tensor.permute(1, 2, 3, 0).cpu().numpy()
    
    # Denormalize from [-1, 1] to [0, 255]
    frames_denorm = (frames + 1.0) / 2.0
    frames_uint8 = np.clip(frames_denorm * 255.0, 0, 255).astype(np.uint8)
    
    logger.info(f"Postprocessed array shape: {frames_uint8.shape}, range: [{frames_uint8.min()}, {frames_uint8.max()}]")
    
    return frames_uint8


def save_video(frames: np.ndarray, output_path: str, fps: int = 16):
    """
    Save frames as MP4 video.
    
    Args:
        frames: numpy array of shape (T, H, W, C) in range [0, 255]
        output_path: Path to save the video
        fps: frames per second
    """
    logger.info(f"Saving video to: {output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get video dimensions
    height, width = frames.shape[1:3]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for: {output_path}")
    
    # Write frames
    for i, frame in enumerate(frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        if i % 10 == 0:
            logger.info(f"Saved frame {i+1}/{len(frames)}")
    
    out.release()
    logger.info(f"Video saved successfully: {output_path}")


def debug_tensor_info(tensor: torch.Tensor, name: str):
    """Print debug information about a tensor."""
    logger.info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                f"range=[{tensor.min():.3f}, {tensor.max():.3f}], "
                f"device={tensor.device}")


def main():
    """Main test function."""
    # Configuration
    video_path = "/home/madsrini/develop/diffusion-forcing-transformer/data/real-estate-10k-mini/test_256/2a1769dddc1dbf8d.mp4"
    vae_checkpoint_path = str(project_root / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth")
    output_dir = project_root / "output"
    max_frames = 16  # Limit for testing
    
    # Check if files exist
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    
    if not os.path.exists(vae_checkpoint_path):
        logger.error(f"VAE checkpoint not found: {vae_checkpoint_path}")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # 1. Load video frames
        logger.info("=" * 50)
        logger.info("STEP 1: Loading video frames")
        frames, fps = load_video_frames(video_path, max_frames=max_frames)
        
        # 2. Preprocess frames
        logger.info("=" * 50)
        logger.info("STEP 2: Preprocessing frames")
        frames_tensor = preprocess_frames(frames)
        # Move to device and ensure correct dtype
        frames_tensor = frames_tensor.to(device=device, dtype=torch.float16)
        debug_tensor_info(frames_tensor, "Input frames")
        
        # 3. Load VAE model
        logger.info("=" * 50)
        logger.info("STEP 3: Loading VAE model")
        vae = WanVAE(
            z_dim=16,
            vae_pth=vae_checkpoint_path,
            dtype=torch.float16,  # Changed to float16 for all operations
            device=str(device)
        )
        logger.info("VAE model loaded successfully")
        
        # 4. Encode video
        logger.info("=" * 50)
        logger.info("STEP 4: Encoding video")
        with torch.no_grad():
            # Convert to list format expected by VAE
            video_list = [frames_tensor]
            latents = vae.encode(video_list)
            logger.info(f"Encoded {len(latents)} video(s)")
            debug_tensor_info(latents[0], "Latent representation")
        
        # 5. Decode video
        logger.info("=" * 50)
        logger.info("STEP 5: Decoding video")
        with torch.no_grad():
            reconstructed_videos = vae.decode(latents)
            logger.info(f"Decoded {len(reconstructed_videos)} video(s)")
            debug_tensor_info(reconstructed_videos[0], "Reconstructed video")
        
        # 6. Postprocess and save
        logger.info("=" * 50)
        logger.info("STEP 6: Saving reconstructed video")
        reconstructed_frames = postprocess_frames(reconstructed_videos[0])
        
        # Save reconstructed video
        output_path = output_dir / "reconstructed_video.mp4"
        save_video(reconstructed_frames, str(output_path), fps=fps)
        
        # Save original video for comparison
        original_output_path = output_dir / "original_video.mp4"
        save_video(frames, str(original_output_path), fps=fps)
        
        logger.info("=" * 50)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info(f"Original video saved to: {original_output_path}")
        logger.info(f"Reconstructed video saved to: {output_path}")
        
        # Print summary statistics
        logger.info("=" * 50)
        logger.info("SUMMARY STATISTICS:")
        logger.info(f"Input video shape: {frames.shape}")
        logger.info(f"Latent shape: {latents[0].shape}")
        logger.info(f"Reconstructed shape: {reconstructed_frames.shape}")
        
        # Calculate compression ratio
        input_size = frames.nbytes
        latent_size = latents[0].nbytes
        compression_ratio = input_size / latent_size
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 