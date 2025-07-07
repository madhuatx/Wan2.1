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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wan.modules.vae import WanVAE


class DebugWanVAE:
    """Wrapper around WanVAE that prints tensor shapes at each layer."""
    
    def __init__(self, vae: WanVAE):
        self.vae = vae
        self.layer_shapes = []
        
    def _hook_fn(self, module, input_tensor, output_tensor):
        """Hook function to capture tensor shapes."""
        if isinstance(input_tensor, tuple):
            input_shape = input_tensor[0].shape
        else:
            input_shape = input_tensor.shape
            
        if isinstance(output_tensor, tuple):
            output_shape = output_tensor[0].shape
        else:
            output_shape = output_tensor.shape
            
        layer_name = module.__class__.__name__
        self.layer_shapes.append({
            'layer': layer_name,
            'input_shape': input_shape,
            'output_shape': output_shape
        })
        logger.info(f"  {layer_name}: {input_shape} -> {output_shape}")
    
    def register_hooks(self):
        """Register hooks on all layers."""
        logger.info("Registering hooks on VAE layers...")
        
        # Register hooks on encoder
        for name, module in self.vae.model.encoder.named_modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d, torch.nn.Linear, 
                                 torch.nn.Upsample, torch.nn.Sequential)):
                module.register_forward_hook(self._hook_fn)
        
        # Register hooks on decoder
        for name, module in self.vae.model.decoder.named_modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d, torch.nn.Linear, 
                                 torch.nn.Upsample, torch.nn.Sequential)):
                module.register_forward_hook(self._hook_fn)
    
    def encode(self, videos):
        """Encode with shape tracking."""
        logger.info("=" * 50)
        logger.info("ENCODER LAYER SHAPES:")
        logger.info("=" * 50)
        self.layer_shapes = []  # Reset for each encode
        result = self.vae.encode(videos)
        logger.info(f"Encoder output shape: {result[0].shape}")
        return result
    
    def decode(self, zs):
        """Decode with shape tracking."""
        logger.info("=" * 50)
        logger.info("DECODER LAYER SHAPES:")
        logger.info("=" * 50)
        self.layer_shapes = []  # Reset for each decode
        result = self.vae.decode(zs)
        logger.info(f"Decoder output shape: {result[0].shape}")
        return result

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


def calculate_quality_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """
    Calculate quality metrics between original and reconstructed videos.
    
    Args:
        original: numpy array of shape (T, H, W, C) in range [0, 255]
        reconstructed: numpy array of shape (T, H, W, C) in range [0, 255]
    
    Returns:
        dict: Dictionary containing PSNR, SSIM, and RMS error metrics
    """
    logger.info("Calculating quality metrics...")
    
    # Ensure both arrays have the same shape (use minimum frame count)
    min_frames = min(original.shape[0], reconstructed.shape[0])
    original_cropped = original[:min_frames]
    reconstructed_cropped = reconstructed[:min_frames]
    
    logger.info(f"Comparing {min_frames} frames: original {original_cropped.shape} vs reconstructed {reconstructed_cropped.shape}")
    
    # Initialize metrics
    psnr_values = []
    ssim_values = []
    rms_errors = []
    
    # Calculate metrics for each frame
    for i in range(min_frames):
        orig_frame = original_cropped[i]
        recon_frame = reconstructed_cropped[i]
        
        # Convert to grayscale for SSIM (or use luminance channel)
        if orig_frame.shape[2] == 3:
            # Convert RGB to grayscale using luminance formula
            orig_gray = 0.299 * orig_frame[:, :, 0] + 0.587 * orig_frame[:, :, 1] + 0.114 * orig_frame[:, :, 2]
            recon_gray = 0.299 * recon_frame[:, :, 0] + 0.587 * recon_frame[:, :, 1] + 0.114 * recon_frame[:, :, 2]
        else:
            orig_gray = orig_frame[:, :, 0]
            recon_gray = recon_frame[:, :, 0]
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(orig_frame, recon_frame, data_range=255)
        psnr_values.append(psnr)
        
        # Calculate SSIM
        ssim = structural_similarity(orig_gray, recon_gray, data_range=255)
        ssim_values.append(ssim)
        
        # Calculate RMS error
        rms = np.sqrt(np.mean((orig_frame.astype(np.float32) - recon_frame.astype(np.float32)) ** 2))
        rms_errors.append(rms)
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_rms = np.mean(rms_errors)
    
    # Calculate per-channel metrics
    channel_metrics = {}
    for ch in range(original_cropped.shape[3]):
        orig_channel = original_cropped[:, :, :, ch]
        recon_channel = reconstructed_cropped[:, :, :, ch]
        
        ch_psnr = np.mean([peak_signal_noise_ratio(orig_channel[i], recon_channel[i], data_range=255) 
                           for i in range(min_frames)])
        ch_ssim = np.mean([structural_similarity(orig_channel[i], recon_channel[i], data_range=255) 
                           for i in range(min_frames)])
        ch_rms = np.sqrt(np.mean((orig_channel.astype(np.float32) - recon_channel.astype(np.float32)) ** 2))
        
        channel_metrics[f'channel_{ch}'] = {
            'psnr': ch_psnr,
            'ssim': ch_ssim,
            'rms': ch_rms
        }
    
    metrics = {
        'overall': {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'rms': avg_rms
        },
        'per_frame': {
            'psnr': psnr_values,
            'ssim': ssim_values,
            'rms': rms_errors
        },
        'per_channel': channel_metrics,
        'frame_count': min_frames
    }
    
    return metrics


def print_quality_metrics(metrics: dict):
    """Print quality metrics in a formatted way."""
    logger.info("=" * 50)
    logger.info("QUALITY METRICS COMPARISON")
    logger.info("=" * 50)
    
    overall = metrics['overall']
    logger.info(f"Overall Metrics (averaged over {metrics['frame_count']} frames):")
    logger.info(f"  PSNR: {overall['psnr']:.2f} dB")
    logger.info(f"  SSIM: {overall['ssim']:.4f}")
    logger.info(f"  RMS Error: {overall['rms']:.2f}")
    
    logger.info("\nPer-Channel Metrics:")
    for ch_name, ch_metrics in metrics['per_channel'].items():
        logger.info(f"  {ch_name.upper()}: PSNR={ch_metrics['psnr']:.2f}dB, "
                   f"SSIM={ch_metrics['ssim']:.4f}, RMS={ch_metrics['rms']:.2f}")
    
    # Print frame-by-frame metrics for first few frames
    logger.info("\nFrame-by-Frame Metrics (first 5 frames):")
    for i in range(min(5, metrics['frame_count'])):
        logger.info(f"  Frame {i+1}: PSNR={metrics['per_frame']['psnr'][i]:.2f}dB, "
                   f"SSIM={metrics['per_frame']['ssim'][i]:.4f}, "
                   f"RMS={metrics['per_frame']['rms'][i]:.2f}")
    
    if metrics['frame_count'] > 5:
        logger.info(f"  ... (showing first 5 of {metrics['frame_count']} frames)")


def main():
    """Main test function."""
    # Configuration
    video_path = "/home/madsrini/develop/diffusion-forcing-transformer/data/real-estate-10k-mini/test_256/2a1769dddc1dbf8d.mp4"
    vae_checkpoint_path = str(project_root / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth")
    output_dir = project_root / "output"
    max_frames = 16  # Limit for testing
    debug_layers = True  # Enable layer-by-layer debugging
    
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
        
        # Create debug wrapper if enabled
        if debug_layers:
            debug_vae = DebugWanVAE(vae)
            debug_vae.register_hooks()
            vae = debug_vae
        
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
        
        # 7. Calculate quality metrics
        logger.info("=" * 50)
        logger.info("STEP 7: Calculating quality metrics")
        quality_metrics = calculate_quality_metrics(frames, reconstructed_frames)
        print_quality_metrics(quality_metrics)
        
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