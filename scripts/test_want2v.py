#!/usr/bin/env python3
"""
Test script for WanT2V text-to-video generation model.
This script instantiates the WanT2V class, generates a video from a text prompt,
and saves the output tensor to a file for reproducibility.
"""

import os
import sys
import torch
import logging
import time
import random
from pathlib import Path
from typing import Optional, Tuple
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wan.text2video import WanT2V
from wan.configs import WAN_CONFIGS


class DebugWanT2V:
    """Wrapper around WanT2V that provides debugging capabilities for DiT layers."""
    
    def __init__(self, want2v: WanT2V):
        self.want2v = want2v
        self.layer_shapes = []
        self.attention_weights = []
        
    def _hook_fn(self, module, input_tensor, output_tensor):
        """Hook function to capture tensor shapes and attention weights."""
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
        
        # Capture attention weights if available
        if hasattr(module, 'attn_weights') and module.attn_weights is not None:
            self.attention_weights.append({
                'layer': layer_name,
                'attention_weights': module.attn_weights.clone().detach()
            })
        
        logger.info(f"  {layer_name}: {input_shape} -> {output_shape}")
    
    def register_hooks(self):
        """Register hooks on DiT model layers for debugging."""
        logger.info("Registering hooks on DiT model layers...")
        
        # Register hooks on transformer blocks
        for name, module in self.want2v.model.blocks.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.MultiheadAttention)):
                module.register_forward_hook(self._hook_fn)
        
        # Register hooks on attention layers specifically
        for name, module in self.want2v.model.named_modules():
            if 'attn' in name and isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                module.register_forward_hook(self._hook_fn)
    
    def generate(self, *args, **kwargs):
        """Generate video with debugging enabled."""
        logger.info("=" * 50)
        logger.info("GENERATING VIDEO WITH DEBUG HOOKS")
        logger.info("=" * 50)
        self.layer_shapes = []  # Reset for each generation
        self.attention_weights = []
        
        start_time = time.time()
        result = self.want2v.generate(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"Video generation completed in {end_time - start_time:.4f} seconds")
        logger.info(f"Output video tensor shape: {result.shape}")
        
        return result


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_environment_variables():
    """Set required environment variables for flash attention."""
    logger.info("Setting environment variables for flash attention...")
    os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'TRUE'
    os.environ['BUILD_TYPE'] = 'rocm'
    logger.info("Environment variables set successfully")


def print_model_config(want2v: WanT2V):
    """Print detailed configuration of the WanT2V model."""
    logger.info("=" * 50)
    logger.info("WANT2V MODEL CONFIGURATION")
    logger.info("=" * 50)
    
    logger.info(f"Device: {want2v.device}")
    logger.info(f"Text Encoder Device: {want2v.text_encoder.device}")
    logger.info(f"VAE Device: {want2v.vae.device}")
    logger.info(f"DiT Model Device: {want2v.model.device}")
    
    logger.info(f"\nModel Parameters:")
    logger.info(f"  Text Length: {want2v.config.text_len}")
    logger.info(f"  T5 Dtype: {want2v.config.t5_dtype}")
    logger.info(f"  Param Dtype: {want2v.config.param_dtype}")
    logger.info(f"  Num Train Timesteps: {want2v.config.num_train_timesteps}")
    
    logger.info(f"\nVAE Configuration:")
    logger.info(f"  VAE Stride: {want2v.vae_stride}")
    logger.info(f"  Patch Size: {want2v.patch_size}")
    
    logger.info(f"\nDiT Model Configuration:")
    total_params = sum(p.numel() for p in want2v.model.parameters())
    trainable_params = sum(p.numel() for p in want2v.model.parameters() if p.requires_grad)
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Print DiT model architecture details
    if hasattr(want2v.model, 'dim'):
        logger.info(f"  Base Dimension: {want2v.model.dim}")
        logger.info(f"  FFN Dimension: {want2v.model.ffn_dim}")
        logger.info(f"  Number of Heads: {want2v.model.num_heads}")
        logger.info(f"  Number of Layers: {want2v.model.num_layers}")
        logger.info(f"  Window Size: {want2v.model.window_size}")
        logger.info(f"  QK Norm: {want2v.model.qk_norm}")
        logger.info(f"  Cross Attention Norm: {want2v.model.cross_attn_norm}")


def save_video_tensor(video_tensor: torch.Tensor, output_path: str):
    """Save the video tensor to a file."""
    logger.info(f"Saving video tensor to: {output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save tensor with metadata
    torch.save({
        'video_tensor': video_tensor,
        'shape': video_tensor.shape,
        'dtype': video_tensor.dtype,
        'device': video_tensor.device,
        'timestamp': time.time()
    }, output_path)
    
    logger.info(f"Video tensor saved successfully: {output_path}")
    logger.info(f"Tensor shape: {video_tensor.shape}")
    logger.info(f"Tensor dtype: {video_tensor.dtype}")
    logger.info(f"Tensor device: {video_tensor.device}")


def load_video_tensor(file_path: str) -> torch.Tensor:
    """Load a video tensor from a file."""
    logger.info(f"Loading video tensor from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video tensor file not found: {file_path}")
    
    data = torch.load(file_path, map_location='cpu')
    video_tensor = data['video_tensor']
    
    logger.info(f"Video tensor loaded successfully: {video_tensor.shape}")
    return video_tensor


def verify_reproducibility(want2v: WanT2V, prompt: str, seed: int, output_dir: Path):
    """Verify that the same seed produces the same output."""
    logger.info("=" * 50)
    logger.info("VERIFYING REPRODUCIBILITY")
    logger.info("=" * 50)
    
    # Generate video with the same seed
    logger.info(f"Generating video with seed: {seed}")
    video1 = want2v.generate(
        input_prompt=prompt,
        seed=seed,
        sampling_steps=20,  # Use fewer steps for faster testing
        offload_model=False
    )
    
    # Save first generation
    output_path1 = output_dir / f"video_seed_{seed}_run1.pt"
    save_video_tensor(video1, str(output_path1))
    
    # Generate video again with the same seed
    logger.info(f"Generating video again with seed: {seed}")
    video2 = want2v.generate(
        input_prompt=prompt,
        seed=seed,
        sampling_steps=20,  # Use fewer steps for faster testing
        offload_model=False
    )
    
    # Save second generation
    output_path2 = output_dir / f"video_seed_{seed}_run2.pt"
    save_video_tensor(video2, str(output_path2))
    
    # Compare outputs
    if torch.allclose(video1, video2, atol=1e-6):
        logger.info("✓ Reproducibility verified: Same seed produces identical output")
    else:
        logger.warning("⚠ Reproducibility issue: Same seed produces different output")
        diff = torch.abs(video1 - video2).max()
        logger.warning(f"  Max difference: {diff:.6f}")
    
    return video1, video2


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test WanT2V text-to-video generation')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='Wan2.1-T2V-1.3B',
                       help='Path to model checkpoint directory')
    parser.add_argument('--model_size', type=str, choices=['t2v-1.3B', 't2v-14B'],
                       default='t2v-1.3B',
                       help='Model size to use')
    parser.add_argument('--device_id', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug_layers', action='store_true',
                       help='Enable DiT layer debugging')
    parser.add_argument('--sampling_steps', type=int, default=50,
                       help='Number of sampling steps')
    parser.add_argument('--frame_num', type=int, default=81,
                       help='Number of frames to generate')
    parser.add_argument('--size', type=str, default='832*480',
                       help='Video resolution (width*height)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for saved tensors')
    
    args = parser.parse_args()
    
    # Configuration
    checkpoint_dir = project_root / args.checkpoint_dir
    model_size = args.model_size
    device_id = args.device_id
    seed = args.seed
    debug_layers = args.debug_layers
    sampling_steps = args.sampling_steps
    frame_num = args.frame_num
    size_str = args.size
    output_dir = project_root / args.output_dir
    
    # Test prompt
    test_prompt = "A cat sitting on a windowsill watching birds fly by"
    
    # Parse size
    if '*' in size_str:
        width, height = map(int, size_str.split('*'))
        size = (width, height)
    else:
        size = (832, 480)  # Default size
    
    # Check if checkpoint directory exists
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        logger.info("Please ensure the checkpoint directory exists or update the path")
        return
    
    # Set environment variables
    set_environment_variables()
    
    # Set device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # 1. Set random seed for reproducibility
        logger.info("=" * 50)
        logger.info("STEP 1: Setting random seed for reproducibility")
        logger.info("=" * 50)
        
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"Random seed set to: {seed}")
        
        # 2. Load model configuration
        logger.info("=" * 50)
        logger.info("STEP 2: Loading model configuration")
        logger.info("=" * 50)
        
        if model_size not in WAN_CONFIGS:
            logger.error(f"Unsupported model size: {model_size}")
            logger.info(f"Supported sizes: {list(WAN_CONFIGS.keys())}")
            return
        
        config = WAN_CONFIGS[model_size]
        logger.info(f"Loaded configuration for: {config.__name__}")
        
        # 3. Instantiate WanT2V model
        logger.info("=" * 50)
        logger.info("STEP 3: Instantiating WanT2V model")
        logger.info("=" * 50)
        
        want2v = WanT2V(
            config=config,
            checkpoint_dir=str(checkpoint_dir),
            device_id=device_id,
            rank=0,
            t5_fsdp=False,  # No FSDP optimization
            dit_fsdp=False,  # No FSDP optimization
            use_usp=False,   # No USP optimization
            t5_cpu=False     # Keep T5 on GPU for faster inference
        )
        
        logger.info("WanT2V model instantiated successfully!")
        
        # 4. Print model configuration
        print_model_config(want2v)
        
        # 5. Create debug wrapper if enabled
        if debug_layers:
            logger.info("=" * 50)
            logger.info("STEP 4: Enabling DiT layer debugging")
            logger.info("=" * 50)
            
            debug_want2v = DebugWanT2V(want2v)
            debug_want2v.register_hooks()
            want2v = debug_want2v
        
        # 6. Generate video
        logger.info("=" * 50)
        logger.info("STEP 5: Generating video from text prompt")
        logger.info("=" * 50)
        
        logger.info(f"Input prompt: '{test_prompt}'")
        logger.info(f"Video size: {size}")
        logger.info(f"Frame number: {frame_num}")
        logger.info(f"Sampling steps: {sampling_steps}")
        logger.info(f"Seed: {seed}")
        
        start_time = time.time()
        video_tensor = want2v.generate(
            input_prompt=test_prompt,
            size=size,
            frame_num=frame_num,
            sampling_steps=sampling_steps,
            seed=seed,
            offload_model=False  # Keep model on GPU for debugging
        )
        end_time = time.time()
        
        logger.info(f"Video generation completed in {end_time - start_time:.4f} seconds")
        logger.info(f"Output video tensor shape: {video_tensor.shape}")
        
        # 7. Save video tensor
        logger.info("=" * 50)
        logger.info("STEP 6: Saving video tensor")
        logger.info("=" * 50)
        
        output_path = output_dir / f"video_tensor_seed_{seed}.pt"
        save_video_tensor(video_tensor, str(output_path))
        
        # 8. Verify reproducibility
        logger.info("=" * 50)
        logger.info("STEP 7: Verifying reproducibility")
        logger.info("=" * 50)
        
        verify_reproducibility(want2v, test_prompt, seed, output_dir)
        
        # 9. Summary
        logger.info("=" * 50)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        logger.info("Summary:")
        logger.info(f"  - Model: {model_size}")
        logger.info(f"  - Checkpoint: {checkpoint_dir}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Seed: {seed}")
        logger.info(f"  - Prompt: '{test_prompt}'")
        logger.info(f"  - Video size: {size}")
        logger.info(f"  - Frame number: {frame_num}")
        logger.info(f"  - Sampling steps: {sampling_steps}")
        logger.info(f"  - Output tensor shape: {video_tensor.shape}")
        logger.info(f"  - Saved to: {output_path}")
        logger.info(f"  - Debug layers: {debug_layers}")
        
        # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
