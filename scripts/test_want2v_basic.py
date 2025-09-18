#!/usr/bin/env python3
"""
Basic test script for WanT2V model instantiation.
This script only loads the model without running full generation,
useful for testing model loading and configuration.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wan.text2video import WanT2V
from wan.configs import WAN_CONFIGS


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_environment_variables():
    """Set required environment variables for flash attention."""
    logger.info("Setting environment variables for flash attention...")
    os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'TRUE'
    os.environ['BUILD_TYPE'] = 'rocm'
    logger.info("Environment variables set successfully")


def test_model_instantiation(checkpoint_dir: str, model_size: str, device_id: int = 0):
    """Test basic model instantiation without generation."""
    logger.info("=" * 50)
    logger.info("TESTING WANT2V MODEL INSTANTIATION")
    logger.info("=" * 50)
    
    try:
        # Check if checkpoint directory exists
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint directory not found: {checkpoint_path}")
            return False
        
        # Check if model size is supported
        if model_size not in WAN_CONFIGS:
            logger.error(f"Unsupported model size: {model_size}")
            logger.info(f"Supported sizes: {list(WAN_CONFIGS.keys())}")
            return False
        
        # Load configuration
        config = WAN_CONFIGS[model_size]
        logger.info(f"Loaded configuration for: {config.__name__}")
        
        # Set device
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Instantiate model
        logger.info("Instantiating WanT2V model...")
        want2v = WanT2V(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False
        )
        
        logger.info("✓ WanT2V model instantiated successfully!")
        
        # Print basic model info
        logger.info(f"Model device: {want2v.device}")
        logger.info(f"Text encoder device: {want2v.text_encoder.device}")
        logger.info(f"VAE device: {want2v.vae.device}")
        logger.info(f"DiT model device: {want2v.model.device}")
        
        # Check model parameters
        total_params = sum(p.numel() for p in want2v.model.parameters())
        logger.info(f"Total DiT model parameters: {total_params:,}")
        
        # Check if models are on correct devices
        if device.type == 'cuda':
            if want2v.model.device.type == 'cuda':
                logger.info("✓ DiT model is on GPU")
            else:
                logger.warning("⚠ DiT model is not on GPU")
            
            if want2v.vae.device.type == 'cuda':
                logger.info("✓ VAE model is on GPU")
            else:
                logger.warning("⚠ VAE model is not on GPU")
        
        # Test basic forward pass on a small input (without full generation)
        logger.info("Testing basic model components...")
        
        # Test text encoder with a simple prompt
        test_prompt = "test"
        try:
            with torch.no_grad():
                text_embeddings = want2v.text_encoder([test_prompt], want2v.text_encoder.device)
            logger.info(f"✓ Text encoder working, output shape: {text_embeddings[0].shape}")
        except Exception as e:
            logger.error(f"✗ Text encoder failed: {e}")
            return False
        
        # Test VAE with a small dummy input
        try:
            dummy_latent = torch.randn(1, 16, 1, 60, 104, device=want2v.vae.device, dtype=torch.float16)
            with torch.no_grad():
                dummy_output = want2v.vae.decode([dummy_latent])
            logger.info(f"✓ VAE working, output shape: {dummy_output[0].shape}")
        except Exception as e:
            logger.error(f"✗ VAE failed: {e}")
            return False
        
        # Test DiT model with a small dummy input
        try:
            dummy_input = torch.randn(1, 16, 1, 60, 104, device=want2v.model.device, dtype=torch.float16)
            dummy_timestep = torch.tensor([500], device=want2v.model.device)
            with torch.no_grad():
                dummy_output = want2v.model(dummy_input, t=dummy_timestep, context=text_embeddings, seq_len=1)
            logger.info(f"✓ DiT model working, output shape: {dummy_output[0].shape}")
        except Exception as e:
            logger.error(f"✗ DiT model failed: {e}")
            return False
        
        logger.info("=" * 50)
        logger.info("ALL TESTS PASSED! Model is ready for generation.")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Basic test for WanT2V model instantiation')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='Wan2.1-T2V-1.3B',
                       help='Path to model checkpoint directory')
    parser.add_argument('--model_size', type=str, choices=['t2v-1.3B', 't2v-14B'],
                       default='t2v-1.3B',
                       help='Model size to use')
    parser.add_argument('--device_id', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set environment variables
    set_environment_variables()
    
    # Run test
    success = test_model_instantiation(
        checkpoint_dir=args.checkpoint_dir,
        model_size=args.model_size,
        device_id=args.device_id
    )
    
    if success:
        logger.info("Basic test completed successfully!")
        sys.exit(0)
    else:
        logger.error("Basic test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()







