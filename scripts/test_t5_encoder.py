#!/usr/bin/env python3
"""
Test script for T5 text encoder model to debug encoding process.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import List, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wan.modules.t5 import T5EncoderModel


class DebugT5Encoder:
    """Wrapper around T5EncoderModel that prints tensor shapes and timing information."""
    
    def __init__(self, t5_encoder: T5EncoderModel):
        self.t5_encoder = t5_encoder
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
        """Register hooks on T5 model layers."""
        logger.info("Registering hooks on T5 model layers...")
        
        # Register hooks on encoder blocks
        for name, module in self.t5_encoder.model.blocks.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                module.register_forward_hook(self._hook_fn)
        
        # Register hooks on attention layers
        for name, module in self.t5_encoder.model.named_modules():
            if 'attn' in name and isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                module.register_forward_hook(self._hook_fn)
    
    def __call__(self, texts: List[str], device) -> List[torch.Tensor]:
        """Forward pass with shape tracking."""
        logger.info("=" * 50)
        logger.info("T5 ENCODER FORWARD PASS:")
        logger.info("=" * 50)
        self.layer_shapes = []  # Reset for each forward pass
        
        start_time = time.time()
        result = self.t5_encoder(texts, device)
        end_time = time.time()
        
        logger.info(f"Forward pass completed in {end_time - start_time:.4f} seconds")
        logger.info(f"Output: {len(result)} text embeddings")
        for i, emb in enumerate(result):
            logger.info(f"  Text {i+1}: {emb.shape}")
        
        return result


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_tensor_info(tensor: torch.Tensor, name: str):
    """Print debug information about a tensor."""
    logger.info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                f"range=[{tensor.min():.3f}, {tensor.max():.3f}], "
                f"device={tensor.device}")


def print_model_config(model, model_name="T5 Model"):
    """Print detailed configuration of a model."""
    logger.info(f"\n{model_name} Configuration:")
    logger.info(f"  Model Type: {type(model).__name__}")
    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {dtype}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total Parameters: {total_params:,}")
        logger.info(f"  Trainable Parameters: {trainable_params:,}")
        logger.info(f"  Non-trainable Parameters: {total_params - trainable_params:,}")
        
        # Model structure
        logger.info(f"\n{model_name} Architecture:")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            logger.info(f"  {name}: {type(module).__name__} ({module_params:,} params)")
            
            # Detailed breakdown for major components
            if name == 'blocks':
                logger.info(f"    Number of encoder blocks: {len(module)}")
                for i, block in enumerate(module):
                    block_params = sum(p.numel() for p in block.parameters())
                    logger.info(f"      Block {i}: {type(block).__name__} ({block_params:,} params)")
        
        # Configuration attributes
        if hasattr(model, 'dim'):
            logger.info(f"\n{model_name} Configuration Attributes:")
            logger.info(f"  Base Dimension: {model.dim}")
            logger.info(f"  Attention Dimension: {model.dim_attn}")
            logger.info(f"  FFN Dimension: {model.dim_ffn}")
            logger.info(f"  Number of Heads: {model.num_heads}")
            logger.info(f"  Number of Layers: {model.num_layers}")
            logger.info(f"  Number of Buckets: {model.num_buckets}")
            logger.info(f"  Shared Position: {model.shared_pos}")
            
    except Exception as e:
        logger.warning(f"Could not analyze {model_name}: {e}")


def test_text_encoding(t5_encoder: T5EncoderModel, test_texts: List[str], device):
    """Test text encoding with various prompts."""
    logger.info("=" * 50)
    logger.info("TESTING TEXT ENCODING")
    logger.info("=" * 50)
    
    for i, text in enumerate(test_texts):
        logger.info(f"\n--- Test Text {i+1} ---")
        logger.info(f"Input: '{text}'")
        logger.info(f"Length: {len(text)} characters")
        
        # Time the encoding
        start_time = time.time()
        with torch.no_grad():
            embeddings = t5_encoder([text], device)
        end_time = time.time()
        
        # Analyze results
        logger.info(f"Encoding time: {end_time - start_time:.4f} seconds")
        logger.info(f"Number of embeddings: {len(embeddings)}")
        
        for j, emb in enumerate(embeddings):
            debug_tensor_info(emb, f"Embedding {j+1}")
            
            # Calculate some statistics
            logger.info(f"  Mean: {emb.mean():.6f}")
            logger.info(f"  Std: {emb.std():.6f}")
            logger.info(f"  Min: {emb.min():.6f}")
            logger.info(f"  Max: {emb.max():.6f}")
            
            # Check for NaN or inf values
            if torch.isnan(emb).any():
                logger.warning(f"  WARNING: NaN values detected!")
            if torch.isinf(emb).any():
                logger.warning(f"  WARNING: Inf values detected!")
    
    return embeddings


def test_batch_encoding(t5_encoder: T5EncoderModel, test_texts: List[str], device):
    """Test batch encoding of multiple texts."""
    logger.info("=" * 50)
    logger.info("TESTING BATCH ENCODING")
    logger.info("=" * 50)
    
    logger.info(f"Batch size: {len(test_texts)}")
    logger.info(f"Texts: {[f'"{t[:50]}{"..." if len(t) > 50 else ""}"' for t in test_texts]}")
    
    # Time the batch encoding
    start_time = time.time()
    with torch.no_grad():
        batch_embeddings = t5_encoder(test_texts, device)
    end_time = time.time()
    
    # Analyze batch results
    logger.info(f"Batch encoding time: {end_time - start_time:.4f} seconds")
    logger.info(f"Number of batch embeddings: {len(batch_embeddings)}")
    
    # Check consistency with individual encoding
    logger.info("\nChecking batch vs individual encoding consistency...")
    individual_embeddings = []
    individual_time = 0
    
    for text in test_texts:
        start = time.time()
        with torch.no_grad():
            ind_emb = t5_encoder([text], device)
        individual_time += time.time() - start
        individual_embeddings.append(ind_emb[0])
    
    logger.info(f"Individual encoding total time: {individual_time:.4f} seconds")
    logger.info(f"Batch speedup: {individual_time / (end_time - start_time):.2f}x")
    
    # Compare embeddings
    for i, (batch_emb, ind_emb) in enumerate(zip(batch_embeddings, individual_embeddings)):
        if torch.allclose(batch_emb, ind_emb, atol=1e-6):
            logger.info(f"  Text {i+1}: Batch and individual embeddings match ✓")
        else:
            logger.warning(f"  Text {i+1}: Batch and individual embeddings differ!")
            diff = torch.abs(batch_emb - ind_emb).max()
            logger.warning(f"    Max difference: {diff:.6f}")
    
    return batch_embeddings


def test_memory_usage(t5_encoder: T5EncoderModel, test_texts: List[str], device):
    """Test memory usage during encoding."""
    logger.info("=" * 50)
    logger.info("TESTING MEMORY USAGE")
    logger.info("=" * 50)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        logger.info(f"Initial GPU memory: {initial_memory / 1024**3:.2f} GB")
        
        # Run encoding
        with torch.no_grad():
            embeddings = t5_encoder(test_texts, device)
        
        peak_memory = torch.cuda.max_memory_allocated(device)
        final_memory = torch.cuda.memory_allocated(device)
        
        logger.info(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")
        logger.info(f"Final GPU memory: {final_memory / 1024**3:.2f} GB")
        logger.info(f"Memory increase: {(final_memory - initial_memory) / 1024**3:.2f} GB")
        logger.info(f"Peak increase: {(peak_memory - initial_memory) / 1024**3:.2f} GB")
        
        torch.cuda.empty_cache()
    else:
        logger.info("Memory usage tracking not available for CPU device")


def main():
    """Main test function."""
    # Configuration
    checkpoint_dir = project_root / "Wan2.1-T2V-1.3B"
    t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer = "google/umt5-xxl"
    
    # Test parameters
    text_len = 512
    dtype = torch.bfloat16
    debug_layers = True  # Enable layer-by-layer debugging
    device_id = 0  # GPU device ID
    
    # Test texts with different characteristics
    test_texts = [
        "A beautiful sunset over the ocean with golden clouds",
        "A cat sitting on a windowsill watching birds fly by",
        "A futuristic city with flying cars and neon lights",
        "A peaceful forest with tall trees and sunlight filtering through the leaves",
        "A busy street market with colorful stalls and people shopping"
    ]
    
    # Check if files exist
    checkpoint_path = checkpoint_dir / t5_checkpoint
    if not checkpoint_path.exists():
        logger.error(f"T5 checkpoint not found: {checkpoint_path}")
        logger.info("Please ensure the checkpoint file exists or update the path")
        return
    
    # Set device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # 1. Load T5 text encoder model
        logger.info("=" * 50)
        logger.info("STEP 1: Loading T5 text encoder model")
        logger.info("=" * 50)
        
        t5_encoder = T5EncoderModel(
            text_len=text_len,
            dtype=dtype,
            device=torch.device('cpu'),  # Start on CPU like in WanT2V
            checkpoint_path=str(checkpoint_path),
            tokenizer_path=t5_tokenizer,
            shard_fn=None  # No FSDP for testing
        )
        
        logger.info("T5 text encoder loaded successfully!")
        
        # 2. Print model configuration
        logger.info("=" * 50)
        logger.info("STEP 2: Model configuration")
        logger.info("=" * 50)
        
        print_model_config(t5_encoder.model, "T5 Encoder")
        logger.info(f"\nT5EncoderModel Configuration:")
        logger.info(f"  Text Length: {t5_encoder.text_len}")
        logger.info(f"  Dtype: {t5_encoder.dtype}")
        logger.info(f"  Device: {t5_encoder.device}")
        logger.info(f"  Checkpoint: {t5_encoder.checkpoint_path}")
        logger.info(f"  Tokenizer: {t5_encoder.tokenizer_path}")
        
        # 3. Move model to target device
        logger.info("=" * 50)
        logger.info("STEP 3: Moving model to target device")
        logger.info("=" * 50)
        
        if device.type == 'cuda':
            t5_encoder.model.to(device)
            logger.info(f"Model moved to {device}")
        else:
            logger.info("Model remains on CPU")
        
        # 4. Create debug wrapper if enabled
        if debug_layers:
            debug_t5 = DebugT5Encoder(t5_encoder)
            debug_t5.register_hooks()
            t5_encoder = debug_t5
        
        # 5. Test individual text encoding
        logger.info("=" * 50)
        logger.info("STEP 4: Testing individual text encoding")
        logger.info("=" * 50)
        
        individual_embeddings = test_text_encoding(t5_encoder, test_texts, device)
        
        # 6. Test batch encoding
        logger.info("=" * 50)
        logger.info("STEP 5: Testing batch encoding")
        logger.info("=" * 50)
        
        batch_embeddings = test_batch_encoding(t5_encoder, test_texts, device)
        
        # 7. Test memory usage
        logger.info("=" * 50)
        logger.info("STEP 6: Testing memory usage")
        logger.info("=" * 50)
        
        test_memory_usage(t5_encoder, test_texts, device)
        
        # 8. Test with longer text
        logger.info("=" * 50)
        logger.info("STEP 7: Testing with longer text")
        logger.info("=" * 50)
        
        long_text = "This is a much longer text that tests the model's ability to handle sequences that approach the maximum text length. " * 20
        long_text = long_text[:text_len]  # Truncate to max length
        
        logger.info(f"Long text length: {len(long_text)} characters")
        with torch.no_grad():
            long_embeddings = t5_encoder([long_text], device)
        
        logger.info(f"Long text encoding successful: {len(long_embeddings)} embeddings")
        for i, emb in enumerate(long_embeddings):
            debug_tensor_info(emb, f"Long text embedding {i+1}")
        
        # 9. Summary
        logger.info("=" * 50)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        logger.info("Summary:")
        logger.info(f"  - Tested {len(test_texts)} different text prompts")
        logger.info(f"  - All encoding operations completed successfully")
        logger.info(f"  - Model parameters: {sum(p.numel() for p in debug_t5.t5_encoder.model.parameters()):,}")
        logger.info(f"  - Text length limit: {text_len}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Dtype: {dtype}")
        
        # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
