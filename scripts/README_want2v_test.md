# WanT2V Test Script

This directory contains a comprehensive test script for the WanT2V text-to-video generation model.

## Files

- `test_want2v.py` - Main test script for WanT2V
- `run_want2v_test.sh` - Shell script to run the test with proper environment setup
- `README_want2v_test.md` - This documentation file

## Features

The test script provides:

1. **Model Instantiation**: Loads WanT2V model from checkpoint directory
2. **Text-to-Video Generation**: Generates video from text prompt
3. **Reproducibility**: Uses manual seed for consistent results
4. **Tensor Output**: Saves generated video tensor to file
5. **Debugging**: Optional DiT layer debugging with hooks
6. **Comprehensive Logging**: Detailed logging of all operations
7. **Environment Setup**: Automatic setup of flash attention environment variables

## Requirements

- Python virtual environment with required packages
- CUDA-capable GPU (recommended)
- WanT2V model checkpoints in the specified directory

## Quick Start

### Option 1: Using the shell script (Recommended)

```bash
# Make sure you're in the scripts directory
cd scripts

# Run the test with default parameters
./run_want2v_test.sh
```

### Option 2: Manual execution

```bash
# Activate virtual environment
source venv/bin/activate

# Set environment variables
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export BUILD_TYPE="rocm"

# Run the test script
python3 test_want2v.py
```

## Command Line Arguments

The test script supports the following command line arguments:

- `--checkpoint_dir`: Path to model checkpoint directory (default: "Wan2.1-T2V-1.3B")
- `--model_size`: Model size to use (choices: "t2v-1.3B", "t2v-14B", default: "t2v-1.3B")
- `--device_id`: GPU device ID (default: 0)
- `--seed`: Random seed for reproducibility (default: 42)
- `--debug_layers`: Enable DiT layer debugging (flag)
- `--sampling_steps`: Number of sampling steps (default: 50)
- `--frame_num`: Number of frames to generate (default: 81)
- `--size`: Video resolution as "width*height" (default: "832*480")
- `--output_dir`: Output directory for saved tensors (default: "output")

## Example Usage

### Basic test with default parameters:
```bash
python3 test_want2v.py
```

### Test with custom parameters:
```bash
python3 test_want2v.py \
    --checkpoint_dir "Wan2.1-T2V-14B" \
    --model_size "t2v-14B" \
    --seed 123 \
    --sampling_steps 30 \
    --frame_num 65 \
    --size "1280*720" \
    --debug_layers
```

### Test with debugging enabled:
```bash
python3 test_want2v.py --debug_layers
```

## Output

The script generates:

1. **Console Output**: Detailed logging of all operations
2. **Video Tensor Files**: Saved as `.pt` files in the output directory:
   - `video_tensor_seed_{seed}.pt` - Main generated video tensor
   - `video_seed_{seed}_run1.pt` - First reproducibility test
   - `video_seed_{seed}_run2.pt` - Second reproducibility test

## Video Tensor Format

The saved video tensors have the format:
- **Shape**: `(C, N, H, W)` where:
  - `C`: Color channels (3 for RGB)
  - `N`: Number of frames
  - `H`: Frame height
  - `W`: Frame width
- **Data Type**: `torch.float32`
- **Range**: Values in range [-1, 1] (normalized)

## Loading Saved Tensors

To load and use the saved video tensors:

```python
import torch

# Load the video tensor
data = torch.load('output/video_tensor_seed_42.pt')
video_tensor = data['video_tensor']

# Access metadata
shape = data['shape']
dtype = data['dtype']
device = data['device']
timestamp = data['timestamp']

print(f"Video tensor shape: {shape}")
```

## Troubleshooting

### Common Issues:

1. **Checkpoint Directory Not Found**:
   - Ensure the checkpoint directory exists and contains required model files
   - Check the path in `--checkpoint_dir` argument

2. **CUDA Out of Memory**:
   - Reduce `--frame_num` or `--sampling_steps`
   - Use smaller video resolution with `--size`
   - Ensure no other processes are using GPU memory

3. **Environment Variables Not Set**:
   - The script automatically sets required environment variables
   - Ensure you're using the provided shell script or manually set:
     ```bash
     export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
     export BUILD_TYPE="rocm"
     ```

4. **Virtual Environment Issues**:
   - Ensure the virtual environment is activated
   - Check that all required packages are installed

## Model Configurations

The script supports two main model configurations:

- **t2v-1.3B**: Smaller model, faster inference, lower memory usage
- **t2v-14B**: Larger model, higher quality, higher memory usage

## Performance Notes

- **Memory Usage**: The 14B model requires significantly more GPU memory
- **Inference Time**: More sampling steps produce higher quality but take longer
- **Frame Count**: Higher frame counts increase memory usage and generation time
- **Resolution**: Higher resolutions require more memory and computation

## Debugging Features

When `--debug_layers` is enabled:

- Forward hooks are registered on DiT model layers
- Tensor shapes are logged for each layer
- Attention weights are captured if available
- Detailed timing information is provided

This is useful for:
- Understanding model architecture
- Debugging tensor shape mismatches
- Analyzing attention patterns
- Performance profiling

