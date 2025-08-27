# Scripts Directory

This directory contains various test and utility scripts for the Wan2.1 project.

## Available Scripts

### WanT2V Text-to-Video Generation Tests

- **`test_want2v.py`** - Comprehensive test script for WanT2V model
  - Instantiates WanT2V class from checkpoint
  - Generates video from text prompt
  - Saves output video tensor to file
  - Includes reproducibility testing
  - Optional DiT layer debugging
  - Command line argument support

- **`test_want2v_basic.py`** - Basic model instantiation test
  - Tests model loading without full generation
  - Useful for debugging model setup issues
  - Faster execution for development

- **`run_want2v_test.sh`** - Shell script for full WanT2V test
  - Sets up environment variables
  - Activates virtual environment
  - Runs comprehensive test with default parameters

- **`run_basic_test.sh`** - Shell script for basic test
  - Quick model instantiation test
  - Useful for development and debugging

### VAE and T5 Encoder Tests

- **`test_vae.py`** - Test script for WanVAE model
  - Tests video encoding/decoding
  - Includes quality metrics calculation
  - Debug layer information

- **`test_t5_encoder.py`** - Test script for T5 text encoder
  - Tests text encoding functionality
  - Includes batch processing tests
  - Memory usage analysis

## Quick Start

### For WanT2V Testing:

1. **Basic Model Test** (recommended first):
   ```bash
   cd scripts
   ./run_basic_test.sh
   ```

2. **Full Generation Test**:
   ```bash
   cd scripts
   ./run_want2v_test.sh
   ```

3. **Custom Parameters**:
   ```bash
   cd scripts
   source ../venv/bin/activate
   export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
   export BUILD_TYPE="rocm"
   python3 test_want2v.py --model_size t2v-14B --seed 123 --debug_layers
   ```

### For Other Components:

```bash
cd scripts
source ../venv/bin/activate
python3 test_vae.py
python3 test_t5_encoder.py
```

## Environment Setup

All scripts require:
- Python virtual environment (`venv/`)
- Required packages installed
- Proper environment variables set for flash attention

The shell scripts automatically handle environment setup.

## Documentation

- **`README_want2v_test.md`** - Detailed documentation for WanT2V testing
- **`README.md`** - This overview file

## Notes

- WanT2V tests require model checkpoints in the specified directory
- VAE and T5 tests require their respective model files
- All scripts include comprehensive logging and error handling
- Debug options are available for development and troubleshooting

