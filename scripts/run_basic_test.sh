#!/bin/bash

# Script to run basic WanT2V test (model instantiation only)

# Exit on any error
set -e

echo "=== WanT2V Basic Test Runner ==="
echo "Testing model instantiation only..."
echo ""

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "Error: Virtual environment not found at ../venv"
    echo "Please create the virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../venv/bin/activate

# Set environment variables for flash attention
echo "Setting environment variables for flash attention..."
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export BUILD_TYPE="rocm"

echo "Environment setup complete!"
echo ""

# Check if CUDA is available
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

echo ""
echo "Running basic WanT2V test..."

# Run the basic test script
python3 test_want2v_basic.py \
    --checkpoint_dir "Wan2.1-T2V-1.3B" \
    --model_size "t2v-1.3B" \
    --device_id 0

echo ""
echo "Basic test completed!"
echo "If successful, the model is ready for full generation testing."

