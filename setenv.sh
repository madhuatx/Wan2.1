# MADHU: These are rocm-specific post-install steps to get pytorch
# working on rocm

# If you are on an hpcfund cluster with MI2xx, then
if [[ "$(uname -n)" == *"hpcfund" ]]; then
    # uninstall the default "CUDA" based packages
    pip3 uninstall --no-input torch torchvision pytorch-triton-rocm
    # install packages from pytorch index
    # see https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#using-a-wheels-package
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
    # we really don't care about migraphx (it doesn't work)
else
    # install special index packages from rocm
    # see https://rocm.docs.amd.com/projects/radeon/en/latest/index.html

    # Download only if packages do not exist in project root
    if ! [ -f "torch-2.4.0+rocm6.3.4.git7cecbf6d-cp312-cp312-linux_x86_64.whl" ]; then
        wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.4/torch-2.4.0%2Brocm6.3.4.git7cecbf6d-cp312-cp312-linux_x86_64.whl
    fi
    if ! [ -f "torchvision-0.19.0+rocm6.3.4.gitfab84886-cp312-cp312-linux_x86_64.whl" ]; then
        wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.4/torchvision-0.19.0%2Brocm6.3.4.gitfab84886-cp312-cp312-linux_x86_64.whl
    fi
    if ! [ -f "torchaudio-2.4.0+rocm6.3.4.git69d40773-cp312-cp312-linux_x86_64.whl" ]; then
        wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.4/torchaudio-2.4.0%2Brocm6.3.4.git69d40773-cp312-cp312-linux_x86_64.whl
    fi
    if ! [ -f "pytorch_triton_rocm-3.0.0+rocm6.3.4.git75cc27c2-cp312-cp312-linux_x86_64.whl" ]; then
        wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.4/pytorch_triton_rocm-3.0.0%2Brocm6.3.4.git75cc27c2-cp312-cp312-linux_x86_64.whl
    fi

    # uninstall the default "CUDA" based packages
    pip3 uninstall --yes torch torchvision pytorch-triton-rocm
    # install the rocm based packages
    pip3 install --no-input pytorch_triton_rocm-3.0.0+rocm6.3.4.git75cc27c2-cp312-cp312-linux_x86_64.whl
    pip3 install --no-input torch-2.4.0+rocm6.3.4.git7cecbf6d-cp312-cp312-linux_x86_64.whl
    pip3 install --no-input torchvision-0.19.0+rocm6.3.4.gitfab84886-cp312-cp312-linux_x86_64.whl
    pip3 install --no-input torchaudio-2.4.0+rocm6.3.4.git69d40773-cp312-cp312-linux_x86_64.whl

    # install migraphx
    # taken from https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-migraphx.html

    # Setup pythonpath for migraphx module
    export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH

    #setup instructions for torch_migraphx module
    git clone https://github.com/ROCmSoftwarePlatform/torch_migraphx.git ./venv/torch_migraphx
    cd ./venv/torch_migraphx/py
    export TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
    pip install --no-input .
    cd ../../../
    # install onnx runtime
    # taken from https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-onnx.html
    pip3 install --no-input onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.4/
fi
