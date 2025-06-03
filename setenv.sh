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
    if ! [ -f "torch-2.6.0+rocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl" ]; then
        wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl
    fi
    if ! [ -f "torchvision-0.21.0+rocm6.4.1.git4040d51f-cp312-cp312-linux_x86_64.whl" ]; then
        wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torchvision-0.21.0%2Brocm6.4.1.git4040d51f-cp312-cp312-linux_x86_64.whl
    fi
    if ! [ -f "torchaudio-2.6.0+rocm6.4.1.gitd8831425-cp312-cp312-linux_x86_64.whl" ]; then
        wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torchaudio-2.6.0%2Brocm6.4.1.gitd8831425-cp312-cp312-linux_x86_64.whl 
    fi
    if ! [ -f "pytorch_triton_rocm-3.2.0+rocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl" ]; then
        wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl
    fi

    # uninstall the default "CUDA" based packages
    pip3 uninstall --yes torch torchvision pytorch-triton-rocm
    # install the rocm based packages
    pip3 install --no-input pytorch_triton_rocm-3.2.0+rocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl
    pip3 install --no-input torch-2.6.0+rocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl
    pip3 install --no-input torchvision-0.21.0+rocm6.4.1.git4040d51f-cp312-cp312-linux_x86_64.whl
    pip3 install --no-input torchaudio-2.6.0+rocm6.4.1.gitd8831425-cp312-cp312-linux_x86_64.whl
    pip3 install --no-input onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/

    # install flash attention for rocm
    pip install triton==3.3.0
    cd venv/
    git clone --recursive git@github.com:ROCm/flash-attention.git
    cd flash-attention
    git checkout main_perf
    export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" 
    export BUILD_TYPE="rocm"
    python setup.py install
    cd ..

fi

   