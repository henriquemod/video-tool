#!/bin/bash

# Check for NVIDIA GPU
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected"
    pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.5.1 torchvision==0.20.1
    pip install -r requirements.txt
    exit 0
fi

# Check for AMD GPU
if lspci | grep -i amd > /dev/null; then
    echo "AMD GPU detected"
    pip install --index-url https://download.pytorch.org/whl/rocm5.6 torch==2.5.1 torchvision==0.20.1
    pip install -r requirements.txt
    exit 0
fi

# No supported GPU found
echo "No NVIDIA or AMD GPU detected. Installing CPU-only version."
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt

echo "Installation completed."