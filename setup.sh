#!/bin/bash
# Install system dependencies
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Install PyTorch first with CUDA support if available
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
