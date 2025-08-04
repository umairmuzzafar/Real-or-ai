#!/bin/bash
set -e

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake

echo "Creating virtual environment..."
python -m venv /home/adminuser/venv
source /home/adminuser/venv/bin/activate

echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Making Streamlit executable..."
chmod +x /home/adminuser/venv/bin/streamlit

echo "Installation complete!"
