#!/bin/bash
set -e

# Install system dependencies
apt-get update
apt-get install -y build-essential cmake

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
