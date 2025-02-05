#!/bin/bash

# Detect and configure GPU architecture
echo "Detecting GPU architecture..."
python3 detect_gpu.py

# Clean previous build
rm ./src/release/cuda_ed25519_vanity;
rm ./src/release/ecc_scan.o;
#export PATH=/usr/local/cuda/bin:$PATH;
make -j$(nproc);
