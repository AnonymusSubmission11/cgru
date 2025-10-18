#!/bin/bash

# Script to run AC training with trained aesthetic critic

echo "Starting AC training with aesthetic critic..."
echo "=============================================="

# Activate virtual environment
source .venv/bin/activate

echo ""
echo "Starting AC training..."
echo "======================"

# Run AC training
accelerate launch scripts/training/train.py --config config/ac_aesthethic.py

echo "AC training completed!"
