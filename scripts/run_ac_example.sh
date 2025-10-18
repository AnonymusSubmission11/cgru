#!/bin/bash

# Example script to run DDPO training with Actor-Critic method
# This script demonstrates how to use the AC implementation

echo "Running DDPO training with Actor-Critic method..."

# Activate virtual environment
source .venv/bin/activate

# Check if config file exists
if [ ! -f "config/ac_example.py" ]; then
    echo "Error: config/ac_example.py not found!"
    echo "Please make sure the AC example config is available."
    exit 1
fi

# Run training with AC method
accelerate launch scripts/training/train.py --config config/ac_example.py

echo "Training completed!"
echo "Check the logs directory for results."
