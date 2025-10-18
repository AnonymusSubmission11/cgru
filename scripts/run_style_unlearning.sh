#!/bin/bash

echo "Running DDPO training with Actor-Critic method for style unlearning..."
echo "This will train the model to reduce Van Gogh-likeness in generated images."

# Run the main training
accelerate launch scripts/training/train.py --config config/ac_style.py

echo "Style unlearning training completed!"
