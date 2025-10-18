#!/bin/bash

echo "Running DDPO training with Actor-Critic method for cat unlearning..."
echo "This will train the model to reduce cat-likeness in generated images."

# Run the main training
accelerate launch scripts/training/train.py --config config/ac_cat_unlearning.py

echo "Cat unlearning training completed!"
