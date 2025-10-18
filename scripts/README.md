# Scripts Directory

This directory contains all executable scripts organized by purpose.

## Structure

- `training/` - Main training scripts
- `testing/` - Test and evaluation scripts  
- `evaluation/` - Analysis and visualization tools
- `run_*.sh` - Shell scripts for easy execution

## Quick Start

```bash
# Basic DDPO training
accelerate launch scripts/training/train.py

# Actor-Critic training
./scripts/run_ac_example.sh

# Cat unlearning pipeline
./scripts/run_cat_unlearning.sh

# Test unlearning progression
./scripts/run_unlearn_test.sh
```

See the main README.md for detailed usage instructions.
