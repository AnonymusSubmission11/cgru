# Testing Scripts

This directory contains scripts for testing and evaluating the RL Machine Unlearning framework.

## Scripts

### `test_unlearning_progression.py`
Tests the unlearning progression by generating images from different checkpoints.

**Usage:**
```bash
python3 test_unlearning_progression.py \
    --log_dir "logs/your_training_run" \
    --output_dir "results/unlearning_test" \
    --device "cuda" \
    --seed 42
```

### `test_cat_likeness_reward.py`
Tests the cat-likeness reward function with sample images.

**Usage:**
```bash
python3 test_cat_likeness_reward.py
```

### `prepare_cat_embeddings.py`
Downloads COCO cat images and computes CLIP embeddings for the cat-likeness reward function.

**Usage:**
```bash
python3 prepare_cat_embeddings.py
```

### `critic_model.py`
Standalone critic model implementation for testing and training.

## Dependencies

- PyTorch
- Diffusers
- CLIP
- PIL
- Matplotlib
- COCO API (for cat embeddings)

## Output

Test scripts generate:
- Individual test images
- Comparison visualizations
- Performance metrics
- Debug information
