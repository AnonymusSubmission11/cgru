# RL Machine Unlearning: Actor-Critic DDPO Framework

This repository implements an enhanced version of [Denoising Diffusion Policy Optimization (DDPO)](https://rl-diffusion.github.io/) with Actor-Critic (AC) methods for machine unlearning applications. The framework supports fine-tuning Stable Diffusion models to "unlearn" specific concepts (e.g., cats) using reinforcement learning.

## 🚀 Key Features

- **Actor-Critic DDPO**: Enhanced DDPO with per-timestep critic predictions for better variance reduction
- **Machine Unlearning**: Specialized for unlearning specific concepts from diffusion models
- **LoRA Support**: Memory-efficient fine-tuning with low-rank adaptation
- **Multiple Reward Functions**: Aesthetic scoring, cat-likeness detection, and custom rewards
- **Comprehensive Evaluation**: Tools for testing unlearning progression and model performance

## 📁 Project Structure

```
rl-machine-unlearning/
├── scripts/
│   ├── training/            # Main training scripts
│   │   ├── train.py         # Enhanced DDPO with AC support
│   │   └── train_original.py
│   ├── testing/             # Test and evaluation scripts
│   │   ├── test_unlearning_progression.py
│   │   ├── generate_images.py
│   │   ├── generate_images_for_metrics.py
│   │   ├── precompute_van_gogh_features.py
│   │   └── prepare_cat_embeddings.py
│   ├── run_ac_example.sh    # AC training with example config
│   ├── run_ac_aesthetic.sh  # AC training with aesthetic critic (if available)
│   ├── run_cat_unlearning.sh
│   ├── run_style_unlearning.sh
│   └── run_unlearn_test.sh
├── config/                  # Configuration files
│   ├── base.py              # Base configuration
│   ├── ac_example.py        # AC example config
│   ├── ac_aesthethic.py     # AC with aesthetic critic
│   ├── ac_cat_unlearning.py # AC for cat unlearning
│   ├── ac_books_unlearning.py
│   ├── ac_style.py
│   ├── ac_nsfw.py
│   ├── ddpo_aesthethic.py
│   ├── ddpo_cat2.py
│   ├── nsfw.py
│   └── dgx.py
├── ddpo_pytorch/            # Core DDPO implementation
│   ├── rewards.py           # Reward functions and critic wrappers
│   ├── prompts.py           # Prompt generation
│   ├── stat_tracking.py     # Per-prompt stat tracking
│   └── assets/              # Datasets, prompts, embeddings, weights
├── prompts/                 # Misc prompt lists
├── train_critic_clip.py     # Timestep-aware CLIP critic training
├── train_critic_nsfw.py     # NSFW critic training (variant)
├── train_critic_nsfw_classification.py
├── critic_model.py          # Critic model definition (used by critics)
```

## 🛠 Installation

Requires Python 3.10 or newer.

```bash
git clone <repository-url>
cd rl-machine-unlearning
pip install -e .
pip install -r requirements.txt
```

### Dependencies

- PyTorch 2.8.0+
- Diffusers 0.35.1+
- Accelerate
- PEFT
- Transformers
- CLIP
- Weights & Biases (optional)

## 🎯 Quick Start

### 1. Basic DDPO Training

```bash
# Activate virtual environment
source .venv/bin/activate

# Run standard DDPO training with default config
accelerate launch scripts/training/train.py --config config/ddpo_aesthethic.py
```

### 2. Actor-Critic Training

```bash
# Train with AC method using dummy critic (no external weights)
./scripts/run_ac_example.sh

# Train with aesthetic critic (requires pretrained weights; see Notes below)
./scripts/run_ac_aesthetic.sh
```

### 3. Cat Unlearning (Complete Pipeline)

```bash
# 1. Prepare cat embeddings (if not already present)
python3 scripts/testing/prepare_cat_embeddings.py

# 2. Run cat unlearning training (uses built-in cat_likeness reward)
./scripts/run_cat_unlearning.sh

# 3. Test unlearning progression
./scripts/run_unlearn_test.sh
```

## 🧠 Actor-Critic Implementation

### Theory

The AC method enhances DDPO by using a critic to predict value functions for intermediate states:

```
∇_θ J(θ) = E[∑_{t=1}^T ∇_θ log p_θ(x_{t-1}|x_t,c) * (r(x_0,c) - V_φ(x_t,c,t))]
```

Where:
- `V_φ(x_t,c,t)` is the critic's prediction of expected terminal reward from state `(x_t,c,t)`
- `r(x_0,c)` is the terminal reward
- The advantage `A_t = r(x_0,c) - V_φ(x_t,c,t)` provides better variance reduction

### Key Features

- **Per-timestep Advantages**: Computes advantages for each denoising timestep
- **Image Space Critic**: Operates on decoded images, not latent representations
- **Backward Compatibility**: Original DDPO remains unchanged when AC is disabled
- **Memory Optimized**: On-demand VAE decoding to minimize memory usage

### Configuration

Enable AC in your config:

```python
config.use_actor_critic = True
config.critic_fn = "dummy_critic"      # or "aesthetic_critic", "cat_likeness_critic"
config.critic_fn_kwargs = {}
config.train.normalize_advantages = True
config.train.time_weighting_beta = 0.9
```

Notes:
- `dummy_critic` works out-of-the-box and demonstrates AC plumbing.
- `aesthetic_critic` and `cat_likeness_critic` expect pretrained critic weights at paths referenced in `ddpo_pytorch/rewards.py`. You may update those paths or provide the weights accordingly.

## 🎨 Reward Functions

### 1. Aesthetic Score
- Uses CLIP-based aesthetic scorer
- Higher scores = more aesthetic images
- Pre-trained on LAION dataset

### 2. Cat-likeness (for Unlearning)
- Computes similarity to COCO cat images (uses `ddpo_pytorch/assets/cat_embeddings.pt`)
- Uses CLIP embeddings for comparison; thresholded sigmoid scoring
- Higher raw similarity = more cat-like; reward is reversed for unlearning

### 3. Custom Rewards
- Easy to add: follow the function interfaces in `ddpo_pytorch/rewards.py`

## 🔧 Configuration Options

### Memory Optimization

For systems with limited memory:

```python
# Low-memory configuration
config.use_lora = True
config.mixed_precision = "fp16"
config.sample.num_steps = 20
config.train.timestep_fraction = 0.5
config.sample.batch_size = 1
config.train.gradient_accumulation_steps = 4
```

### Training Parameters

Key hyperparameters in `config/base.py`:

- `sample.batch_size`: Images per GPU per epoch
- `train.batch_size`: Training batch size per GPU
- `train.gradient_accumulation_steps`: Gradient accumulation
- `sample.num_batches_per_epoch`: Number of batches per epoch
- `train.timestep_fraction`: Fraction of timesteps to train on

## 📊 Evaluation and Testing

### Unlearning Progression Test

Visualize how well the model unlearns across training:

```bash
python3 scripts/testing/test_unlearning_progression.py \
    --log_dir "logs/your_training_run" \
    --output_dir "results/unlearning_test"
```

### Reward and Image Generation Utilities
- `scripts/testing/generate_images.py`: quick image generation from a pipeline/checkpoint
- `scripts/testing/generate_images_for_metrics.py`: bulk image generation for metrics

### Gradient Variance Tracking

The AC method includes gradient variance tracking to demonstrate variance reduction benefits. Check Weights & Biases logs for gradient statistics.

## 🏗 Critic Training
This repo includes a standalone, timestep-aware CLIP critic training script:

- `train_critic_clip.py` trains a critic on DDIM intermediate states using class prompts.

High-level flow:
- Generate images from SD1.5 for prompts and timesteps
- Encode via VAE/CLIP; predict scores per timestep
- Save critic weights; plug into AC via `ddpo_pytorch/rewards.py`

Example:
```bash
python3 train_critic_clip.py \
  --help
```
Adjust training flags within the script or adapt it for your dataset. For NSFW variants, see `train_critic_nsfw*.py`.

## 📈 Results and Visualizations

The framework generates several types of outputs:

1. **Training Logs**: Weights & Biases integration for monitoring
2. **Checkpoints**: Model weights saved during training
3. **Progression Visualizations**: Images showing unlearning progression
4. **Gradient Statistics**: Variance reduction analysis
5. **Reward Curves**: Training progress over time

## 🔬 Research Applications

This framework is designed for:

- **Machine Unlearning**: Removing specific concepts from diffusion models
- **Reward Learning**: Training models on custom reward functions
- **Policy Optimization**: Advanced RL methods for diffusion models
- **Concept Erasure**: Targeted removal of unwanted behaviors

## 🐛 Troubleshooting

### Memory Issues
- Use LoRA: `config.use_lora = True`
- Reduce batch size: `config.sample.batch_size = 1`
- Enable mixed precision: `config.mixed_precision = "fp16"`

### Training Issues
- Check gradient accumulation: Ensure `sample.batch_size % train.batch_size == 0`
- Verify critic loading: Check that critic weights are loaded correctly
- Monitor gradient variance: Use wandb logs to track training stability

### Performance Issues
- AC method is slower due to additional computations
- Use smaller timestep fractions for faster training
- Consider pre-training critics for better performance

## 📚 Theoretical Background

This implementation is based on:

1. **DDPO**: [Denoising Diffusion Policy Optimization](https://rl-diffusion.github.io/)
2. **Actor-Critic Methods**: Policy gradient with value function approximation
3. **Machine Unlearning**: Targeted concept removal from neural networks
4. **LoRA**: Low-rank adaptation for efficient fine-tuning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original DDPO implementation by [kvablack](https://github.com/kvablack/ddpo-pytorch)
- Diffusers library by Hugging Face
- CLIP model by OpenAI
- COCO dataset by Microsoft

**Note**: This implementation is research-oriented and may require significant computational resources for training. The default hyperparameters are designed for experimentation, not production use.
