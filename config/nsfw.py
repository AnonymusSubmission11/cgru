#!/usr/bin/env python3
"""
Configuration for Actor-Critic NSFW unlearning training.
"""

import ml_collections
import imp
import os

# Load base config the same way as dgx.py does
base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def get_config():
    config = base.get_config() 
    config.pretrained.model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # config.resume_from = "logs/nsfw_ddpo_unlearning/nsfw_ddpo_unlearning_2025.10.14_17.25.49/checkpoints/checkpoint_5"

    config.logdir = "logs/_nibbler_nsfw_ddpo_unlearning"
    config.run_name = "_nibbler_nsfw_ddpo_unlearning"

    # Use NSFW unlearning reward - choose one of the reward functions
    # Option 1: Using NudeNet detector (requires nudenet package)
    config.reward_fn = "nsfw_unlearning"
    
    # Option 2: Using Q16 CLIP-based classifier (requires prompts file)
    # config.reward_fn = "q16_inappropriateness_reward"
    # config.reward_fn_kwargs = {
    #     "prompts_path": "path/to/q16_prompts.pkl"  # Update with actual path
    # }
    
    # Training configuration
    config.train.learning_rate = 3e-4
    config.train.batch_size = 2  # Conservative for memory
    config.train.gradient_accumulation_steps = 4  # Increased for better gradient estimates
    
    config.train.time_weighting_beta = 0.0
    config.train.normalize_advantages = True
    
    # Sampling configuration optimized for 48GB GPU
    config.sample.batch_size = 4
    config.sample.num_batches_per_epoch = 4  # More samples per epoch
    config.sample.num_steps = 50  # Inference steps
    
    # Training epochs
    config.num_epochs = 201
    
    # Use LoRA
    config.use_lora = True
    config.save_freq = 10
    
    # Prompts
    config.prompt_fn = "nsfw_nibbler"
    config.prompt_fn_kwargs = {}
    
    # Logging
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    
    return config