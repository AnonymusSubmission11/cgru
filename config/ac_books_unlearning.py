#!/usr/bin/env python3
"""
Configuration for Actor-Critic books unlearning training.
"""

from config.base import get_base_config

def get_config():
    config = get_base_config()
    
    # Enable Actor-Critic method
    config.use_actor_critic = True
    config.critic_fn = "books_critic"
    config.critic_fn_kwargs = {}

    config.logdir = "logs/ac_books_unlearning"
    config.run_name = "books_unlearning_ac"
    
    # Use books unlearning reward
    config.reward_fn = "books_unlearning"
    
    # Training configuration
    config.train.learning_rate = 3e-4
    config.train.batch_size = 2  # Conservative for memory
    config.train.gradient_accumulation_steps = 4  # Increased for better gradient estimates
    
    config.train.time_weighting_beta = 0.0
    config.train.normalize_advantages = True
    
    # Sampling configuration optimized for 48GB GPU
    config.sample.batch_size = 4  # Increased for 48GB GPU
    config.sample.num_batches_per_epoch = 4  # More samples per epoch
    config.sample.num_steps = 50  # Increased from 20 to 30 steps
    
    # Training epochs
    config.num_epochs = 201

    # Logging
    config.prompt_fn = "books_diffusion_dataset"  
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    
    return config
