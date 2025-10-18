import ml_collections
from config.base import get_config as get_base_config


def get_config():
    """Configuration for using the Actor-Critic method with trained style critic"""
    config = get_base_config()  # Start with base config
    config.pretrained.model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # config.resume_from = "logs/critic_clip_nsfw/ddpo_ac_nsfw_critic_2025.10.09_11.11.52/checkpoints/checkpoint_6"
    # Enable Actor-Critic method
    config.use_actor_critic = True
    config.critic_fn = "nsfw_clip_critic"  # Use the trained style critic
    config.critic_fn_kwargs = {}
    
    # Set logging and run name
    config.logdir = "logs/nsfw_q16_clip_unlearning"
    config.run_name = "nsfw_q16_clip_unlearning"

    # Use style score as reward function (same as critic training)
    config.reward_fn = "nsfw_clip_unlearning"

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
    config.prompt_fn = "nsfw"
    config.prompt_fn_kwargs = {}
    
    # Logging
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    
    return config
