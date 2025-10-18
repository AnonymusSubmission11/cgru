import ml_collections
from config.base import get_config as get_base_config


def get_config():
    """Configuration for using the Actor-Critic method with trained style critic"""
    config = get_base_config()  # Start with base config
    
    # Enable Actor-Critic method
    config.use_actor_critic = True
    config.critic_fn = "van_gogh_style"  # Use the trained style critic
    config.critic_fn_kwargs = {}
    
    # Set logging and run name
    config.logdir = "logs/ac_training"
    config.run_name = "ddpo_ac_style_critic"
    
    # Use style score as reward function (same as critic training)
    config.reward_fn = "van_gogh_style"
    
    config.train.learning_rate = 3e-4
    config.train.batch_size = 2  # Increased for 48GB GPU
    config.train.gradient_accumulation_steps = 4  # Increased for better gradient estimates
    
    # Enable time weighting for AC method (default beta=0.9)
    config.train.time_weighting_beta = 0.0
    # Disable advantage normalization by default
    config.train.normalize_advantages = True

    # Sampling configuration optimized for 48GB GPU
    config.sample.batch_size = 4  # Increased for 48GB GPU
    config.sample.num_batches_per_epoch = 4  # More samples per epoch
    config.sample.num_steps = 50  # Increased from 20 to 30 steps
    
    # Training epochs
    config.num_epochs = 201

    config.prompt_fn = "van_gogh_dataset"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
        
    
    return config
