import ml_collections
from config.base import get_config as get_base_config


def get_config():
    """Configuration for using the Actor-Critic method with trained aesthetic critic"""
    config = get_base_config()  # Start with base config
     
    # Set logging and run name
    config.logdir = "logs/ddpo_training"
    config.run_name = "ddpo_aesthetic_critic"
    
    # Use aesthetic score as reward function (same as critic training)
    config.reward_fn = "aesthetic_score"
    
    config.train.learning_rate = 3e-4
    config.train.batch_size = 2  # Increased for 48GB GPU
    config.train.gradient_accumulation_steps = 4  # Increased for better gradient estimates

    # Sampling configuration optimized for 48GB GPU
    config.sample.batch_size = 4  # Increased for 48GB GPU
    config.sample.num_batches_per_epoch = 4  # More samples per epoch
    config.sample.num_steps = 50  # Increased from 20 to 30 steps
    
    # Training epochs
    config.num_epochs = 200

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
        
    
    return config
