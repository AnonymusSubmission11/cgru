import ml_collections
from config.base import get_config as get_base_config


def get_config():
    """Configuration for using the Actor-Critic method with cat-likeness critic for unlearning"""
    config = get_base_config()  # Start with base config
    
    # Enable Actor-Critic method
    config.use_actor_critic = True
    config.critic_fn = "cat_likeness_critic"  # Use the trained cat-likeness critic
    config.critic_fn_kwargs = {}
    
    # Set logging and run name
    config.logdir = "logs/ac_cat_unlearning"
    config.run_name = "ddpo_ac_cat_unlearning"
    
    # Use cat-likeness as reward function (higher = more cat-like, we want to minimize this)
    config.reward_fn = "cat_likeness"
    
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
    
    # Use cat datasets for unlearning
    config.prompt_fn = "cats_50_percent"  # Start with 100% cats for unlearning
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
        
    
    return config
