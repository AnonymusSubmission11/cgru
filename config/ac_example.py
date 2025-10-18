import ml_collections
from config.base import get_config as get_base_config


def get_config():
    """Example configuration for using the Actor-Critic method"""
    config = get_base_config()  # Start with base config
    
    # Enable Actor-Critic method
    config.use_actor_critic = True
    config.critic_fn = "dummy_critic"  # Use the dummy critic
    config.critic_fn_kwargs = {}
    
    # Set logging and run name
    config.logdir = "logs/ac_training"
    config.run_name = "ddpo_ac_example"
    
    # You can also use a different reward function
    config.reward_fn = "jpeg_compressibility"
    
    # Adjust training parameters if needed
    config.train.learning_rate = 3e-4
    config.train.batch_size = 1
    config.sample.batch_size = 1
    
    # Reduce epochs for testing
    config.num_epochs = 2
    
    return config
