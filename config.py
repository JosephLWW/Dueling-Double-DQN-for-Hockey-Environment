import numpy as np
import torch
import random

# For experiments
def set_seed(seed_value):
    if seed_value is None:
        return
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

# Hyperparameters (DDDQN V2 20/02)
config = {
    "seed": 42,
    "episodes": 30000,
    "suffix": "_lr_1e-3",      # Suffix for the model
    "mode": "NORMAL",        # Mode: NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE
    "opponent": "weak",      # Oponent: weak, strong, static
    "use_icm": False,      # ICM
    "save_buffer": False,    # save replay buffer
    "load_model": False,   # resume training
    "suffix_load": "_local_test",

    # DDDQN
    "learning_rate": 1e-4  ,        # Eta
    "discount": 0.95,               # Gamma
    "target_update_freq": 500,
    "eps": 1,
    "eps_min": 0.01,
    "eps_decay": 0.9995,

    # PER
    "buffer_size": 1000000,
    "batch_size": 32,
    "eps_td": 5e-3,     # Avoid 0 td error
    "alpha": 0.5,
    "beta": 0.4,
    "beta_increment_per_sampling": 1e-7,
    "max_beta": 0.9,
    "min_priority": 1e-8, # Only to avoid 0 priority

    # ICM
    "icm_lr": 1e-6,        # learning rate
    "icm_beta": 0.6,       # balance inverse (0) and forward (1)
    "icm_feature_dim": 32, # Dimension for the feature space
    "icm_scale": 1e-4
}
