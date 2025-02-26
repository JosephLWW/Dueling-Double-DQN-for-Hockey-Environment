# Dueling-Double-DQN-for-Hockey-Environment
A deep reinforcement learning project implementing a Dueling Double DQN (DDDQN) agent for the Hockey environment (https://github.com/martius-lab/hockey-env/). It uses neural networks as function approximators and includes enhancements such as prioritized experience replay, Intrinsic Curiosity Module, and curriculum learning.



## Features

- **Dueling Architecture:** Separates value and advantage streams.
- **Double DQN:** Reduces overestimation bias.
- **Prioritized Experience Replay:** Improves sampling efficiency.
- **Curriculum Learning:** Train using specialized modes via `coach.py`.
- **Optional ICM:** Enhances exploration through intrinsic rewards.

## Usage

Execute the trainer.py method to start training the model and coach.py if you want to do it using the curriculum learning implementation.
The output includes the model weights, state optimizer and stats (including memory buffer if wished).

edit config.py for training configuration and hyperparameter setting

