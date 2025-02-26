# Dueling-Double-DQN-for-Hockey-Environment
A deep reinforcement learning project implementing a Dueling Double DQN (DDDQN) agent for the Hockey environment (https://github.com/martius-lab/hockey-env/). DQN is an Off-Policy RL algorithm that uses neural networks as their function approximation method. This implementation includes enhancements such as prioritized experience replay, Intrinsic Curiosity Module, and curriculum learning on top of a dueling and double architecture for DQN.



## Features

- **Dueling Architecture:** Separates value and advantage streams.
- **Double DQN:** Reduces overestimation bias.
- **Prioritized Experience Replay:** Improves experience sampling.
- **Curriculum Learning:** Train using curriculum learning via `coach.py`.
- **ICM:** Enhances exploration through intrinsic rewards.

## Usage

Execute the trainer.py method to start training the model and coach.py if you want to do it using the curriculum learning implementation.
The output includes the model weights, state optimizer and stats (including memory buffer if wished).

edit config.py for training configuration and hyperparameter setting

To test the model trained for the tournament, use the weights provided in "dddqn_tournament_model.pth"


![image](https://github.com/user-attachments/assets/3972cfb2-1b86-438b-8584-71875e6954d7)
