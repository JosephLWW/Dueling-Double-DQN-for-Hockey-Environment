# Dueling-Double-DQN-for-Hockey-Environment

A deep reinforcement learning project implementing a Dueling Double DQN (DDDQN) agent for the Hockey environment (https://github.com/martius-lab/hockey-env/). DQN is an Off-Policy RL algorithm that uses neural networks as their function approximation method. This implementation includes enhancements such as prioritized experience replay, Intrinsic Curiosity Module, and curriculum learning on top of a dueling and double architecture for DQN.

## Features

- **Dueling Architecture:** Separates value and advantage streams.
- **Double DQN:** Reduces overestimation bias.
- **Prioritized Experience Replay:** Improves experience sampling.
- **Curriculum Learning:** Train using curriculum learning via `coach.py`.
- **ICM:** Enhances exploration through intrinsic rewards.
- **Model Tester**: Use the "Model Tester.ipynb" notebook to evaluate and test the trained model.
- **Model used for the final tournament**: Use as an example model to evaluate, render, play against or retrain. It consists of the model weights, optimizer state and training stats (Including experience replay buffer).


## Usage

Execute the trainer.py method to start training the model and coach.py if you want to do it using the curriculum learning implementation.
The output includes the model weights, state optimizer and stats (including memory buffer if wished).

To test a previously trained model, use output the weights and stat files.  provided in dddqn_tournament_model.pth and run the Model Tester.ipynb notebook.

edit config.py for training configuration and hyperparameter setting

To test the model trained for the tournament, use the weights provided in "dddqn_tournament_model.pth"


![image](https://github.com/user-attachments/assets/3972cfb2-1b86-438b-8584-71875e6954d7)

## Repository Structure

.
├── LICENSE
├── README.md
├── agent.py
├── coach.py
├── config.py
├── hockey_env.py
├── icm.py
├── memory.py
├── qfunction.py
├── requirements.txt
├── trainer.py
└── assets
    ├── RL_Presentation_LQ_24_25.mp4
    ├── DDDQN vs Strong Opponent.mp4
    ├── RL_Course_2024_25__Final_Project_Report.pdf
    └── Tournament_results.ods

