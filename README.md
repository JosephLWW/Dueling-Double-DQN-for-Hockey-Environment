# Dueling-Double-DQN-for-Hockey-Environment

A deep reinforcement learning project implementing a Dueling Double DQN (DDDQN) agent for the Hockey environment (https://github.com/martius-lab/hockey-env/). DQN is an Off-Policy RL algorithm that uses neural networks as their function approximation method. This implementation includes enhancements such as prioritized experience replay, Intrinsic Curiosity Module, and curriculum learning on top of a dueling and double architecture for DQN.

## Features

- **Dueling Architecture:** Separates value and advantage streams.
- **Double DQN:** Reduces overestimation bias.
- **Prioritized Experience Replay:** Improves experience sampling.
- **ICM:** Enhances exploration through intrinsic rewards.
- **Standard Training**: Train with custom game mode and opponent via `trainer.py`.
- **Curriculum Learning:** Train using curriculum learning via `coach.py`.
- **Model Tester**: Use the `Model Tester.ipynb` notebook to evaluate and test the trained model.
- **Example Model trained with curriculum learning**: Use as an example model to evaluate, render, play against or retrain. It consists of the model weights (`model_weights_dddqn.pth`), optimizer state for retraining (`optimizer_state_dddqn.pth`) and training stats (`training_stats_dddqn.pkl.gz`) (Experience replay buffer was not included to save storage space).
- **Hockey Environment**: Based on Gymnasium and Box2D (`hockey_env.py`). Acknowledgments to Dr. Georg Martius


## Usage

### Requirements
This project was developed with python 3.10.11. All necessary dependencies are specified in `requirements.txt` (please install via `pip`).

### Training
1. Adjust `config.py` for training configuration and hyperparameter setting.
2. Execute the `trainer.py` method to start training the model in custom mode and `coach.py` if you want to do it using the curriculum learning implementation.
3. Retraining of old models is supported. Old model weights are required in that case. Optimizer state and experience replay buffer are recommended.
   - You can use `model_weights_dddqn.pth`, `optimizer_state_dddqn.pth` and `training_stats_dddqn.pkl.gz` as an example
4. The output includes the model weights, state optimizer and training statistics (including memory buffer if wished).
   - Customize output file names if needed

### Testing
To test a previously trained model, use output the weights and stat files and run the `Model Tester.ipynb` notebook. Follow the instructions thoroughly. An example model is provided in `training_stats_dddqn.pkl.gz` and `model_weights_dddqn.pth`.

![image](https://github.com/user-attachments/assets/3972cfb2-1b86-438b-8584-71875e6954d7)

@ Uni-Tuebingen


Repository Structure
---------------------
<pre>
.
├── Assets
    ├── RL_Presentation_LQ_24_25.mp4
    ├── DDDQN vs Strong Opponent.mp4
    ├── RL_Course_2024_25__Final_Project_Report.pdf
    └── Tournament_results.ods
├── LICENSE
├── Model Tester.ipynb
├── README.md
├── agent.py
├── coach.py
├── config.py
├── hockey_env.py
├── icm.py
├── memory.py
├── model_weights_dddqn.pth
├── optimizer_state_dddqn.pth
├── qfunction.py
├── requirements.txt
├── tester.py
├── trainer.py
└── training_stats_dddqn.pkl.gz
</pre>
