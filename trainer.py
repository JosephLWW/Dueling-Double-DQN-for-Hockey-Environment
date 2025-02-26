import numpy as np
import torch
import gzip
import argparse
from gymnasium import spaces
from hockey_env import HockeyEnv, BasicOpponent
from agent import DuelingAgent
import os
import random
from config import config, set_seed

# CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

mode = config.get("mode", "NORMAL").upper()
opponent = config.get("opponent", "weak")
env = HockeyEnv()
env.mode = mode

if opponent == "static":
    player2 = None
elif opponent == "weak":
    player2 = BasicOpponent(weak=True)
else:
    player2 = BasicOpponent(weak=False)

obs_agent2 = env.obs_agent_two() if player2 else None

discrete_actions = 8
max_steps = 300

# Initializations
avg_total_reward = 0
p2 = 0
p1 = 0

set_seed(config["seed"])
max_episodes = config["episodes"]

ac_map = {tuple(env.discrete_to_continous_action(i)): i for i in range(discrete_actions)}
ac_space = spaces.Discrete(len(ac_map))
o_space = env.observation_space

q_agent = DuelingAgent(observation_space=o_space, action_space=ac_space, config=config)
q_agent.Q.to(device)
q_agent.target_Q.to(device)

# Save & Load
script_dir = os.path.dirname(os.path.abspath(__file__))
trainings_dir = os.path.join(script_dir, "Trainings")
os.makedirs(trainings_dir, exist_ok=True)

if config.get("load_model", False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    suffix_load = config.get("suffix_load", "") #name file
    model_weights_load = os.path.join(trainings_dir, f"model_weights{suffix_load}.pth")
    optimizer_state_load = os.path.join(trainings_dir, f"optimizer_state{suffix_load}.pth")
    replay_buffer_load = os.path.join(trainings_dir, f"training_stats{suffix_load}.pkl.gz")
    
    print("model weights path:", model_weights_load)
    
    if os.path.exists(model_weights_load):
        q_agent.Q.load_state_dict(torch.load(model_weights_load, map_location=device))
        print(f"Loaded model weights from {model_weights_load}")
    else:
        print("No weights file found, continuing without loading")
    
    if os.path.exists(optimizer_state_load):
        q_agent.Q.optimizer.load_state_dict(torch.load(optimizer_state_load, map_location=device))
        print(f"Loaded optimizer state from {optimizer_state_load}")
    else:
        print("No optimizer state file found, continuing without loading.")
    
    # load replay buffer
    if "suffix_load" in config:
        print("replay buffer path:", replay_buffer_load)
        if os.path.exists(replay_buffer_load):
            try:
                # uncompress
                with gzip.open(replay_buffer_load, 'rb') as f:
                    loaded_data = torch.load(f, map_location=device)
                if "replay_buffer" in loaded_data:
                    q_agent.buffer = loaded_data["replay_buffer"]
                    print(f"Loaded replay buffer from {replay_buffer_load}")
                else:
                    print("No replay buffer found in file, continuing without it")
            except Exception as e:
                print("Error loading replay buffer:", e)
        else:
            print("Replay buffer file not found, continuing without loading.")


ob, _info = env.reset()
stats = []
losses = []

# Training loop
for i in range(max_episodes):
    total_reward = 0
    ob, _info = env.reset()
    ob = torch.tensor(ob, dtype=torch.float32, device=device) 

    for t in range(max_steps):
        done = False
        a = q_agent.act(ob)
        a1 = env.discrete_to_continous_action(a)

        # Opponen behavior
        if player2:
            a2 = player2.act(obs_agent2)  # Active
        else:
            a2 = np.zeros_like(a1)  # Static

        a_step = np.hstack([a1, a2])
        obs_agent2 = env.obs_agent_two() if player2 else None

        ob_new, reward, done, trunc, _info = env.step(a_step)
        total_reward += reward

        ob_new = torch.tensor(ob_new, dtype=torch.float32, device=device)

        q_agent.store_transition((ob, a, reward, ob_new, done))
        ob = ob_new

        if done:
            break

    losses.extend(q_agent.train(32))
    q_agent.decay_epsilon()
    avg_total_reward += total_reward
    stats.append([
        i, 
        total_reward, 
        t + 1, 
        q_agent._eps, 
        _info["winner"], 
        _info["reward_closeness_to_puck"], 
        _info["reward_touch_puck"], 
        _info["reward_puck_direction"]
    ])

    if (i - 1) % 100 == 0:
        print(f"Episode {i} done after {t+1} steps. Reward: {total_reward}")

    if _info.get("winner") == 1:
        p1 += 1
    elif _info.get("winner") == -1:
        p2 += 1

    if (i - 1) % 200 == 0:
        print(f"Player 1: {p1} points. Player 2: {p2} points")

# nombres de archivos según configuración
optional_suffix = config.get("suffix", "")
# filename_suffix = f"{mode}_opponent_{opponent}{optional_suffix}"

model_weights_file = os.path.join(trainings_dir, f"model_weights{optional_suffix}.pth")
optimizer_state_file = os.path.join(trainings_dir, f"optimizer_state{optional_suffix}.pth")
training_stats_file = os.path.join(trainings_dir, f"training_stats{optional_suffix}.pkl")

try:
    # Guardar el modelo entrenado
    torch.save(q_agent.Q.state_dict(), model_weights_file)
    torch.save(q_agent.Q.optimizer.state_dict(), optimizer_state_file)

    training_data = {
        'losses': losses,
        'stats': stats,
        'config': q_agent._config
    }

    if config.get("save_buffer", False):
        training_data['replay_buffer'] = q_agent.buffer
        print("Replay buffer saved.")
    else:
        print("Replay buffer not saved")

    # Compress
    compressed_file = training_stats_file + ".gz"
    with gzip.open(compressed_file, 'wb') as f:
        torch.save(training_data, f)
    
    # Report
    print(f"Model {optional_suffix} and stats saved in {script_dir}")
except Exception as e:
    print(f"Error in: {e}")

env.close()
