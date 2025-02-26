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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def coach_training():
    total_episodes = config["episodes"]

    # Phase 1: Specialization
    phase1_total = int(0.20 * total_episodes)        # 20% of total episodes for phase 1
    phase1_shooting = phase1_total // 2              # Half of phase 1 for TRAIN_SHOOTING
    phase1_defense = phase1_total - phase1_shooting    # Half of phase 1 for TRAIN_DEFENSE

    # Phase 2: Integration
    phase2_episodes = int(0.40 * total_episodes)     # 40% of total episodes for phase 2
    phase3_episodes = total_episodes - phase1_total - phase2_episodes  # 40% of total episodes for phase 3

    max_steps = 300

    # Initialize environment and agent
    env = HockeyEnv()
    discrete_actions = 8  
    ac_space = spaces.Discrete(discrete_actions)
    q_agent = DuelingAgent(observation_space=env.observation_space, action_space=ac_space, config=config)
    q_agent.Q.to(device)
    q_agent.target_Q.to(device)
    
    # Data
    stats = []
    losses = []
    avg_total_reward = 0
    p1_wins = 0
    p2_wins = 0
    set_seed(config["seed"])

    # Train loop
    for episode in range(total_episodes):
        # Configuration selection based on the current phase
        if episode < phase1_total:
            # Phase 1: specialization
            if episode < phase1_shooting:
                env.mode = "TRAIN_SHOOTING"
            else:
                env.mode = "TRAIN_DEFENSE"
            opponent_type = "weak"
        elif episode < (phase1_total + phase2_episodes):
            # Fase 2: Integración
            env.mode = "NORMAL"
            integration_idx = episode - phase1_total
            # Chances of facing a weak opponent decay linearly
            prob_weak = 1.0 - (integration_idx / (phase2_episodes - 1))
            opponent_type = "weak" if random.random() < prob_weak else "strong"
        else:
            # Fase 3: Full competency
            env.mode = "NORMAL"
            opponent_type = "strong"

        # Weak or strong opponent
        if opponent_type == "static":
            player2 = None
        elif opponent_type == "weak":
            player2 = BasicOpponent(weak=True)
        else:
            player2 = BasicOpponent(weak=False)

        # Reset
        ob, _info = env.reset()
        ob = torch.tensor(ob, dtype=torch.float32, device=device)
        obs_agent2 = env.obs_agent_two() if player2 is not None else None

        total_reward = 0
        for t in range(max_steps):
            done = False
            a = q_agent.act(ob)
            a1 = env.discrete_to_continous_action(a)

            # Acción del oponente
            if player2:
                a2 = player2.act(obs_agent2)
            else:
                a2 = np.zeros_like(a1)
            
            a_step = np.hstack([a1, a2])
            obs_agent2 = env.obs_agent_two() if player2 is not None else None

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
            episode, 
            total_reward, 
            t + 1, 
            q_agent._eps, 
            _info.get("winner", 0),
            _info.get("reward_closeness_to_puck", 0),
            _info.get("reward_touch_puck", 0),
            _info.get("reward_puck_direction", 0)
        ])

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{total_episodes} finished in {t+1} steps. Reward: {total_reward}")

        if _info.get("winner") == 1:
            p1_wins += 1
        elif _info.get("winner") == -1:
            p2_wins += 1

        if (episode + 1) % 200 == 0:
            print(f"Player 1 (agent): {p1_wins} | Player 2 (bot): {p2_wins}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    trainings_dir = os.path.join(script_dir, "Trainings")
    os.makedirs(trainings_dir, exist_ok=True)
    optional_suffix = config.get("suffix", "")
    model_weights_file = os.path.join(trainings_dir, f"model_weights{optional_suffix}.pth")
    optimizer_state_file = os.path.join(trainings_dir, f"optimizer_state{optional_suffix}.pth")
    training_stats_file = os.path.join(trainings_dir, f"training_stats{optional_suffix}.pkl")

    try:
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

        compressed_file = training_stats_file + ".gz"
        with gzip.open(compressed_file, 'wb') as f:
            torch.save(training_data, f)
        print(f"Model and stats saved in {script_dir}")
    except Exception as e:
        print(f"Error: {e}")

    env.close()

if __name__ == "__main__":
    coach_training()
