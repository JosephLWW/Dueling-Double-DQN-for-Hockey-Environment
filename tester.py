import numpy as np
import torch
import random
import gzip
import os
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
from hockey.hockey_env import HockeyEnv_BasicOpponent, HumanOpponent
from agent import DuelingAgent
from gymnasium import spaces

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class ModelTester:
    def __init__(self, config, base_path=".", device=None):
        self.config = config
        self.base_path = base_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = None
        self.training_data = None
        self.replay_buffer = None
        seed = config.get("seed", None)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        self.o_space = None
        self.ac_space = None

    def init_env(self, weak_opponent=True, render=False, random_start=True):
        self.env = h_env.HockeyEnv_BasicOpponent(weak_opponent=weak_opponent)
        if render:
            _ = self.env.render()
        obs, info = self.env.reset(one_starting=np.random.choice([True, False]) if random_start else True)
        self.o_space = self.env.observation_space
        discrete_actions = 8
        ac_map = {tuple(self.env.discrete_to_continous_action(i)): i for i in range(discrete_actions)}
        self.ac_space = spaces.Discrete(len(ac_map))
        return obs

    def load_model(self, optional_suffix=""):
        if self.o_space is None or self.ac_space is None:
            raise ValueError("No se ha inicializado el entorno. Llama antes a init_env().")
        self.agent = DuelingAgent(observation_space=self.o_space, action_space=self.ac_space, config=self.config)
        model_file = os.path.join(self.base_path, f"model_weights{optional_suffix}.pth")
        print(f"Estamos cargando el modelo desde: {model_file}")
        state_dict = torch.load(model_file, map_location=self.device)
        self.agent.Q.load_state_dict(state_dict)
        self.agent.Q.to(self.device)
        self.agent.Q.eval()

    def load_stats(self, suffixes=""):
        if isinstance(suffixes, str):
            suffixes = [suffixes]
        if len(suffixes) == 1:
            stats_file = os.path.join(self.base_path, f"training_stats{suffixes[0]}.pkl.gz")
            print(f"Estamos cargando las estadísticas desde: {stats_file}")
            try:
                with gzip.open(stats_file, 'rb') as f:
                    data = torch.load(f, map_location=self.device)
                    self.training_data = {suffixes[0]: data}
                    if 'replay_buffer' in data:
                        self.replay_buffer = data['replay_buffer']
                        print("El replay buffer fue asignado correctamente.")
                    print(f"Estadísticas de {suffixes[0]} cargadas con éxito.")
                    if 'losses' in self.training_data[suffixes[0]]:
                        print(f"Primeros valores de 'losses' en {suffixes[0]}: {self.training_data[suffixes[0]]['losses'][:5]}")
                    else:
                        print(f"'losses' no encontrado en {suffixes[0]}.")
            except Exception as e:
                print(f"Error al cargar {stats_file}: {e}")
        else:
            self.training_data = {}
            for suffix in suffixes:
                stats_file = os.path.join(self.base_path, f"training_stats{suffix}.pkl.gz")
                print(f"Estamos cargando las estadísticas desde: {stats_file}")
                try:
                    with gzip.open(stats_file, 'rb') as f:
                        self.training_data[suffix] = torch.load(f, map_location=self.device)
                        print(f"Estadísticas de {suffix} cargadas con éxito.")
                except Exception as e:
                    print(f"Error al cargar {stats_file}: {e}")
        print(f"Se han cargado {len(suffixes)} archivo(s) de estadísticas.")

    def plot_reward_and_winrate(self, window_size=100):
        if self.training_data is None:
            print("No hay training_data. Llama a load_stats() primero.")
            return
        if len(self.training_data) == 1:
            data = list(self.training_data.values())[0]
        else:
            print("Se han cargado múltiples archivos. Usa plot_rewards_and_winrate() en su lugar.")
            return
        stats_np = np.array(data['stats'])
        rewards = stats_np[:, 1]
        wins = stats_np[:, 4]
        smoothed_rewards = running_mean(rewards, window_size)
        winrate = running_mean(np.cumsum(wins == 1) / (np.arange(1, len(wins) + 1)), window_size)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].plot(rewards, label="Total Reward / Return per Episode", color="#337BFF", alpha=0.2)
        axes[0].plot(range(window_size - 1, len(rewards)), smoothed_rewards, label=f"Average Reward per {window_size} Episodes", color="#FF5733")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Final Return")
        axes[0].set_title("Training Reward Over Episodes")
        axes[0].legend(loc="lower right", fontsize=11, frameon=False, edgecolor="grey", fancybox=True)
        axes[1].plot(100 * winrate, label="Average winrate", color="#FF5733", alpha=0.8)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Winrate (%)")
        axes[1].set_title("Winrate Over Episodes")
        axes[1].legend(loc="lower right", fontsize=11, frameon=False, edgecolor="grey", fancybox=True)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15)
        plt.show()

    def plot_rewards_and_winrate(self, window_size=100, custom_names=None):
        if not self.training_data:
            print("No hay training_data. Llama a load_stats() primero.")
            return
        if custom_names is None:
            custom_names = [name.lstrip('_') for name in self.training_data.keys()]
        elif len(custom_names) != len(self.training_data):
            print("La cantidad de nombres personalizados no coincide con la cantidad de archivos cargados.")
            return
        colors = ["#337BFF", "#FFAA00", "#FF5733", "#AA00FF", "#33CC33"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for i, ((suffix, data), name) in enumerate(zip(self.training_data.items(), custom_names)):
            if 'stats' not in data:
                print(f"No se encontraron estadísticas en {suffix}.")
                continue
            stats_np = np.array(data['stats'])
            rewards = stats_np[:, 1]
            wins = stats_np[:, 4]
            smoothed_rewards = running_mean(rewards, window_size)
            winrate = running_mean(np.cumsum(wins == 1) / (np.arange(1, len(wins) + 1)), window_size)
            color = colors[i % len(colors)]
            label = name
            axes[0].plot(rewards, label=f"{label} - Episode Return", color=color, alpha=0.2)
            axes[0].plot(range(window_size - 1, len(rewards)), smoothed_rewards, label=f"{label} - Avg per {window_size} ep", color=color)
            axes[1].plot(100 * winrate, label=f"{label}", color=color)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Episode Return")
        axes[0].set_title("Training Reward Over Episodes")
        axes[0].legend(loc="lower right", fontsize=11, frameon=False, edgecolor="grey", fancybox=True)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Winrate (%)")
        axes[1].set_title("Cumulative Winrate Over Episodes")
        axes[1].legend(loc="lower right", fontsize=11, frameon=False, edgecolor="grey", fancybox=True)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15)
        plt.show()

    def plot_rewards_and_winrate_last_500(self, window_size=100, win_window=500, custom_names=None):
        if not self.training_data:
            print("No hay training_data. Llama a load_stats() primero.")
            return
        if custom_names is None:
            custom_names = [name.lstrip('_') for name in self.training_data.keys()]
        elif len(custom_names) != len(self.training_data):
            print("La cantidad de nombres personalizados no coincide con la cantidad de archivos cargados.")
            return
        colors = ["#337BFF", "#FFAA00", "#FF5733", "#AA00FF", "#33CC33"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        def moving_winrate(wins, window_size):
            win_array = np.where(wins == 1, 1, 0)
            winrate = np.convolve(win_array, np.ones(window_size) / window_size, mode='valid')
            return winrate
        for i, ((suffix, data), name) in enumerate(zip(self.training_data.items(), custom_names)):
            if 'stats' not in data:
                print(f"No se encontraron estadísticas en {suffix}.")
                continue
            stats_np = np.array(data['stats'])
            rewards = stats_np[:, 1]
            wins = stats_np[:, 4]
            smoothed_rewards = running_mean(rewards, window_size)
            winrate = moving_winrate(wins, win_window)
            color = colors[i % len(colors)]
            label = name
            axes[0].plot(rewards, label=f"{label} - Episode Return", color=color, alpha=0.2)
            axes[0].plot(range(window_size - 1, len(rewards)), smoothed_rewards, label=f"{label} - Avg per {window_size} ep", color=color)
            axes[1].plot(range(win_window - 1, len(wins)), 100 * winrate, label=f"{label} - Winrate", color=color)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Episode Return")
        axes[0].set_title("Training Reward Over Episodes")
        axes[0].legend(loc="lower right", fontsize=11, frameon=False, edgecolor="grey", fancybox=True)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Winrate (%)")
        axes[1].set_title(f"Winrate Over {win_window} Episodes")
        axes[1].legend(loc="lower right", fontsize=11, frameon=False, edgecolor="grey", fancybox=True)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15)
        plt.show()

    def plot_no_custom(self, window_size=100):
        if not self.training_data:
            print("No hay training_data. Llama a load_stats() primero.")
            return
        colors = ["#337BFF", "#FFAA00", "#FF5733", "#AA00FF", "#33CC33"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for i, (suffix, data) in enumerate(self.training_data.items()):
            if 'stats' not in data:
                print(f"No se encontraron estadísticas en {suffix}.")
                continue
            stats_np = np.array(data['stats'])
            rewards = stats_np[:, 1]
            wins = stats_np[:, 4]
            smoothed_rewards = running_mean(rewards, window_size)
            winrate = running_mean(np.cumsum(wins == 1) / (np.arange(1, len(wins) + 1)), 10)
            color = colors[i % len(colors)]
            label = f"Run {i+1} ({suffix})"
            axes[0].plot(rewards, label=f"{label} - Total Reward", color=color, alpha=0.2)
            axes[0].plot(range(window_size - 1, len(rewards)), smoothed_rewards, label=f"{label} - Avg Reward", color=color)
            axes[1].plot(100 * winrate, label=f"{label} - Winrate", color=color)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Final Return")
        axes[0].set_title("Training Reward Over Episodes")
        axes[0].legend(loc="lower right", fontsize=11, frameon=False, edgecolor="grey", fancybox=True)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Winrate (%)")
        axes[1].set_title("Winrate Over Episodes")
        axes[1].legend(loc="lower right", fontsize=11, frameon=False, edgecolor="grey", fancybox=True)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15)
        plt.show()

    def plot_loss(self):
        if self.training_data is None:
            print("No hay training_data. Llama a load_stats() primero.")
            return
        if len(self.training_data) == 1:
            data = list(self.training_data.values())[0]
            if 'losses' not in data:
                print("No se encontró 'losses' en el archivo de estadísticas.")
                return
            losses = data['losses']
        else:
            print("Se han cargado múltiples archivos de estadísticas. Maneja cada uno en un bucle.")
            return
        plt.figure(figsize=(20, 6))
        plt.plot(losses, label="Training Loss", alpha=0.7, color="#337BFF")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Loss During Training")
        plt.legend(loc="upper right", fontsize=11, frameon=False, edgecolor="grey", fancybox=True)
        plt.tight_layout()
        plt.show()

    def plot_per_distributions(self):
        if len(self.training_data) != 1:
            print("Debes haber cargado exactamente un modelo/archivo.")
            return
        suffix = next(iter(self.training_data.keys()))
        data = self.training_data[suffix]
        if 'replay_buffer' not in data:
            print("No se encontró 'replay_buffer' en los datos.")
            return
        replay_buffer = data['replay_buffer']
        leaf_start_idx = replay_buffer.sum_tree.max_size - 1
        td_errors = np.abs(replay_buffer.sum_tree.tree[leaf_start_idx:leaf_start_idx + replay_buffer.size])
        sampling_probs = td_errors / td_errors.sum()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(td_errors, bins=50, alpha=0.7, color='#337BFF', edgecolor='black')
        axes[0].set_xlabel("TD Error Value")
        axes[0].set_ylabel("Frecuency")
        axes[0].set_title("TD Error Distribution")
        axes[0].grid(False)
        axes[1].hist(sampling_probs, bins=50, alpha=0.7, color='#FF5733', edgecolor='black')
        axes[1].set_xlabel("Probability Weight")
        axes[1].set_ylabel("Frecuency")
        axes[1].set_title("Probability Distribution for PER Sampling")
        axes[1].grid(False)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25)
        plt.show()

    def play_one_episode(self, steps=251, render=True):
        if self.agent is None:
            print("El agente no está cargado. Por favor, llama a load_model() primero.")
            return
        if not hasattr(self, 'env'):
            print("El entorno no está inicializado. Por favor, llama a init_env() primero.")
            return
        obs, info = self.env.reset(one_starting=np.random.choice([True, False]))
        for _ in range(steps):
            if render:
                self.env.render()
            discrete_action = self.agent.act(obs)
            a1 = self.env.discrete_to_continous_action(discrete_action)
            next_obs, reward, done, truncated, _ = self.env.step(a1)
            obs = next_obs
            if done or truncated:
                break
        self.env.close()

    def play_vs_human(self, steps=251, render=True):
        env = h_env.HockeyEnv()
        if render:
            _ = env.render()
        human_opp = HumanOpponent(env, player=2)
        obs, info = env.reset(one_starting=np.random.choice([True, False]))
        discrete_actions = 8
        ac_map = {tuple(env.discrete_to_continous_action(i)): i for i in range(discrete_actions)}
        ac_space = spaces.Discrete(len(ac_map))
        if self.agent is None:
            print("El agente no está cargado. Por favor, llama a load_model() primero.")
            return
        for _ in range(steps):
            if render:
                env.render()
            discrete_action = self.agent.act(obs)
            a1 = env.discrete_to_continous_action(discrete_action)
            a2 = human_opp.act(obs)
            joint_action = np.hstack([a1, a2])
            next_obs, reward, done, truncated, _info = env.step(joint_action)
            obs = next_obs
            if done or truncated:
                break
        env.close()

    def randomize_environment(self, env):
        W = h_env.VIEWPORT_W / h_env.SCALE
        H = h_env.VIEWPORT_H / h_env.SCALE
        env.player1.position = [
            np.random.uniform(0.1 * W, 0.4 * W),
            np.random.uniform(0.2 * H, 0.8 * H)
        ]
        env.player2.position = [
            np.random.uniform(0.6 * W, 0.9 * W),
            np.random.uniform(0.2 * H, 0.8 * H)
        ]
        env.puck.position = [
            np.random.uniform(0.4 * W, 0.6 * W),
            np.random.uniform(0.3 * H, 0.7 * H)
        ]
        env.player1.linearVelocity = [0, 0]
        env.player2.linearVelocity = [0, 0]
        env.puck.linearVelocity = [0, 0]

    def play_models_balanced(self, model_id1=None, model_id2=None, num_episodes=100, steps=251, render=False):
        if model_id1 is None or model_id2 is None:
            print("Debes proporcionar model_id1 y model_id2!")
            return
        model_file1 = os.path.join(self.base_path, f"model_weights{model_id1}.pth")
        print(f"Estamos cargando Modelo 1 desde: {model_file1}")
        model1 = DuelingAgent(observation_space=self.o_space, action_space=self.ac_space, config=self.config)
        state_dict1 = torch.load(model_file1, map_location=self.device)
        model1.Q.load_state_dict(state_dict1)
        model1.Q.to(self.device)
        model1.Q.eval()
        model_file2 = os.path.join(self.base_path, f"model_weights{model_id2}.pth")
        print(f"Estamos cargando Modelo 2 desde: {model_file2}")
        model2 = DuelingAgent(observation_space=self.o_space, action_space=self.ac_space, config=self.config)
        state_dict2 = torch.load(model_file2, map_location=self.device)
        model2.Q.load_state_dict(state_dict2)
        model2.Q.to(self.device)
        model2.Q.eval()
        env = h_env.HockeyEnv()
        results = {"model_1": 0, "model_2": 0, "draws": 0}
        for episode in range(num_episodes):
            swap = (episode % 2 == 1)
            obs, info = env.reset(one_starting=np.random.choice([True, False]))
            self.randomize_environment(env)
            done = False
            step = 0
            while not done and step < steps:
                if render:
                    env.render()
                if not swap:
                    action1 = model1.act(obs)
                    obs_agent2 = env.obs_agent_two()
                    action2 = model2.act(obs_agent2)
                    a1 = env.discrete_to_continous_action(action1)
                    a2 = env.discrete_to_continous_action(action2)
                else:
                    action1 = model2.act(obs)
                    obs_agent2 = env.obs_agent_two()
                    action2 = model1.act(obs_agent2)
                    a1 = env.discrete_to_continous_action(action1)
                    a2 = env.discrete_to_continous_action(action2)
                joint_action = np.hstack([a1, a2])
                obs, reward, done, truncated, info = env.step(joint_action)
                step += 1
            print(f"Episodio {episode+1}: info['winner'] = {info['winner']}, recompensa = {reward}, swap = {swap}")
            if 'winner' in info:
                if not swap:
                    if info['winner'] == 1:
                        results["model_1"] += 1
                    elif info['winner'] == -1:
                        results["model_2"] += 1
                    else:
                        results["draws"] += 1
                else:
                    if info['winner'] == 1:
                        results["model_2"] += 1
                    elif info['winner'] == -1:
                        results["model_1"] += 1
                    else:
                        results["draws"] += 1
            else:
                if reward > 0:
                    if not swap:
                        results["model_1"] += 1
                    else:
                        results["model_2"] += 1
                elif reward < 0:
                    if not swap:
                        results["model_2"] += 1
                    else:
                        results["model_1"] += 1
                else:
                    results["draws"] += 1
        env.close()
        print(f"Resultados tras {num_episodes} episodios:")
        print(f"  Modelo 1 (ID: {model_id1}) gana: {results['model_1']}")
        print(f"  Modelo 2 (ID: {model_id2}) gana: {results['model_2']}")
        print(f"  Empates: {results['draws']}")
        return results
