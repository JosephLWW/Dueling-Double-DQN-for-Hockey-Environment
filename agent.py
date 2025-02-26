import torch
import numpy as np
from memory import Memory
from qfunction import DuelingQFunction
from config import config, set_seed
from icm import IntrinsicCuriosityModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This implementation is based on the course material from:
# "Reinforcement Learning (ML-4350)", Prof. Georg Martius. University of Tübingen, Winter Semester 2024-25
# Homework assignment 7: Gym DQN
# Source: ILIAS platform (restricted access)

class DuelingAgent(object):
    def __init__(self, observation_space, action_space, config):
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n

        # Parameters
        self._config = config.copy()
        self._eps = self._config['eps']
        set_seed(self._config["seed"])

        # Memory buffer
        self.buffer = Memory(
        max_size=self._config["buffer_size"],
        eps_td=self._config["eps_td"],
        alpha=self._config["alpha"],
        beta=self._config["beta"],
        beta_increment_per_sampling=self._config["beta_increment_per_sampling"],
        max_beta=self._config["max_beta"],
        min_priority=self._config["min_priority"],
        device=device,
        seed=self._config["seed"]
        )

        # Create Q-networks and target network
        self.Q = DuelingQFunction(
            observation_dim=observation_space.shape[0],
            action_dim=action_space.n,
            learning_rate=self._config["learning_rate"],
            device=device
        )
        self.target_Q = DuelingQFunction(
            observation_dim=observation_space.shape[0],
            action_dim=action_space.n,
            learning_rate=self._config["learning_rate"],
            device=device
        )

        # Intrinsic Curiosity Module (if enabled)
        self.icm = None
        if self._config.get("use_icm", False):
            obs_dim = observation_space.shape[0]
            act_dim = action_space.n  # Acciones discretas
            self.icm = IntrinsicCuriosityModule(
                obs_dim=obs_dim,
                action_dim=act_dim,
                feature_dim=self._config["icm_feature_dim"],
                icm_lr=self._config["icm_lr"],
                beta=self._config["icm_beta"],
                device=device
            )    
        self._update_target_net()
        self.train_iter = 0

    # Copy weights from Q to target Q
    def _update_target_net(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    # Action selection epsilon-greedy
    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # for TCML
        if not torch.is_tensor(observation):
            observation = torch.tensor(observation, dtype=torch.float32).to(device)
        else:
            observation = observation.to(device)
        observation = observation.unsqueeze(0)
        
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            action = self._action_space.sample()
        return action

    # Update epsilon until min
    def decay_epsilon(self):
        self._eps = max(self._config["eps_min"], self._eps * self._config["eps_decay"])

    # Buffer add
    def store_transition(self, transition):
        state, action, reward, next_state, done = transition

        # Intrinsic Curiosity Module
        if self.icm is not None:
            s_t = state.clone().detach().float().unsqueeze(0).to(device)
            ns_t = next_state.clone().detach().float().unsqueeze(0).to(device)
            a_t = torch.tensor([action], dtype=torch.long, device=device)

            fwd_error = self.icm.calc_intrinsic_reward(s_t, ns_t, a_t)
            reward += self._config["icm_scale"] * fwd_error.item()

        # recompensa modificada con ICM si está activo
        transition = (state, action, reward, next_state, done)

        self.buffer.add_transition(transition)

    # Train the Q-network using buffer samples
    def train(self, iter_fit=32):
        losses = []
        batch_size = self._config["batch_size"]
        discount = self._config["discount"]

        for _ in range(iter_fit):
            states_t, actions_t, rewards_t, next_states_t, dones_t, is_weights_t, tree_idxs = self.buffer.sample(batch_size)

            states_t = states_t.to(device)
            actions_t = actions_t.to(device)
            rewards_t = rewards_t.to(device)
            next_states_t = next_states_t.to(device)
            dones_t = dones_t.to(device)
            is_weights_t = is_weights_t.to(device)

            # Calcutale best action for next state using current Q network
            best_actions = self.Q.greedyAction(next_states_t)
            next_q_values = self.target_Q.Q_value(next_states_t, best_actions)
            next_q_values[dones_t.bool()] = 0.0  # Ignore if done

            targets_tensor = rewards_t + discount * next_q_values
            targets_tensor = targets_tensor.to(device)

            # Update Q-network
            loss_value, td_errors = self.Q.fit(
                states_t,
                actions_t,
                targets_tensor,
                weights=is_weights_t
            )
            losses.append(loss_value)

            # update priorities in buffer
            self.buffer.update_priorities(tree_idxs, td_errors)

            # update target
            self.train_iter += 1
            if self.train_iter % self._config["target_update_freq"] == 0:
                self._update_target_net()

            if self.icm is not None:
                icm_loss = self.icm.update(states_t, next_states_t, actions_t)
        
        return losses
