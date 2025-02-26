import numpy as np
import torch
import random
from config import config, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Binary SumTree for better sampling and storing priorities
class SumTree:
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree = np.zeros(2 * max_size - 1, dtype=np.float32)        # El árbol tiene 2*max_size-1 posiciones: hojas + nodos internos
        self.data = np.zeros(max_size, dtype=np.int32)
        self.size = 0

    @property
    def total_priority(self):
        # Total priority of the tree (root node)
        return self.tree[0]

    # Add a new priority to the tree in the leaf of the buffer_idx
    def add(self, priority, data_idx, buffer_idx):
        leaf_idx = buffer_idx + (self.max_size - 1)
        self.data[buffer_idx] = data_idx
        self.update(leaf_idx, priority)
        self.size = min(self.size + 1, self.max_size)

    # Update the priority of a leaf and propagate the change upwards
    def update(self, leaf_idx, priority):
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        parent = (leaf_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    # Get the leaf index, priority and data index for a given value
    def get(self, value):
        idx = 0
        left = 1
        right = 2

        while left < len(self.tree):
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
            left = 2 * idx + 1
            right = left + 1

        leaf_idx = idx
        data_idx = self.data[leaf_idx - (self.max_size - 1)]
        return leaf_idx, self.tree[leaf_idx], data_idx


# Memory buffer with Prioritized Experience Replay
class Memory:
    def __init__(self, max_size, eps_td, alpha, beta, beta_increment_per_sampling,max_beta, min_priority, device=device, seed=None):
        self.max_size = max_size
        self.device = device
        set_seed(seed)

        # Parámetros de PER
        self.eps_td = eps_td
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.max_beta = max_beta
        self.min_priority = min_priority
        self.max_priority = 1.0
        self.sum_tree = SumTree(self.max_size)

        # Buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        # Indices
        self.current_idx = 0
        self.size = 0

    # Add a new transition to the memory buffer
    def add_transition(self, transition):
        state, action, reward, next_state, done = transition

        if self.size < self.max_size:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
        else:
            # Overwrites old data
            self.states[self.current_idx] = state
            self.actions[self.current_idx] = action
            self.rewards[self.current_idx] = reward
            self.next_states[self.current_idx] = next_state
            self.dones[self.current_idx] = done

        # Add/update priority in the SumTree
        if self.size < self.max_size:
            self.sum_tree.add(self.max_priority, self.current_idx, self.current_idx)
        else:
            leaf_idx = self.current_idx + (self.max_size - 1)
            self.sum_tree.update(leaf_idx, self.max_priority)

        self.current_idx = (self.current_idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # Sample a batch of transitions with Prioritized Experience Replay
    def sample(self, batch_size):
        actual_batch_size = min(batch_size, self.size)
        self.beta = min(self.max_beta, self.beta + self.beta_increment_per_sampling)

        batch_indices = []
        batch_priorities = np.zeros((actual_batch_size, 1), dtype=np.float32)
        tree_idxs = np.zeros(actual_batch_size, dtype=np.int32)

        total_p = self.sum_tree.total_priority
        total_p = max(total_p, 1e-8)  # Evitar división por 0

        segment = total_p / actual_batch_size

        for i in range(actual_batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            leaf_idx, priority, data_idx = self.sum_tree.get(value)
            tree_idxs[i] = leaf_idx
            batch_priorities[i] = priority
            batch_indices.append(data_idx)

        # No 0 priorities
        batch_priorities[batch_priorities < 1e-8] = 1e-8

        # Convert to probabilities and then to IS weights
        sampling_probs = batch_priorities / total_p
        sampling_probs = np.maximum(sampling_probs, 1e-8)

        N = self.size
        is_weights = (N * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()  # Normalize

        # Numpy is cpu
        batch_states_np = np.stack([self.states[idx].cpu().numpy() for idx in batch_indices]).astype(np.float32)
        batch_actions_np = np.array([self.actions[idx] for idx in batch_indices], dtype=np.int64)
        batch_rewards_np = np.array([self.rewards[idx] for idx in batch_indices], dtype=np.float32)
        batch_next_states_np = np.stack([self.next_states[idx].cpu().numpy() for idx in batch_indices]).astype(np.float32)
        batch_dones_np = np.array([self.dones[idx] for idx in batch_indices], dtype=np.float32)
        batch_weights_np = is_weights.squeeze().astype(np.float32)

        # To tensor
        batch_states_t = torch.from_numpy(batch_states_np).to(self.device)
        batch_actions_t = torch.from_numpy(batch_actions_np).to(self.device)
        batch_rewards_t = torch.from_numpy(batch_rewards_np).to(self.device)
        batch_next_states_t = torch.from_numpy(batch_next_states_np).to(self.device)
        batch_dones_t = torch.from_numpy(batch_dones_np).to(self.device)
        batch_weights_t = torch.from_numpy(batch_weights_np).to(self.device)

        return (batch_states_t, batch_actions_t, batch_rewards_t,
                batch_next_states_t, batch_dones_t, batch_weights_t, tree_idxs)

    # Update priorities in the SumTree based on the TD error
    def update_priorities(self, tree_idxs, td_errors):
        for leaf_idx, td_error in zip(tree_idxs, td_errors):
            priority_raw = (abs(td_error) + self.eps_td) ** self.alpha
            priority = max(priority_raw, self.min_priority)  # Clamp
            self.sum_tree.update(leaf_idx, priority)
            if priority > self.max_priority:
                self.max_priority = priority
