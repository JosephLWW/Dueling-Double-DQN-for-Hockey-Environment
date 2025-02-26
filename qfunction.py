import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DuelingQFunction(nn.Module):
    def __init__(self, observation_dim, action_dim, learning_rate, device=None):
        super(DuelingQFunction, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Red de extracción / Feature extraction net
        self.feature_network = nn.Sequential(
            nn.Linear(self.observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Corriente de valor/Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Corriente de ventaja/Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

        self.to(self.device)

        # Optimizador y función de pérdida/ Optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-6) # Don't divide by zero
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.feature_network(state)
        state_value = self.value_stream(features)
        action_advantages = self.advantage_stream(features)
        q_values = state_value + (action_advantages - torch.mean(action_advantages, dim=1, keepdim=True))
        return q_values

    def Q_value(self, observations, actions):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float().to(self.device)
        elif observations.device != self.device:
            observations = observations.to(self.device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).long().to(self.device)
        elif actions.device != self.device:
            actions = actions.to(self.device)
        actions = actions.to(torch.int64)
        q_values = self.forward(observations)
        q_values = torch.gather(q_values, 1, actions.unsqueeze(-1)).squeeze(-1)
        return q_values

    def maxQ(self, observations):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float().to(self.device)
        elif observations.device != self.device:
            observations = observations.to(self.device)
        q_values = self.forward(observations)
        max_q_values, _ = torch.max(q_values, dim=1)
        return max_q_values.detach().cpu().numpy()

    def greedyAction(self, observations):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float().to(self.device)
        elif observations.device != self.device:
            observations = observations.to(self.device)
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        q_values = self.forward(observations)
        greedy_actions = torch.argmax(q_values, dim=1).detach().cpu().numpy()
        if greedy_actions.size == 1:
            return greedy_actions.item()
        return greedy_actions

    def fit(self, observations, actions, targets, weights=None):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float().to(self.device)
        elif observations.device != self.device:
            observations = observations.to(self.device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).long().to(self.device)
        elif actions.device != self.device:
            actions = actions.to(self.device)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float().to(self.device)
        elif targets.device != self.device:
            targets = targets.to(self.device)
        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights).float().to(self.device)
            elif weights.device != self.device:
                weights = weights.to(self.device)

        q_values = self.Q_value(observations, actions)
        loss_per_sample = self.loss_fn(q_values, targets)

        if weights is not None:
            loss = (loss_per_sample * weights).mean()
        else:
            loss = loss_per_sample.mean()

        # Retropropagación y optimización/ Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors = (targets.detach() - q_values.detach()).cpu().numpy()
        return loss.item(), td_errors
