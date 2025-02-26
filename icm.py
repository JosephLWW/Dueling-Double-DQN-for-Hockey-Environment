import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, obs_dim, action_dim, feature_dim, icm_lr, beta, device=None):
        super(IntrinsicCuriosityModule, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta # Forward / Inverse ratio

        # Feature extractor: phi(s)
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )

        # Inverse network
        self.inverse = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Forward network with discrete actions
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=icm_lr)
        self.to(self.device)

    # Extracts the internal representation phi(s)
    def forward_feature(self, state):
        return self.feature(state)

    # logits of the inverse model and prediction of phi(s_{t+1}) of the forward model
    def forward_icm(self, phi_s, phi_next_s, action_onehot):
        # Inverse model
        inv_in = torch.cat([phi_s, phi_next_s], dim=1)
        pred_action_logits = self.inverse(inv_in)

        # Forward model
        fwd_in = torch.cat([phi_s, action_onehot], dim=1)
        pred_phi_next = self.forward_model(fwd_in)

        return pred_action_logits, pred_phi_next

    # Calculates the intrinsic reward for a batch of transitions
    def calc_intrinsic_reward(self, state, next_state, action):
        with torch.no_grad():
            phi_s = self.forward_feature(state)
            phi_next_s = self.forward_feature(next_state)
            action_onehot = self._one_hot(action, pred_dim=self.inverse[-1].out_features)  # discrete one hot: out_features = action_dim

            # for the reward
            _, pred_phi_next = self.forward_icm(phi_s, phi_next_s, action_onehot)
            fwd_error = 0.5 * torch.sum((pred_phi_next - phi_next_s)**2, dim=1)  # MSE from forward model
        return fwd_error

    # Backward pass of the ICM
    def update(self, state, next_state, action):
        phi_s = self.forward_feature(state)
        phi_next_s = self.forward_feature(next_state)
        action_onehot = self._one_hot(action, pred_dim=self.inverse[-1].out_features)

        # Forward + Inverse
        pred_action_logits, pred_phi_next = self.forward_icm(phi_s, phi_next_s, action_onehot)

        # Inverse loss: cross-entropy (discrete actions)
        inverse_loss = F.cross_entropy(pred_action_logits, action.long())

        # Forward loss: MSE (continuous states)
        forward_loss = 0.5 * torch.mean((pred_phi_next - phi_next_s)**2)

        # Combine
        icm_loss = (1 - self.beta)*inverse_loss + self.beta*forward_loss

        self.optimizer.zero_grad()
        icm_loss.backward()
        self.optimizer.step()

        return icm_loss.item()

    # One-hot encoding for discrete actions
    def _one_hot(self, action, pred_dim):
        batch_size = action.shape[0]
        one_hot = torch.zeros(batch_size, pred_dim, device=self.device)
        one_hot[torch.arange(batch_size), action.long()] = 1.0
        return one_hot
