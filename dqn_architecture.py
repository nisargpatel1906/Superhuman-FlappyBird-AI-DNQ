import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network: a simple fully-connected neural network.
    Input  : observation (state) from the environment
    Output : Q-value for every possible action
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)
