import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPDiscretePolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(MLPDiscretePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1) # for stochastic action

class DiscreteMLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DiscreteMLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)