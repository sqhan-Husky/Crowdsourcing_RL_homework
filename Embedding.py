import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_states, 50),
            nn.ReLU()
        )
        self.out = nn.Linear(50, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        actions_value = self.out(x)
        return actions_value