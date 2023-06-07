import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_actions, n_observations, n_hidden_layers=1, n_neurons=128):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_observations, n_neurons))  # input layer

        for _ in range(n_hidden_layers):
            self.layers.append(nn.Linear(n_neurons, n_neurons))

        self.layers.append(nn.Linear(n_neurons, n_actions))  # output layer

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
        return self.layers[-1](x)  # output layer
