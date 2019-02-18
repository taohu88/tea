import random
import torch
import torch.nn as nn


class CartPoloDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(CartPoloDQN, self).__init__()
        self.num_actions = num_actions

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon, device=None):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action


class CartPoloDuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CartPoloDuelingDQN, self).__init__()
        self.num_actions = num_outputs

        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, epsilon, device=None):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_value = self.forward(state).detach()
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action
