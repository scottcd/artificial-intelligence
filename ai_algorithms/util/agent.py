import torch
import random
import math
import gymnasium as gym
import numpy as np


class Agent():
    def __init__(self, env) -> None:
        self.env = env
        if (isinstance(env.action_space, gym.spaces.Discrete)):
            self.n_actions = env.action_space.n
        else:
            self.n_actions = env.action_space.shape[0]
        if (isinstance(env.observation_space, gym.spaces.Discrete)):
            self.n_observations = env.observation_space.n
        else:
            self.n_observations = env.observation_space.shape[0]

        self.memory = None
        self.steps_taken = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, model, state, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=1000) -> torch.Tensor:
        sample = random.random()

        eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
            math.exp(-1. * self.steps_taken / epsilon_decay)

        self.steps_taken += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = state.clone().detach()
                if (isinstance(self.env.action_space, gym.spaces.Discrete)):
                    return model(state_tensor).max(1)[1].unsqueeze(0).item()
                else:
                    return model(state_tensor).squeeze().tolist()
                    
        else:
            if (isinstance(self.env.action_space, gym.spaces.Discrete)):
                return torch.tensor([[random.randint(0, self.n_actions-1)]], device=self.device, dtype=torch.long).item()
            else:
                action_values = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high)
                action_tensor = torch.tensor(action_values, device=self.device, dtype=torch.float32).squeeze().tolist()
                return action_tensor


    def learn(self) -> None:
        pass

    def remember(self) -> None:
        pass
