from ..util.agent import Agent
from .transition import Transition
from .dqn import DQN
from .memory import ReplayMemory
import torch
import torch.optim as optim
import torch.nn as nn
import gymnasium as gym


class DQNAgent(Agent):
    def __init__(self, env, batch_size=128, tau=0.005, gamma=0.99, learning_rate=0.1) -> None:
        super().__init__(env)
        self.name = 'dqn'
        self.memory = ReplayMemory()
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.policy_network = DQN(self.n_actions, self.n_observations)
        self.target_network = DQN(self.n_actions, self.n_observations)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = optim.AdamW(
            self.policy_network.parameters(), lr=self.learning_rate, amsgrad=True)

    def remember(self, state, next_state, reward, action) -> None:
        self.memory.push(state, action, next_state, reward)

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            self.update_target_network()
            return
        
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #
        if (isinstance(self.env.action_space, gym.spaces.Discrete)):
            state_action_values = self.policy_network(
                state_batch).gather(1, action_batch.unsqueeze(1))

        else:
            state_action_values = self.policy_network(state_batch)

        #
        next_state_values = torch.zeros(self.batch_size, device=self.device, )

        #
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(
                non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))
        print(state_action_values.shape)
        print(expected_state_action_values.shape)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

        # update target network
        self.update_target_network()

    def update_target_network(self) -> None:
        target_network_state_dict = self.target_network.state_dict()
        policy_network_state_dict = self.policy_network.state_dict()
        for key in policy_network_state_dict:
            target_network_state_dict[key] = policy_network_state_dict[key] * \
                self.tau + target_network_state_dict[key]*(1-self.tau)
        self.target_network.load_state_dict(target_network_state_dict)