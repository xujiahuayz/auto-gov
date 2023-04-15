import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.types import Number
import numpy as np
import test_market as tm
from test_market import TestMarket
from rl_env import ProtocolEnv


class DQNAgent(nn.Module):
    # agent for DQN

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.01,
        gamma: float = 0.9,
        epsilon: float = 0.5,
        epsilon_decay: float = 1 / 400,
    ):
        # initialize agent
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # 0 0 0
        # - - - - - - - - - -
        # 0 0 0

        self.fc1 = nn.Linear(state_size, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def act(self, state: torch.Tensor) -> Number:
        # choose action to take
        state = torch.from_numpy(state).float().unsqueeze(0)
        # print(state)
        if torch.rand(1).item() <= self.epsilon:
            action = torch.randint(self.action_size, (1,))
        else:
            # state = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.forward(state)
            action = torch.argmax(q_values).item()

        # If the current collateral factor is less than 0, only allow keep or raise actions
        if state[0][2] <= 0:
            if action == 0:
                action = 1
        # If the current collateral factor is more than 1, only allow keep or lower actions
        elif state[0][2] >= 1:
            if action == 2:
                action = 1

        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        q_values = self.forward(state)
        q_values_next = self.forward(next_state)
        if done:
            q_values[0][action] = reward
        else:
            q_values[0][action] = reward + self.gamma * torch.max(q_values_next)
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_values, q_values)
        loss.backward()
        self.optimizer.step()
        if self.epsilon > 0.05:
            self.epsilon -= self.epsilon_decay


if __name__ == "__main__":
    # initialize market and environment
    market = TestMarket()
    _env = ProtocolEnv(market)

    # initialize agent
    state_size = _env.observation_space.shape[0]
    action_size = _env.action_space.n
    agent = DQNAgent(state_size, action_size)

    num_episodes = 10000
    batch_size = 128

    for episode in range(num_episodes):
        state = _env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = _env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode {episode} finished with reward {total_reward}")
