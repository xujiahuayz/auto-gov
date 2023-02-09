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
        learning_rate: float = 0.05,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.999,
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

        self.fc1 = nn.Linear(state_size, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, state: torch.Tensor) -> Number:
        # choose action to take
        if torch.rand(1).item() <= self.epsilon:
            return torch.randint(self.action_size, (1,))
        # state = torch.Tensor(state, dtype=torch.float32).unsqueeze(0)
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.forward(state)
        action = torch.argmax(q_values).item()
        return action

    def learn(
        self,
        state,
        action: int,
        reward,
        next_state: torch.Tensor,
        done: bool,
    ):
        # learn from experience
        # state = torch.Tensor(state, dtype=torch.float32).unsqueeze(0)
        state = torch.from_numpy(state).float().unsqueeze(0)

        q_values = self.forward(state)
        if done:
            target = reward
        else:
            # next_state = torch.Tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            next_q_values = self.forward(next_state)
            target = reward + self.gamma * torch.max(next_q_values).item()

        q_values[0][action] = target
        target = torch.tensor([target], dtype=torch.float32)
        loss = F.mse_loss(input=q_values, target=target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > 0.05:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    # initialize market and environment
    market = TestMarket()
    env = ProtocolEnv(market)

    # initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    num_episodes = 10000
    batch_size = 128

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode {episode} finished with reward {total_reward}")
