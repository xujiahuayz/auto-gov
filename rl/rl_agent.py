import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import gym
import numpy as np
import rl_env as env


class DQNAgent(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=0.05,
        epsilon_decay=0.999,
    ):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, state):
        if torch.rand(1).item() <= self.epsilon:
            return torch.randint(self.action_size, (1,))
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.forward(state)
        action = torch.argmax(q_values).item()
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.forward(state)
        if done:
            target = reward
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_q_values = self.forward(next_state)
            target = reward + self.gamma * torch.max(next_q_values).item()
        q_values[0][action] = target
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > 0.05:
            self.epsilon *= self.epsilon_decay


state_size = 5
action_size = 3
agent = DQNAgent(state_size, action_size)

num_episodes = 1000
batch_size = 64

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done
