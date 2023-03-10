import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DQN(nn.Module):
    # agent for DQN

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        # initialize agent
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size: int,
        n_actions: int,
        max_mem_size: int = 100_000,
        eps_end: float = 0.05,
        eps_dec: float = 5e-5,
        layer1_size: int = 256,
        layer2_size: int = 256,
        target_net_enabled: bool = False,
        target_update: int = 100,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.input_dims = input_dims
        self.lr = lr
        # self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DQN(
            self.lr,
            n_actions=n_actions,
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
        )

        self.target_net_enabled = target_net_enabled
        if self.target_net_enabled:
            self.Q_target = DQN(
                self.lr,
                n_actions=n_actions,
                input_dims=input_dims,
                fc1_dims=layer1_size,
                fc2_dims=layer2_size,
            )

            self.Q_target.load_state_dict(self.Q_eval.state_dict())
            self.Q_target.eval()
            self.target_update = target_update
            self.update_counter = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.loss_list = []

    def store_transition(self, state, action, reward, state_, done: bool) -> None:
        index = self.mem_cntr % self.mem_size

        # ===============================
        # for i in state:
        #     print(i)
        # print("=====")
        # print(np.array(state).shape)
        # ===============================
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation) -> int:
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation])).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = int(T.argmax(actions).item())
        else:
            action = np.random.choice(self.action_space)

        # print("=========================")
        # print(observation[2])
        # print(action)
        # print("=========================")

        # # add constraint
        # # If the current collateral factor is less than 0, only allow keep or raise actions
        # if observation[2] <= 0:
        #     if action == 0:
        #         action = 1
        # # If the current collateral factor is more than 1, only allow keep or lower actions
        # elif observation[2] >= 1:
        #     if action == 2:
        #         action = 1

        return action

    def learn(self) -> None:
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # # print("=====")
        # print(self.state_memory[batch])
        # print(type(self.state_memory[batch]))
        # # print("=====")

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)

        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        # if self.target_net_enabled, Double DQN with target network enabled
        if self.target_net_enabled:
            q_next = self.Q_target.forward(new_state_batch)
        else:
            q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.loss_list.append(loss.item())
        self.Q_eval.optimizer.step()

        if self.target_net_enabled:
            self.update_counter += 1
            if self.update_counter % self.target_update == 0:
                self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )
