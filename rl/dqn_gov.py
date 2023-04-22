import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

        # weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # if there is a GPU, use it, otherwise use CPU
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
        eps_dec_decrease_with_target: float = 0.2,
        layer1_size: int = 256,
        layer2_size: int = 256,
        target_on_point: int | None = None,
        target_update: int = 100,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
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

        self.target_on_point = target_on_point

        # eps_dec_decrease_with_target is used to decrease the eps_dec with the target switch on
        self.eps_dec_decrease_with_target = eps_dec_decrease_with_target
        # when eps_dec_check_flag is True, we need to check whether we need to decrease eps_dec
        self.eps_dec_check_flag = self.target_on_point

        if self.target_on_point:
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

    @property
    def target_net_enabled(self) -> bool:
        if self.target_on_point:
            return (
                self.epsilon
                < self.epsilon_start
                - (self.epsilon_start - self.eps_min) * self.target_on_point
            )
        return False

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

    def choose_action(self, observation, evaluate=False) -> int:
        if np.random.random() > self.epsilon or evaluate:
            state = T.tensor(np.array([observation])).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = int(T.argmax(actions).item())
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self) -> None:
        # if there is not enough memory, do not learn
        if self.mem_cntr < self.batch_size:
            return

        # set the gradients of all the model parameters (weights and biases) in the Q_eval network to zero.
        self.Q_eval.optimizer.zero_grad()

        # sample a batch of transitions
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # create an index for each element of the current batch
        batch_index = np.arange(self.batch_size, dtype=np.int32)

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

        # calculate the loss
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.loss_list.append(loss.item())
        self.Q_eval.optimizer.step()

        if self.target_on_point:
            self.update_counter += 1
            if self.update_counter % self.target_update == 0:
                self.Q_target.load_state_dict(self.Q_eval.state_dict())

        # check if epsilon decay should be decreased
        if self.eps_dec_check_flag:
            if self.target_net_enabled:
                self.eps_dec = self.eps_dec * self.eps_dec_decrease_with_target
                self.eps_dec_check_flag = False

        # update epsilon
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )


def save_trained_model(
    agent: Agent, model_dir: str, model_name: str = "trained_model.pt"
) -> None:
    """
    Save the trained model.

    Args:
        agent (Agent): The trained DQN agent.
        model_dir (str): The directory to save the model in.
        model_name (str, optional): The filename for the saved model. Defaults to "trained_model.pt".
    """
    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the Q-network's state_dict
    model_path = os.path.join(model_dir, model_name)
    T.save(agent.Q_eval.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")


def load_trained_model(agent: Agent, model_path: str) -> None:
    """
    Load the trained model.

    Args:
        agent (Agent): The DQN agent to load the model into.
        model_path (str): The path to the saved model.
    """
    agent.Q_eval.load_state_dict(T.load(model_path))
    agent.Q_eval.eval()
    print(f"Trained model loaded from {model_path}")


def contain_nan(model_state_dict):
    """
    Check if the model state dict contains NaN values.
    """
    for key in model_state_dict:
        if T.isnan(model_state_dict[key]).any():
            return True
    return False
    