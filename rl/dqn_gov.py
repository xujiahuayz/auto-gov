import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl.data_structure import SumTree


class DQN(nn.Module):
    """Agent for DQN"""

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        """Initialize agent."""
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions

        # define layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)

        # weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

        # use Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # # use MAE loss (L1 loss) function
        # self.loss = nn.L1Loss()

        # use MSE loss (L2 loss) function
        self.loss = nn.MSELoss()

        # if there is a GPU, use it, otherwise use CPU
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # # restrict the number of CPU threads used to run the model
        # T.set_num_threads(3)
        self.to(self.device)

    def forward(self, state):
        # # use LeakyReLU as activation function
        # layer1 = F.leaky_relu(self.fc1(state))
        # layer2 = F.leaky_relu(self.fc2(layer1))
        # actions = self.fc3(layer2)

        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = F.relu(self.fc3(layer2))
        actions = self.fc4(layer3)

        return actions

class PrioritizedReplayBuffer:
    # replay buffer for DQN
    def __init__(self, max_size: int, input_shape: tuple, n_actions: int, alpha: float = 0.6):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha
        self.sumtree = SumTree(max_size)

        self.input_shape = input_shape
        self.n_actions = n_actions
    
    def store_transition(self, state, action, reward: float, next_state, done: bool):
        # calculate priority
        # priority = np.max(self.sumtree.tree[-self.sumtree.capacity:])
        priority = self.sumtree.total() / self.sumtree.capacity
        if priority == 0:
            priority = 1
        data = (state, action, reward, next_state, done)
        self.sumtree.add(priority, data)
        self.mem_cntr += 1
    
    def sample(self, batch_size: int, beta):
        idxs, experiences, priorities = [], [], []
        segment_length = self.sumtree.total() / batch_size

        for i in range(batch_size):
            a = segment_length * i
            b = segment_length * (i + 1)
            # a could be equal to b, check first
            if a == b:
                s = a
            s = np.random.uniform(a, b)
            idx, priority, data = self.sumtree.get(s)
            idxs.append(idx)
            experiences.append(data)
            priorities.append(priority)
        
        max_priority = max(priorities)
        scaling_factor = np.array(priorities) / max_priority
        is_weights = (self.mem_cntr * scaling_factor) ** (-beta)
        is_weights /= is_weights.max()

        return idxs, experiences, is_weights
    
    def update_priority(self, idx, td_error):
        priority = td_error + 1e-5  # Small constant to ensure nonzero priority
        priority = priority ** self.alpha
        self.sumtree.update(idx, priority)

    def __len__(self):
        return self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size

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
        layer3_size: int = 256,
        target_on_point: int | None = None,
        target_update: int = 100,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 1e-4,
        PrioritizedReplay_switch: bool = False,
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

        # initialize the Q_eval network
        self.Q_eval = DQN(
            self.lr,
            n_actions=n_actions,
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            fc3_dims=layer3_size,
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
                fc3_dims=layer3_size,
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

        self.PrioritizedReplay_switch = PrioritizedReplay_switch
        if PrioritizedReplay_switch:
            self.beta = beta
            self.beta_increment_per_sampling = beta_increment_per_sampling
            self.buffer = PrioritizedReplayBuffer(max_mem_size, input_dims, n_actions, alpha=alpha)

    @property
    def target_net_enabled(self) -> bool:
        # eps_start -----eps_dec-----> target_on_point -----eps_dec*eps_dec_decrease_with_target-----> eps_min
        #                   ^                                                 ^
        #                   |                                                 |
        #                   |                                                 |
        #        length: eps_start - target_on_point              length: target_on_point - eps_min

        if self.target_on_point:
            return self.epsilon < self.target_on_point
        return False

    def store_transition(self, state, action, reward, state_, done: bool) -> None:
        if self.PrioritizedReplay_switch == False:
            # when prioritized replay is off
            # calculate the index of the memory
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
        else:
            self.buffer.store_transition(state, action, reward, state_, done)

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
        if self.PrioritizedReplay_switch == False:
            # when prioritized replay is off
            if self.mem_cntr < self.batch_size:
                return
        else:
            # when prioritized replay is on
            if self.buffer.mem_cntr < self.batch_size:
                return

        """
        set the gradients of all the model parameters (weights and biases) in the Q_eval network to zero.
        should generally be called before calculating the loss,
        specifically at the beginning of each training iteration, and before the forward pass.

        The typical order of operations within a training loop is as follows:
        1. Zero Gradients: Reset gradients to zero before any forward pass or loss calculation.
            This is done to ensure that gradients from the previous iteration or batch do not affect the current batch's gradients.
        2. Forward Pass: Compute predictions (forward pass) using the current model weights.
        3. Loss Calculation: Calculate the loss based on the predictions and the target values.
        4. Backpropagation: Perform backpropagation to compute gradients with respect to the loss.
        5. Gradient Update: Update the model weights using an optimization algorithm (e.g., SGD, Adam) based on the computed gradients.
        """
        self.Q_eval.optimizer.zero_grad()

        if self.PrioritizedReplay_switch == False:
            # when prioritized replay is off
            # sample a batch of transitions
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)

            # create an index for each element of the current batch (from 0 to self.batch_size - 1)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)

            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

            action_batch = self.action_memory[batch]

            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        
        else:
            # when prioritized replay is on
            idxs, experiences, is_weights = self.buffer.sample(self.batch_size, self.beta)
            is_weights = T.tensor(is_weights).to(self.Q_eval.device)

            states, actions, rewards, next_states, dones = zip(*experiences)
            print("states: ", states)

            states = T.tensor(states).to(self.Q_eval.device)
            next_states = T.tensor(next_states).to(self.Q_eval.device)
            rewards = T.tensor(rewards).to(self.Q_eval.device)
            actions = T.tensor(actions).to(self.Q_eval.device)
            dones = T.tensor(dones).to(self.Q_eval.device)
            # make sure the variables are with the same names
            terminal_batch = dones
            new_state_batch = next_states
            reward_batch = rewards

            q_eval = self.Q_eval(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            # q_eval = self.Q_eval.forward(states)[T.arange(self.batch_size), actions]

        # if self.target_net_enabled, Double DQN with target network enabled
        if self.target_net_enabled:
            # try delete the detach() function if it does not work
            q_next = self.Q_target.forward(new_state_batch).detach()
        else:
            # try delete the detach() function if it does not work
            q_next = self.Q_eval.forward(new_state_batch).detach()
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # calculate the loss
        if self.PrioritizedReplay_switch == False:
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        else:
            td_errors = F.mse_loss(q_target, q_eval, reduction="none")
            weighted_td_errors = is_weights * td_errors
            loss = T.mean(weighted_td_errors)
        # if loss is inf, print the target and prediction
        if loss == float("inf"):
            print("!!! loss is inf !!!")
            print("target: ", q_target)
            print("prediction: ", q_eval)
        loss.backward()
        # # clip the gradients to avoid exploding gradients
        # nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=1.0)
        self.loss_list.append(loss.item())
        self.Q_eval.optimizer.step()

        if self.PrioritizedReplay_switch == True:
            # when prioritized replay is on
            self.beta += self.beta_increment_per_sampling
            self.beta = min(self.beta, 1.0)

            # update the priorities
            for idx, td_error in zip(idxs, td_errors.detach().numpy()):
                self.buffer.update_priority(idx, td_error)

        if self.target_on_point:
            self.update_counter += 1
            if self.update_counter % self.target_update == 0:
                self.Q_target.load_state_dict(self.Q_eval.state_dict())

        # check if epsilon decay should be decreased
        if self.eps_dec_check_flag:
            if self.target_net_enabled:
                # decrease epsilon decay
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
