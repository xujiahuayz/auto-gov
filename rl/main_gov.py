import logging
from os import path
import numpy as np
import matplotlib.pyplot as plt
from market_env.constants import FIGURES_PATH

from market_env.env import DefiEnv, PlfPool, PriceDict, User
from rl.dqn_gov import Agent
from rl.rl_env import ProtocolEnv
from rl.utils import plot_learning_curve
import torch

# show logging level at info
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # initialize environment
    defi_env = DefiEnv(prices=PriceDict({"tkn": 1}))
    Alice = User(name="alice", env=defi_env, funds_available={"tkn": 1_000_000})
    plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        initial_starting_funds=800_000,
        asset_name="tkn",
        collateral_factor=0.8,
    )

    env = ProtocolEnv(defi_env)

    # initialize agent
    agent = Agent(
        gamma=0.99,
        epsilon=1,
        batch_size=128,
        n_actions=env.action_space.n,
        eps_end=0.01,
        input_dims=env.observation_space.shape,
        lr=0.003,
        # eps_dec=0,
        eps_dec=5e-5,
    )
    # agent = Agent(state_size, action_size)

    scores, eps_history = [], []
    n_games = 500

    collateral_factors = []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation.astype(np.float32))
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-30:])
        if i % 10 == 0:
            logging.info(
                f"logging.debug {i} score {score:.2f} average score {avg_score:.2f} epsilon {agent.epsilon:.2f} collateral factor {next(iter(defi_env.plf_pools.values())).collateral_factor:.2f}"
            )
        collateral_factors.append(
            next(iter(defi_env.plf_pools.values())).collateral_factor
        )

    # torch.save(agent.q_eval.state_dict(), "models/dqn_gov.pth")

    x = [i + 1 for i in range(n_games)]
    filename = path.join(FIGURES_PATH, "defi.png")
    plot_learning_curve(x, scores, eps_history, filename)
    plt.clf()
    plt.plot(x, collateral_factors)
    plt.savefig(path.join(FIGURES_PATH, "collateral_factors.png"))
