import logging
import time
from typing import Any

import numpy as np

from market_env.caching import cache
from rl.dqn_gov import Agent
from rl.rl_env import ProtocolEnv
from rl.utils import init_env
from rl.dqn_gov import save_trained_model


def bench_env(**kwargs) -> tuple[list[float], list[dict[str, Any]]]:
    defi_env = init_env(**kwargs)
    env = ProtocolEnv(defi_env)
    state_this_game = [defi_env.state_summary]
    score = 0
    done = False
    rewards = [0.0]
    defi_env.reset()

    while not done:
        # get states for plotting
        # never change collateral factor
        _, reward, done, _ = env.step(0)
        score += reward
        state_this_game.append(defi_env.state_summary)
        rewards.append(reward)

    return rewards, state_this_game


# @cache(ttl=60 * 60 * 24 * 7, min_memory_time=0.00001, min_disk_time=0.1)
def train_env(
    gamma: float = 0.99,
    n_games: int = 2_000,
    epsilon: float = 1,
    eps_end: float = 0.01,
    eps_dec: float = 5e-5,
    batch_size: int = 128,
    lr: float = 0.003,
    target_net_enabled: bool = False,
    compared_to_benchmark: bool = True,
    **kwargs,
) -> tuple[
    list[float],
    list[float],
    list[list[dict]],
    list[float],
    list[float],
    list[dict[str, Any]],
]:
    # initialize environment
    defi_env = init_env(**kwargs)
    env = ProtocolEnv(defi_env)

    bench_rewards, bench_states = bench_env(**kwargs)

    # initialize agent
    agent = Agent(
        gamma=gamma,
        epsilon=epsilon,
        batch_size=batch_size,
        n_actions=env.action_space.n,
        eps_end=eps_end,
        input_dims=env.observation_space.shape,
        lr=lr,
        eps_dec=eps_dec,
        target_net_enabled=target_net_enabled,
    )

    scores, eps_history, time_cost, states = [], [], [], []

    for i in range(n_games):
        state_this_game = []
        score = 0
        done = False
        observation = env.reset()
        start_time = time.time()
        state_this_game.append(defi_env.state_summary)
        while not done:
            # get states for plotting
            action = agent.choose_action(observation.astype(np.float32))
            # this checks done or not

            observation_, reward, done, _ = env.step(action)
            score += reward - (
                bench_rewards[env.defi_env.step] if compared_to_benchmark else 0
            )
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            state_this_game.append(defi_env.state_summary)
        time_cost.append(time.time() - start_time)
        scores.append(score)
        eps_history.append(agent.epsilon)
        states.append(state_this_game)

        avg_score = np.mean(scores[-30:])
        if i % 50 == 0:
            logging.info(
                "episode: {}, score: {:.2f}, average score: {:.2f}, epsilon: {:.2f}".format(
                    i,
                    score,
                    avg_score,
                    agent.epsilon,
                )
            )
        
        # save the trained model
        model_name = "trained_model.pt" # change to your own name
        model_dir = "models" # change to your own directory
        save_trained_model(agent, model_name, model_dir)

    return scores, eps_history, states, time_cost, bench_rewards, bench_states

if __name__ == "__main__":
    # show logging level at info
    logging.basicConfig(level=logging.INFO)
    N_GAMES = 20

    (
        training_scores,
        training_eps_history,
        training_states,
        training_time_cost,
        training_bench_rewards,
        training_bench_states,
    ) = train_env(
        n_games=N_GAMES,
        eps_dec=0,
        lr=0.05,
        initial_collateral_factor=0.75,
        max_steps=30,
        target_net_enabled=True,
        compared_to_benchmark=True,
    )
