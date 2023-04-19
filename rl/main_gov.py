import logging
import time
from typing import Any, Callable

import numpy as np

from market_env.caching import cache
from rl.dqn_gov import Agent
from rl.rl_env import ProtocolEnv
from rl.utils import init_env


def run_episode(
    env: ProtocolEnv,
    agent: Agent,
    compared_to_benchmark: bool,
    tkn_price_trend_this_game: np.ndarray,
    usdc_price_trend_this_game: np.ndarray,
    training: bool,
    **add_env_kwargs,
) -> tuple[float, list[float], list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    bench_rewards, bench_states_this_game = bench_env(
        tkn_price_trend_func=lambda t, s: tkn_price_trend_this_game,
        usdc_price_trend_func=lambda t, s: usdc_price_trend_this_game,
        **add_env_kwargs,
    )

    # extend bench_rewards with 0 to match the length of the game (max_steps)
    bench_rewards.extend([0.0] * (env.defi_env.max_steps + 1 - len(bench_rewards)))
    env.defi_env.plf_pools[
        "tkn"
    ].price_trend_func = lambda t, s: tkn_price_trend_this_game
    env.defi_env.plf_pools[
        "usdc"
    ].price_trend_func = lambda t, s: usdc_price_trend_this_game
    score = 0
    done = False
    policy = []
    reward_this_game = []
    observation = env.reset()
    state_this_game = [env.defi_env.state_summary]
    while not done:
        # get states for plotting
        action = agent.choose_action(
            observation.astype(np.float32), evaluate=not training
        )
        observation_, reward, done, _ = env.step(action)
        reward -= bench_rewards[env.defi_env.step] if compared_to_benchmark else 0

        if training:
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
        observation = observation_

        score += reward
        policy.append(action)
        reward_this_game.append(reward)
        state_this_game.append(env.defi_env.state_summary)

    return score, reward_this_game, policy, state_this_game, bench_states_this_game


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


@cache(ttl=60 * 60 * 24 * 7, min_memory_time=0.00001, min_disk_time=0.1)
def train_env(
    agent_args: dict[str, Any],
    n_games: int = 2_000,
    compared_to_benchmark: bool = True,
    tkn_price_trend_func: Callable[
        [int, int | None], np.ndarray
    ] = lambda x, y: np.ones(x),
    usdc_price_trend_func: Callable[
        [int, int | None], np.ndarray
    ] = lambda x, y: np.ones(x),
    tkn_seed: int | None = None,
    usdc_seed: int | None = None,
    **add_env_kwargs,
) -> tuple[
    list[float],
    list[float],
    list[list[dict[str, Any]]],
    list[list[float]],
    list[float],
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    # initialize environment
    defi_env = init_env(**add_env_kwargs)
    env = ProtocolEnv(defi_env)

    agent_args["n_actions"] = env.action_space.n
    agent_args["input_dims"] = env.observation_space.shape

    # initialize agent
    agent = Agent(
        **agent_args,
    )

    (
        scores,
        eps_history,
        time_cost,
        states,
        trained_model,
        bench_states,
        policies,
        rewards,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i in range(n_games):
        start_time = time.time()
        (
            score,
            reward_this_game,
            policy,
            state_this_game,
            bench_states_this_game,
        ) = run_episode(
            env=env,
            agent=agent,
            compared_to_benchmark=compared_to_benchmark,
            tkn_price_trend_this_game=tkn_price_trend_func(
                defi_env.max_steps, tkn_seed
            ),
            usdc_price_trend_this_game=usdc_price_trend_func(
                defi_env.max_steps, usdc_seed
            ),
            training=True,
            **add_env_kwargs,
        )

        bench_states.append(bench_states_this_game)
        time_cost.append(time.time() - start_time)
        scores.append(score)
        policies.append(policy)
        eps_history.append(agent.epsilon)
        states.append(state_this_game)
        rewards.append(reward_this_game)
        # if score is the highest, save the model
        if score >= max(scores) or (i + 1) % 100 == 0:
            trained_model.append(
                {
                    "episode": i,
                    "score": score,
                    "model": agent.Q_eval.state_dict(),
                }
            )

        chunk_size = 50

        avg_score = np.mean(scores[-chunk_size:])
        if i % chunk_size == 0:
            logging.info(
                "episode: {}, score: {:.2f}, average last {} scores: {:.2f}, epsilon: {:.2f}".format(
                    i,
                    score,
                    chunk_size,
                    avg_score,
                    agent.epsilon,
                )
            )

    return (
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        trained_model,
    )


def inference_with_trained_model(
    model: dict[str, Any],
    env: ProtocolEnv,
    agent_args: dict[str, Any],
    num_test_episodes: int = 1,
    compared_to_benchmark: bool = True,
) -> tuple[
    list[float],
    list[list[dict[str, Any]]],
    list[list[float]],
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
    list[list[float]],
]:
    """
    Interact with the environment using the loaded model.

    Args:
        model_path: The path of the saved model.
        env (ProtocolEnv): The environment to interact with.
        num_episodes (int, optional): The number of episodes to run. Defaults to 1.
        compared_to_benchmark (bool, optional): Whether to compare the performance with the benchmark. Defaults to True.

    Returns:
        The scores, states, rewards, bench_states, trained_model, policies.
    """
    # Create an agent with the same settings as during training
    agent_args["n_actions"] = env.action_space.n
    agent_args["input_dims"] = env.observation_space.shape
    agent = Agent(**agent_args)
    agent.Q_eval.load_state_dict(model["model"])
    agent.Q_eval.eval()
    states = []

    (
        scores,
        eps_history,
        states,
        trained_model,
        bench_states,
        policies,
        rewards,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Run the specified number of episodes
    for i in range(num_test_episodes):
        (
            score,
            reward_this_game,
            policy,
            state_this_game,
            bench_states_this_game,
        ) = run_episode(
            env=env,
            agent=agent,
            compared_to_benchmark=compared_to_benchmark,
            tkn_price_trend_this_game=env.defi_env.plf_pools["tkn"].asset_price_history,
            usdc_price_trend_this_game=env.defi_env.plf_pools[
                "usdc"
            ].asset_price_history,
            training=False,
            max_steps=env.defi_env.max_steps,
            initial_collateral_factor=env.defi_env.plf_pools[
                "tkn"
            ]._initial_collar_factor,
            init_safety_borrow_margin=env.defi_env.users[
                "alice"
            ]._initial_safety_borrow_margin,
            init_safety_supply_margin=env.defi_env.users[
                "alice"
            ]._initial_safety_supply_margin,
        )

        bench_states.append(bench_states_this_game)
        scores.append(score)
        policies.append(policy)
        states.append(state_this_game)
        rewards.append(reward_this_game)

    return (
        scores,
        eps_history,
        states,
        rewards,
        bench_states,
        trained_model,
    )


if __name__ == "__main__":
    # show logging level at info
    logging.basicConfig(level=logging.INFO)
    N_GAMES = 20

    def tkn_price_trend_func(x, y):
        series = np.array(range(1, x + 2))
        series[3] = series[2] / 10
        series[9] = series[8] * 10
        return series

    agent_vars = {
        "eps_dec": 0,
        "eps_end": 0.01,
        "lr": 0.05,
        "gamma": 0.99,
        "epsilon": 1,
        "batch_size": 128,
        "target_on_point": 0.9,
    }

    (
        training_scores,
        training_eps_history,
        training_states,
        training_rewards,
        training_time_cost,
        training_bench_states,
        training_models,
    ) = train_env(
        agent_args=agent_vars,
        n_games=N_GAMES,
        initial_collateral_factor=0.75,
        max_steps=30,
        compared_to_benchmark=True,
        tkn_price_trend_func=tkn_price_trend_func,
    )

    test_env = init_env(
        initial_collateral_factor=0.75,
        max_steps=20,
        tkn_price_trend_func=tkn_price_trend_func,
    )

    test_protocol_env = ProtocolEnv(test_env)

    inference_with_trained_model(
        model=training_models[-1],
        env=test_protocol_env,
        agent_args=agent_vars,
        num_test_episodes=3,
    )
