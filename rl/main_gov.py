import logging
import time
from typing import Any, Callable

import numpy as np

from market_env.caching import cache
from rl.dqn_gov import Agent
from rl.rl_env import ProtocolEnv
from rl.utils import init_env


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
        # pre-generate price trend to make sure that the same price trend is used for both envs
        tkn_price_trend_this_game = tkn_price_trend_func(defi_env.max_steps, tkn_seed)
        usdc_price_trend_this_game = usdc_price_trend_func(
            defi_env.max_steps, usdc_seed
        )
        bench_rewards, bench_states_this_game = bench_env(
            tkn_price_trend_func=lambda t, s: tkn_price_trend_this_game,
            usdc_price_trend_func=lambda t, s: usdc_price_trend_this_game,
            **add_env_kwargs,
        )

        # extend bench_rewards with 0 to match the length of the game (max_steps)
        bench_rewards.extend([0.0] * (defi_env.max_steps + 1 - len(bench_rewards)))
        bench_states.append(bench_states_this_game)
        defi_env.plf_pools[
            "tkn"
        ].price_trend_func = lambda t, s: tkn_price_trend_this_game
        defi_env.plf_pools[
            "usdc"
        ].price_trend_func = lambda t, s: usdc_price_trend_this_game
        state_this_game = []
        score = 0
        done = False
        policy = []
        reward_this_game = []
        observation = env.reset()
        start_time = time.time()
        state_this_game.append(defi_env.state_summary)
        while not done:
            # get states for plotting
            action = agent.choose_action(observation.astype(np.float32))
            # this checks done or not

            observation_, reward, done, _ = env.step(action)
            # catch index error
            assert defi_env.step < len(
                bench_rewards
            ), "index out of range, step: {}, len(bench_rewards): {}".format(
                defi_env.step, len(bench_rewards)
            )
            reward -= bench_rewards[defi_env.step] if compared_to_benchmark else 0
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            policy.append(action)
            reward_this_game.append(reward)
            observation = observation_
            state_this_game.append(defi_env.state_summary)
        time_cost.append(time.time() - start_time)
        scores.append(score)
        policies.append(policy)
        eps_history.append(agent.epsilon)
        states.append(state_this_game)
        # if score is the highest, save the model
        if score >= max(scores) or (i + 1) % 100 == 0:
            trained_model.append(
                {
                    "episode": i,
                    "score": score,
                    "model": agent.Q_eval.state_dict(),
                }
            )
        rewards.append(reward_this_game)

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
) -> None:
    """
    Interact with the environment using the loaded model.

    Args:
        model_path: The path of the saved model.
        env (ProtocolEnv): The environment to interact with.
        num_episodes (int, optional): The number of episodes to run. Defaults to 1.
    """
    # Create an agent with the same settings as during training
    agent_args["n_actions"] = env.action_space.n
    agent_args["input_dims"] = env.observation_space.shape
    agent = Agent(**agent_args)
    agent.Q_eval.load_state_dict(model)

    # Run the specified number of episodes
    for episode in range(num_test_episodes):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation.astype(np.float32))
            observation, reward, done, _ = env.step(action)
            score += reward

        print(f"Episode {episode + 1}, score of this episode: {score}")


if __name__ == "__main__":
    # show logging level at info
    logging.basicConfig(level=logging.INFO)
    N_GAMES = 20

    def tkn_price_trend_func(x, y):
        return np.array(range(x + 1))

    agent_vars = {
        "eps_dec": 0,
        "eps_end": 0.01,
        "lr": 0.05,
        "gamma": 0.99,
        "epsilon": 1,
        "batch_size": 128,
        "target_net_enabled": True,
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
