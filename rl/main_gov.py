import logging
import time
from typing import Any, Callable

import numpy as np

from market_env.caching import cache
from rl.dqn_gov import Agent, contain_nan
from rl.rl_env import ProtocolEnv
from rl.utils import init_env


def run_episode(
    env: ProtocolEnv,
    agent: Agent,
    compared_to_benchmark: bool,
    tkn_price_trend_this_episode: np.ndarray,
    usdc_price_trend_this_episode: np.ndarray,
    training: bool,
    attack_steps: list[int] | None,
    **add_env_kwargs,
) -> tuple[
    float,
    list[float],
    list[int],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
]:
    bench_rewards, bench_states_this_episode = bench_env(
        tkn_price_trend_func=lambda t, s: tkn_price_trend_this_episode,
        usdc_price_trend_func=lambda t, s: usdc_price_trend_this_episode,
        attack_steps=attack_steps,
        **add_env_kwargs,
    )

    # extend bench_rewards with 0 to match the length of the game (max_steps)
    bench_rewards.extend([0.0] * (env.defi_env.max_steps + 1 - len(bench_rewards)))
    env.defi_env.plf_pools[
        "tkn"
    ].price_trend_func = lambda t, s: tkn_price_trend_this_episode
    env.defi_env.plf_pools[
        "usdc"
    ].price_trend_func = lambda t, s: usdc_price_trend_this_episode
    env.defi_env.attack_steps = attack_steps

    score = 0
    done = False
    policy = []
    loss_this_episode = []
    reward_this_episode = []
    observation = env.reset()
    state_this_episode = [env.defi_env.state_summary]
    while not done:
        # for debug!
        arr = np.array(observation, dtype=np.float32)
        if np.any(~np.isfinite(arr)):
            print("Array contains NaN or infinity!!!!!!!!!!!!!")
            print("Index number: " + str(len(agent.loss_list)))
            print("=" * 20 + "Before converting to float32" + "=" * 20)
            for num in observation:
                print(f"{num:.20f}")
            print("=" * 20 + "After converting to float32" + "=" * 20)
            for num in arr:
                print(f"{num:.20f}")
            observation = observation.astype(np.float32)

        # for debug!
        # check whether the agent model contains nan
        if contain_nan(agent.Q_eval.state_dict()):
            print("Model contains nan !!!")
            print("Model:")
            print(agent.Q_eval.state_dict())
            print("Index number: " + str(len(agent.loss_list)))
            print("the last 20 losses:")
            print(agent.loss_list[-20:])
            print("=" * 20 + "observation_" + "=" * 20)
            for num in observation_:
                print(f"{num:.20f}")
            print("=" * 20 + "observation" + "=" * 20)
            for num in observation:
                print(f"{num:.20f}")
            exit()

        # get states for plotting
        action = agent.choose_action(
            observation.astype(np.float32), evaluate=not training
        )
        # observation_ is the next state
        # reward is the reward of the current state
        # done is whether the game is over
        observation_, reward, done, _ = env.step(action)
        reward -= bench_rewards[env.defi_env.step] if compared_to_benchmark else 0

        if training:
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
        observation = observation_
        # if agent.loss_list[-1] exists, append it to loss_this_episode, else append 0
        if len(agent.loss_list) > 0:
            loss_this_episode.append(agent.loss_list[-1])
        else:
            loss_this_episode.append(0)

        score += reward
        policy.append(action)
        reward_this_episode.append(reward)
        state_this_episode.append(env.defi_env.state_summary)

    return (
        score,
        reward_this_episode,
        policy,
        state_this_episode,
        bench_states_this_episode,
        # average loss of this episode
        np.mean(loss_this_episode),
    )


def bench_env(**kwargs) -> tuple[list[float], list[dict[str, Any]]]:
    defi_env = init_env(**kwargs)
    env = ProtocolEnv(defi_env)
    state_this_episode = [defi_env.state_summary]
    score = 0
    done = False
    rewards = [0.0]
    defi_env.reset()

    while not done:
        # get states for plotting
        # never change collateral factor
        _, reward, done, _ = env.step(0)
        score += reward
        state_this_episode.append(defi_env.state_summary)
        rewards.append(reward)

    return rewards, state_this_episode


@cache(ttl=60 * 60 * 24 * 7, min_memory_time=0.00001, min_disk_time=0.1)
def train_env(
    agent_args: dict[str, Any],
    n_episodes: int = 2_000,
    compared_to_benchmark: bool = True,
    tkn_price_trend_func: Callable[
        [int, int | None], np.ndarray
    ] = lambda x, y: np.ones(x),
    usdc_price_trend_func: Callable[
        [int, int | None], np.ndarray
    ] = lambda x, y: np.ones(x),
    tkn_seed: int | None = None,
    usdc_seed: int | None = None,
    attack_steps: Callable[[int], list[int]] | None = None,
    **add_env_kwargs,
) -> tuple[
    list[float],
    list[float],
    list[list[dict[str, Any]]],
    list[list[float]],
    list[float],
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
    list[float],
    list[dict[str, Any]],
]:
    """
    Return:
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        trained_model,
        losses,
        exogenous_states,
    """
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
        avg_loss,
        exogenous_states,
    ) = ([], [], [], [], [], [], [], [], [], [])

    for i in range(n_episodes):
        start_time = time.time()
        attack_steps_this_episode = (
            attack_steps(defi_env.max_steps) if attack_steps else None
        )
        tkn_price_trend_this_episode = tkn_price_trend_func(
            defi_env.max_steps, tkn_seed
        )
        usdc_price_trend_this_episode = usdc_price_trend_func(
            defi_env.max_steps, usdc_seed
        )

        exogenous_states.append(
            {
                "tkn_price_trend": tkn_price_trend_this_episode,
                "usdc_price_trend": usdc_price_trend_this_episode,
                "attack_steps": attack_steps_this_episode,
            }
        )
        (
            score,
            reward_this_episode,
            policy,
            state_this_episode,
            bench_states_this_episode,
            avg_loss_this_episode,
        ) = run_episode(
            env=env,
            agent=agent,
            compared_to_benchmark=compared_to_benchmark,
            tkn_price_trend_this_episode=tkn_price_trend_this_episode,
            usdc_price_trend_this_episode=usdc_price_trend_this_episode,
            training=True,
            attack_steps=attack_steps_this_episode,
            **add_env_kwargs,
        )

        bench_states.append(bench_states_this_episode)
        time_cost.append(time.time() - start_time)
        scores.append(score)
        policies.append(policy)
        eps_history.append(agent.epsilon)
        states.append(state_this_episode)
        rewards.append(reward_this_episode)
        avg_loss.append(avg_loss_this_episode)
        # if score is the highest, save the model
        if score >= max(scores) or (i + 1) % 100 == 0:
            trained_model.append(
                {
                    "episode": i,
                    "model": agent.Q_eval.state_dict(),
                }
            )

        chunk_size = 50

        avg_score = np.mean(scores[-chunk_size:])
        if i % chunk_size == 0:
            logging.info(
                "Episode: {}, time taken: {:.2f}s, the last score: {:.2f}, average last {} scores: {:.2f}, epsilon: {:.4f}".format(
                    i,
                    sum(time_cost[-chunk_size:]),
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
        avg_loss,
        exogenous_states,
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
    list[list[dict[str, Any]]],
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
        scores,
        states,
        rewards,
        bench_states,
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
        states,
        bench_states,
        policies,
        rewards,
    ) = (
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
            reward_this_episode,
            policy,
            state_this_episode,
            bench_states_this_episode,
            # loss will be all zeros in inference mode
            avg_loss,
        ) = run_episode(
            env=env,
            agent=agent,
            compared_to_benchmark=compared_to_benchmark,
            tkn_price_trend_this_episode=env.defi_env.plf_pools[
                "tkn"
            ].asset_price_history,
            usdc_price_trend_this_episode=env.defi_env.plf_pools[
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
            attack_steps=None,
        )

        bench_states.append(bench_states_this_episode)
        scores.append(score)
        policies.append(policy)
        states.append(state_this_episode)
        rewards.append(reward_this_episode)

    return (
        scores,
        states,
        rewards,
        bench_states,
    )


if __name__ == "__main__":
    # show logging level at info
    logging.basicConfig(level=logging.INFO)
    N_EPISODES = 2

    def tkn_price_trend_func(x, y):
        series = np.array(range(1, x + 2)).astype(float)
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

    result_unpacked = train_env(
        agent_args=agent_vars,
        n_episodes=N_EPISODES,
        initial_collateral_factor=0.99,
        max_steps=360,
        compared_to_benchmark=True,
        tkn_price_trend_func=tkn_price_trend_func,
        attack_steps=None,
    )

    # also in rl.plotting
    test_env = init_env(
        initial_collateral_factor=0.99,
        max_steps=20,
        tkn_price_trend_func=tkn_price_trend_func,
        attack_steps=[6, 11, 32],
    )

    # test_protocol_env = ProtocolEnv(test_env)

    # inference_with_trained_model(
    #     model=training_models[-1],
    #     env=test_protocol_env,
    #     agent_args=agent_vars,
    #     num_test_episodes=3,
    # )
