from market_env.env import DefiEnv, PlfPool, User, PriceDict
from dqn_gov import Agent
from utils2 import plot_learning_curve
import numpy as np
from rl_env import ProtocolEnv

if __name__ == "__main__":
    # initialize environment
    defi_env = DefiEnv(prices=PriceDict({"tkn": 1}))
    Alice = User(name="alice", env=defi_env, funds_available={"tkn": 2_000})
    plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        initial_starting_funds=1000,
        asset_name="tkn",
        collateral_factor=0.8,
    )

    env = ProtocolEnv(defi_env)

    # initialize agent
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=128,
        n_actions=env.action_space.n,
        eps_end=0.1,
        input_dims=env.observation_space.shape,
        lr=0.001,
    )
    # agent = Agent(state_size, action_size)

    scores, eps_history = [], []
    n_games = 50000

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        # print("=====================================")
        # print(observation.astype(np.float32))
        # print("=====================================")
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
        print(
            "episode ",
            i,
            "score %.2f" % score,
            "average score %.2f" % avg_score,
            "epsilon %.2f" % agent.epsilon,
        )

    x = [i + 1 for i in range(n_games)]
    filename = "defi.png"
    plot_learning_curve(x, scores, eps_history, filename)
