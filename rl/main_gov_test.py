import gym
from dqn_gov import Agent
from utils2 import plot_learning_curve
import numpy as np
from test_market import TestMarket
from rl_env import ProtocolEnv, DefiProtocolEnv

if __name__ == "__main__":
    # initialize market and environment
    market = TestMarket()
    env = DefiProtocolEnv(market)

    # initialize agent
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=128,
        n_actions=env.action_space.n,
        eps_end=0.1,
        input_dims=env.observation_space.shape,
        lr=0.003,
    )

    scores, eps_history = [], []
    n_games = 500000

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

        avg_score = np.mean(scores[-256:])
        print(
            "episode ",
            i,
            "score %.2f" % score,
            "average score %.2f" % avg_score,
            "epsilon %.2f" % agent.epsilon,
        )

    x = [i + 1 for i in range(n_games)]
    filename = "defi_test.png"
    plot_learning_curve(x, scores, eps_history, filename)

    # for episode in range(num_episodes):
    #     state = env.reset()
    #     total_reward = 0
    #     done = False
    #     while not done:
    #         action = agent.act(state)
    #         next_state, reward, done, _ = env.step(action)
    #         agent.learn(state, action, reward, next_state, done)
    #         state = next_state
    #         total_reward += reward
    #     print(f"Episode {episode} finished with reward {total_reward}")
