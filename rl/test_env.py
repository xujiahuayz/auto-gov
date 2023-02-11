import random

from market_env.env import DefiEnv, PlfPool, PriceDict, User
from rl.rl_env import ProtocolEnv
from rl.test_market import TestMarket


def test_testmarket():
    # initialize market and environment
    market = TestMarket()
    env = ProtocolEnv(market)
    print(env.reset())
    print(env.step(2))
    print(env.reset())


def test_env1():
    # initialize
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

    print("initial state:", list(env.reset()))
    for j in range(100):
        for i in range(10):
            action = random.randint(0, 2)
            observation_, reward, done, _ = env.step(action)
            print(
                "Action: ",
                action,
                "\tReward: ",
                reward,
                "\tState: ",
                list(observation_),
            )
        print("initial state:", list(env.reset()))


def test_env2():
    # initialize
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

    for i in range(300):
        env.reset()
        # print(Alice)
        # print(plf)
        observation_, reward, done, _ = env.step(2)
        observation_, reward, done, _ = env.step(0)
        observation_, reward, done, _ = env.step(0)
        observation_, reward, done, _ = env.step(0)
        observation_, reward, done, _ = env.step(0)
        # print(Alice)
        # print(plf)
        # print("=====")
        # print(plf)
        print(plf)


if __name__ == "__main__":
    test_testmarket()
    test_env1()
    test_env2()
