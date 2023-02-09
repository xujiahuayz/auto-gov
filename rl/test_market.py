import gym
import numpy as np


class TestMarket:
    def __init__(self):
        self.collateral_factor: float = 0.85
        self.cumulative_protocol_earning: float = 0
        self.this_step_protocol_earning: float = 0

        self.steps = 0
        self.max_steps = 10000

        # initial funds
        self.initial_starting_funds: float = 10000
        self.total_available_funds: float = self.initial_starting_funds
        self.total_borrowed_funds: float = 0

    def get_state(self) -> np.ndarray:
        return np.array(
            [
                self.total_available_funds,
                self.total_borrowed_funds,
                self.collateral_factor,
            ]
        )

    def update_market(self) -> None:
        # TODO: update market
        self.total_available_funds
        self.total_borrowed_funds
        self.this_step_protocol_earning
        self.cumulative_protocol_earning += self.this_step_protocol_earning

    def lower_collateral_factor(self) -> None:
        self.collateral_factor -= 0.01
        self.update_market()

    def keep_collateral_factor(self) -> None:
        self.update_market()

    def raise_collateral_factor(self) -> None:
        self.collateral_factor += 0.01
        self.update_market()

    def get_reward(self) -> float:
        reward = self.this_step_protocol_earning
        return reward

    def is_done(self) -> bool:
        self.steps += 1
        if self.steps >= self.max_steps:
            return True
        return False
