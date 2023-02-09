import numpy as np


class TestMarket:
    def __init__(self):
        self.collateral_factor: float = 0.8
        self.cumulative_protocol_earning: float = 0
        self.this_step_protocol_earning: float = 0

        self.steps = 0
        self.max_steps = 128

        # initial funds
        self.total_funds: float = 10000
        if self.collateral_factor <= 0.5:
            util_ratio = self.collateral_factor
        else:
            util_ratio = 1 - self.collateral_factor
        self.total_borrowed_funds = self.total_funds * util_ratio
        self.total_available_funds = self.total_funds - self.total_borrowed_funds

    def reset(self):
        self.collateral_factor: float = 0.8
        self.cumulative_protocol_earning: float = 0
        self.this_step_protocol_earning: float = 0

        self.steps = 0
        self.max_steps = 128

        # initial funds
        self.total_funds: float = 10000
        if self.collateral_factor <= 0.5:
            util_ratio = self.collateral_factor
        else:
            util_ratio = 1 - self.collateral_factor
        self.total_borrowed_funds = self.total_funds * util_ratio
        self.total_available_funds = self.total_funds - self.total_borrowed_funds

    def get_state(self) -> np.ndarray:
        return np.array(
            [
                self.total_available_funds,
                self.total_borrowed_funds,
                self.collateral_factor,
            ]
        )

    def update_market(self) -> None:
        # update all the market parameters

        # print(self.collateral_factor)
        if self.collateral_factor <= 0.5:
            util_ratio = self.collateral_factor
        else:
            util_ratio = 1 - self.collateral_factor

        self.total_borrowed_funds = self.total_funds * util_ratio
        self.total_available_funds = self.total_funds - self.total_borrowed_funds

        self.this_step_protocol_earning = self.total_borrowed_funds * 0.1
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
            print(self.collateral_factor)
            return True
        return False
