import numpy as np


class TestMarket:
    def __init__(self):
        self.steps = 0
        self.max_steps = 256

        self.collateral_factor: float = 0.8
        # initial funds
        self.total_funds: float = 10000
        if self.collateral_factor <= 0.5:
            util_ratio = self.collateral_factor
        else:
            util_ratio = 1 - self.collateral_factor
        self.total_borrowed_funds = self.total_funds * util_ratio
        self.total_available_funds = self.total_funds - self.total_borrowed_funds
        self.this_step_protocol_earning: float = self.total_borrowed_funds * 0.1
        self.previous_earning = self.this_step_protocol_earning

    def reset(self):
        self.steps = 0
        self.max_steps = 128

        self.collateral_factor: float = 0.8
        # initial funds
        self.total_funds: float = 10000
        if self.collateral_factor <= 0.5:
            util_ratio = self.collateral_factor
        else:
            util_ratio = 1 - self.collateral_factor
        self.total_borrowed_funds = self.total_funds * util_ratio
        self.total_available_funds = self.total_funds - self.total_borrowed_funds
        self.this_step_protocol_earning: float = self.total_borrowed_funds * 0.1
        self.previous_earning = self.this_step_protocol_earning

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
        self.previous_earning = self.this_step_protocol_earning
        # print(self.collateral_factor)
        if self.collateral_factor <= 0.5:
            util_ratio = self.collateral_factor
        else:
            util_ratio = 1 - self.collateral_factor

        self.total_borrowed_funds = self.total_funds * util_ratio
        self.total_available_funds = self.total_funds - self.total_borrowed_funds

        self.this_step_protocol_earning = self.total_borrowed_funds * 0.1

    def lower_collateral_factor(self) -> None:
        self.collateral_factor -= 0.02
        self.update_market()

    def keep_collateral_factor(self) -> None:
        self.update_market()

    def raise_collateral_factor(self) -> None:
        self.collateral_factor += 0.02
        self.update_market()

    def get_reward(self) -> float:
        reward = self.this_step_protocol_earning - self.previous_earning
        return reward

    def is_done(self) -> bool:
        self.steps += 1
        if self.steps >= self.max_steps:
            print("collateral_factor " + str(self.collateral_factor))
            return True
        return False


if __name__ == "__main__":
    # # initialize market and environment
    market = TestMarket()
    market.collateral_factor = 0.4
    market.update_market()

    market.raise_collateral_factor()
    print(market.get_reward())
    market.lower_collateral_factor()
    print(market.get_reward())
    market.lower_collateral_factor()
    print(market.get_reward())
