import logging
from typing import Callable, Optional, TypeVar, Union
from plf_env.computations import scale
from plf_env.constants import (
    ATOKEN_EPSILON,
    DECIMALS,
    EPSILON,
    NULL_ADDRESS,
    RESERVE_BY_ATOKEN,
)
from plf_env.core import BaseAction, Event, ProtocolAction, UserAction
from plf_env.env import PLFEnv

BORROW_EPSILON = 0.999

T = TypeVar("T")

LazyOrStrict = Union[T, Callable[[], T]]


def to_strict(value: LazyOrStrict[T]) -> T:
    if callable(value):
        return value()
    return value


class Exchange(UserAction):
    """
    roughly modeling a swap on some AMM
    """

    # https://github.com/aave/protocol-v2/blob/77acec9395c250a0f9df8f73ed99618f9c14a101/contracts/mocks/swap/MockUniswapV2Router02.sol
    SWAP_FEE = 0.004  # fee + some fixed slippage

    def __init__(
        self, user: str, from_market: str, to_market: str, value: LazyOrStrict[float]
    ):
        super().__init__(user, from_market)
        self.value = value
        self.from_market = from_market
        self.to_market = to_market

    def execute(self, env: PLFEnv):
        self.value = to_strict(self.value)
        from_market_user = env.markets[self.from_market].get_user(self.user)
        to_market_user = env.markets[self.to_market].get_user(self.user)

        if from_market_user.unsupplied_amount < self.value:
            return

        from_quote_price = env.external_data.prices[self.from_market]
        to_quote_price = env.external_data.prices[self.to_market]
        resulting_amount = self.value * from_quote_price / to_quote_price
        fees = resulting_amount * self.SWAP_FEE
        final_amount = resulting_amount - fees
        from_market_user.unsupplied_amount -= self.value
        to_market_user.unsupplied_amount += final_amount


class Deposit(UserAction):  # equivalent to mint (eg. cTokens, aTokens)
    def __init__(
        self, user: str, on_behalf_of: str, market: str, value: LazyOrStrict[float]
    ):
        super().__init__(user, market)
        self.on_behalf_of = on_behalf_of
        self.value = value

    def execute(self, env: PLFEnv):
        self.value = to_strict(self.value)
        # market_user = env.markets[self.market].get_user(self.user)

        # assert (
        #     market_user.unsupplied_amount >= self.value
        # ), "Insufficient funds to deposit"

        market = env.markets[self.market]

        market.update_indices()
        market.update_interest(liquidity_delta=self.value)

        market.process_deposit(self.user, self.on_behalf_of, self.value)


class Redeem(UserAction):
    def __init__(self, user: str, on_behalf_of: str, market: str, value: float):
        super().__init__(user, market)
        self.on_behalf_of = on_behalf_of
        self.value = value

    def execute(self, env: PLFEnv):
        market = env.markets[self.market]
        deltas = {self.market: self.value}
        health_factor = env.get_health(self.user, deltas)
        # TODO: this needs to be changed
        if health_factor >= 1 - EPSILON:
            market.update_indices()
            market.update_interest(liquidity_delta=-self.value)
            market.process_redeem(self.user, self.on_behalf_of, self.value)
        else:
            value_eth = self.value * env.external_data.prices[self.market]
            raise ValueError(
                f"{self.user} cannot redeem {self.value} ({value_eth} ETH) {market.name} on behalf of "
                f"{self.on_behalf_of} with health factor of {round(health_factor, 3)}, "
                f"borrow = {env.get_user_borrow_in_quote(self.user)}, "
                f"supply = {env.get_user_supply_value(self.user)}, "
                f"prices = {env.external_data.prices}"
                f"borrow balance = {[{name: market.get_user(self.user).total_borrow}for name, market in env.markets.items()]}"
                f"supply balance = {[{name: market.get_user(self.user).atoken_balance} for name, market in env.markets.items()]}"
            )


class Borrow(UserAction):
    def __init__(
        self,
        user: str,
        on_behalf_of: str,
        market: str,
        value: Union[float, Callable[[], float]],
        is_stable_borrow: bool = True,
    ):
        super().__init__(user, market)
        self.value = value
        self.on_behalf_of = on_behalf_of
        self.is_stable_borrow = is_stable_borrow

    def execute(self, env: PLFEnv):
        self.value = to_strict(self.value)
        market = env.markets[self.market]

        amount_in_quote = self.value * env.external_data.prices[self.market]

        ltv = env.get_ltv(self.on_behalf_of, amount_in_quote)
        if ltv > 1 + EPSILON:
            raise ValueError(
                f"unable to borrow {self.value} {market.name} on behalf of {self.on_behalf_of}, ltv = {ltv}"
            )

        if market.underlying_available < self.value * BORROW_EPSILON:
            logging.warning(
                f"{self.user} unable to borrow: not enough liquidity, %s < %s",
                market.underlying_available,
                self.value,
            )
            return

        market.update_indices()

        market.process_borrow(
            self.user,
            self.on_behalf_of,
            self.value,
            self.is_stable_borrow,
        )

        market.update_interest()


class Repay(UserAction):
    def __init__(
        self,
        user: str,
        on_behalf_of: str,
        market: str,
        value: LazyOrStrict[float],
        is_stable_borrow: Optional[bool] = None,
    ):
        super().__init__(user, market)
        self.value = value
        self.on_behalf_of = on_behalf_of
        self.is_stable_borrow = is_stable_borrow

    def execute(self, env: PLFEnv):
        self.value = to_strict(self.value)
        market = env.markets[self.market]

        market.update_indices()

        market.process_repay(
            self.user, self.on_behalf_of, self.value, self.is_stable_borrow
        )

        market.update_interest()


class Swap(UserAction):
    def __init__(self, user: str, market: str, was_stable_borrow: bool):
        super().__init__(user, market)
        self.was_stable_borrow = was_stable_borrow

    def execute(self, env: "PLFEnv"):
        market = env.markets[self.market]
        market.process_swap(user=self.user, was_stable=self.was_stable_borrow)
        market.update_interest()


class Transfer(UserAction):
    def __init__(self, market: str, sender: str, receiver: str, amount: float):
        super().__init__(sender, market)
        self.receiver = receiver
        self.amount = amount

    def execute(self, env: "PLFEnv"):
        if self.user == NULL_ADDRESS or self.receiver == NULL_ADDRESS:
            return
        market = env.markets[self.market]
        market.process_transfer(self.user, self.receiver, self.amount)


class Liquidate(UserAction):
    def __init__(
        self,
        liquidator: str,
        borrower: str,
        borrow_market: str,
        repaid_borrow_amount: float,
        collateral_market: str,
        receive_atoken: bool = False,
    ):
        super().__init__(liquidator, borrow_market)
        self.borrower = borrower
        self.liquidator = liquidator
        self.market_loan = borrow_market
        self.borrow_amount_repaid = repaid_borrow_amount
        self.market_coll = collateral_market
        self.receive_atoken = receive_atoken

    def execute(self, env: PLFEnv):
        loan_price = env.external_data.prices[self.market_loan]

        market_loan = env.markets[self.market_loan]
        market_collateral = env.markets[self.market_coll]

        # make sure loan insufficiently collateralized at liquidation
        # TODO: unable to check during replay
        health_factor = env.get_health(self.borrower)
        assert health_factor <= 1 + EPSILON, (
            f"health factor = {health_factor}, "
            f"borrow in quote = {env.get_user_borrow_in_quote(self.borrower)}",
            f"user supply = {env.get_user_discounted_supply_value_liquidation(self.borrower, {})}",
            f"cannot liquidate {self.borrower}",
        )

        market_loan.update_indices()
        market_collateral.update_indices()

        # make sure the asset to be slashed has been enabled as collateral
        assert market_collateral.get_user(
            self.borrower
        ).collateral_enabled, "Borrower supply not enabled as collateral"

        # make sure not to repay more than the loan * its close_factor
        borrow = market_loan.get_user(self.borrower).total_borrow
        # cannot repay more than close factor
        self.borrow_amount_repaid = min(
            self.borrow_amount_repaid, borrow * market_loan.close_factor
        )

        liquidation_discount = market_collateral.liquidation_discount

        coll_price = env.external_data.prices[self.market_coll]

        coll_slashed = (
            loan_price * self.borrow_amount_repaid / liquidation_discount / coll_price
        )

        # coll_iquidable_fraction = (loan_price * loanamt_repaid) / (ld * coll_value)

        # repaid loan value cannot exceed discounted collateral value
        # sanity check, should be fine if repay below close_factor
        user_collateral = market_collateral.user_atoken_balance(self.borrower)
        assert (
            user_collateral * (1 + ATOKEN_EPSILON) >= coll_slashed
        ), f"Insufficient collateral to purchase {self.borrower} collateral: {user_collateral} < {coll_slashed}"

        if user_collateral < coll_slashed:
            coll_slashed = user_collateral

        # liquidator have enough balance to pay
        # TODO: unable to check in replay
        # assert (
        #     self.borrow_amount_repaid
        #     <= market_loan.get_user(self.liquidator).unsupplied_amount
        # ), "Insufficient balance to repay loan"

        market_collateral.liquidate_collateral_market(
            self.liquidator,
            self.borrower,
            coll_slashed,
            receive_atoken=self.receive_atoken,
        )
        market_loan.liquidate_borrow_market(
            self.liquidator,
            self.borrower,
            self.borrow_amount_repaid,
        )

        market_collateral.update_interest()
        market_loan.update_interest()


class UpdateCollateralFactor(ProtocolAction):
    def __init__(self, market: str, value: float):
        super().__init__(market)
        self.market = market
        self.value = value

    def execute(self, env: PLFEnv):
        market = env.markets[self.market]
        assert (
            0 <= self.value <= market.liquidation_threshold
        ), "collateral factor must be between 0 and liquidation threshold"
        market.collateral_factor = self.value


class UpdateLiquidationThreshold(ProtocolAction):
    def __init__(self, market: str, value: float):
        super().__init__(market)
        self.market = market
        self.value = value

    def execute(self, env: PLFEnv):
        market = env.markets[self.market]
        assert (
            market.collateral_factor <= self.value <= market.liquidation_discount
        ), "liquidation threshold must be between collateral factor and liquidation discount"
        market.liquidation_threshold = self.value


class UpdateLiquidationDiscount(ProtocolAction):
    def __init__(self, market: str, value: float):
        super().__init__(market)
        self.market = market
        self.value = value

    def execute(self, env: PLFEnv):
        market = env.markets[self.market]
        assert (
            market.liquidation_threshold <= self.value <= 1
        ), "Liquidation discount must be between liquidation threshold and 1"
        market.liquidation_discount = self.value


class ConfigureRiskParam(ProtocolAction):
    def __init__(
        self,
        market: str,
        new_collateral_factor: float,
        new_liquidation_threshold: float,
        new_liquidation_discount: float,
    ):
        super().__init__(market)
        self.market = market
        self.new_collateral_factor = new_collateral_factor
        self.new_liquidation_threshold = new_liquidation_threshold
        self.new_liquidation_discount = new_liquidation_discount

    def execute(self, env: PLFEnv):
        market = self.market
        UpdateCollateralFactor(market, self.new_collateral_factor).execute(env)
        UpdateLiquidationThreshold(market, self.new_liquidation_threshold).execute(env)
        UpdateLiquidationDiscount(market, self.new_liquidation_discount).execute(env)


class UpdateReserveFactor(ProtocolAction):
    def __init__(self, market: str, new_reserve_factor: float):
        super().__init__(market)
        self.new_reserve_factor = new_reserve_factor

    def execute(self, env: PLFEnv):
        market = env.markets[self.market]
        assert (
            0 <= self.new_reserve_factor <= 1
        ), f"reserve factor should be between 0 and 1, got {self.new_reserve_factor}"
        market.reserve_factor = self.new_reserve_factor


def create_from_event(event: Event) -> Optional[BaseAction]:
    args = event.args
    event_name = event.event
    if event_name == "Deposit":
        return Deposit(
            user=args["user"],
            on_behalf_of=args["onBehalfOf"],
            market=args["reserve"],
            value=scale(args["amount"], DECIMALS[args["reserve"]]),
        )
    elif event_name == "Withdraw":
        return Redeem(
            user=args["user"],
            on_behalf_of=args["to"],
            market=args["reserve"],
            value=scale(args["amount"], DECIMALS[args["reserve"]]),
        )
    elif event_name == "Borrow":
        return Borrow(
            user=args["user"],
            on_behalf_of=args["onBehalfOf"],
            market=args["reserve"],
            value=scale(args["amount"], DECIMALS[args["reserve"]]),
            is_stable_borrow=args["borrowRateMode"] == 1,
        )
    elif event_name == "Repay":
        # see: https://github.com/aave/protocol-v2/blob/master/contracts/protocol/lendingpool/LendingPool.sol#L287
        return Repay(
            user=args["repayer"],
            on_behalf_of=args["user"],
            market=args["reserve"],
            value=scale(args["amount"], DECIMALS[args["reserve"]]),
        )
    elif event_name == "Swap":
        return Swap(
            user=args["user"],
            market=args["reserve"],
            was_stable_borrow=args["rateMode"] == 1,
        )
    elif event_name == "LiquidationCall":
        debt_asset = args["debtAsset"]
        return Liquidate(
            liquidator=args["liquidator"],
            borrower=args["user"],
            borrow_market=debt_asset,
            collateral_market=args["collateralAsset"],
            repaid_borrow_amount=scale(args["debtToCover"], DECIMALS[debt_asset]),
            receive_atoken=args["receiveAToken"],
        )
    elif event_name == "CollateralConfigurationChanged":
        liquidation_bonus = args["liquidationBonus"]
        liquidation_discount = (
            1 if liquidation_bonus == 0 else 10_000 / liquidation_bonus
        )
        return ConfigureRiskParam(
            args["asset"],
            args["ltv"] / 10_000,
            args["liquidationThreshold"] / 10_000,
            liquidation_discount,
        )
    elif event_name == "ReserveFactorChanged":
        return UpdateReserveFactor(args["asset"], args["factor"] / 10_000)
    elif event_name == "Transfer":
        market = RESERVE_BY_ATOKEN[event.address]
        return Transfer(
            market=market,
            sender=args["from"],
            receiver=args["to"],
            amount=scale(args["value"], DECIMALS[market]),
        )
    else:
        logging.debug("unsupported action %s", event_name)
