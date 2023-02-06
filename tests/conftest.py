import pandas as pd
from plf_env.simulation_manager import organize_market_data, organize_oracle_price
from plf_env.env import Market, PLFEnv, ExternalData
import pytest


@pytest.fixture
def plf_env(
    dai_market_name: str,
    dai_address: str,
    bat_market_name: str,
    bat_address: str,
    collateral_factors: dict[str, float],
    liquidation_thresholds: dict[str, float],
    liquidation_discounts: dict[str, float],
    prices: dict[str, float],
    volumes: dict[str, float],
    volatility: dict[str, float],
    alice: str,
    bob: str,
):
    plf_env = PLFEnv()
    plf_env.external_data = ExternalData(
        prices=prices, volume=volumes, volatility=volatility
    )
    plf_env.markets[dai_market_name] = Market(
        dai_market_name,
        dai_address,
        collateral_factor=collateral_factors[dai_market_name],
        liquidation_threshold=liquidation_thresholds[dai_market_name],
        liquidation_discount=liquidation_discounts[dai_market_name],
    )
    plf_env.markets[bat_market_name] = Market(
        bat_market_name,
        bat_address,
        collateral_factor=collateral_factors[bat_market_name],
        liquidation_threshold=liquidation_thresholds[bat_market_name],
        liquidation_discount=liquidation_discounts[bat_market_name],
    )

    # alice starts with 100 DAI
    plf_env.markets[dai_market_name].get_user(alice).unsupplied_amount = 100
    plf_env.markets[bat_market_name].get_user(bob).unsupplied_amount = 100

    return plf_env


@pytest.fixture
def market_data() -> pd.DataFrame:
    return organize_market_data()


@pytest.fixture
def oracle_feeds() -> pd.DataFrame:
    return organize_oracle_price()


@pytest.fixture
def alice() -> str:
    return "alice"


@pytest.fixture
def bob() -> str:
    return "bob"


@pytest.fixture
def dai_market_name() -> str:
    return "dai"


@pytest.fixture
def dai_address() -> str:
    return "0x6B175474E89094C44Da98b954EedeAC495271d0F"


@pytest.fixture
def bat_market_name() -> str:
    return "bat"


@pytest.fixture
def bat_address() -> str:
    return "0x0D8775F648430679A709E98d2b0Cb6250d2887EF"


@pytest.fixture
def collateral_factors(dai_market_name: str, bat_market_name: str) -> dict[str, float]:
    return {
        dai_market_name: 0.8,
        bat_market_name: 0.7,
    }


@pytest.fixture
def liquidation_thresholds(
    dai_market_name: str, bat_market_name: str
) -> dict[str, float]:
    return {
        dai_market_name: 0.9,
        bat_market_name: 0.8,
    }


@pytest.fixture
def liquidation_discounts(
    dai_market_name: str, bat_market_name: str
) -> dict[str, float]:
    return {
        dai_market_name: 0.95,
        bat_market_name: 0.9,
    }


@pytest.fixture
def prices(dai_market_name: str, bat_market_name: str) -> dict[str, float]:
    return {dai_market_name: 1.0, bat_market_name: 3.0}


@pytest.fixture
def volumes(dai_market_name: str, bat_market_name: str) -> dict[str, float]:
    return {dai_market_name: 5e9, bat_market_name: 2e7}


@pytest.fixture
def volatility(dai_market_name: str, bat_market_name: str) -> dict[str, float]:
    return {dai_market_name: 1.0, bat_market_name: 2.8}
