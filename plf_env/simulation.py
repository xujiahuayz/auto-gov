from plf_env.strategist import NaiveStrategist, Strategist
from typing import Optional, Union
from plf_env.env import PLFEnv
from plf_env.agent import (
    Agent,
    MarketAggregateAgent,
    NaiveAgent,
    SpeculativeAgent,
    SpiralBorrowAgent,
)
from plf_env.constants import ADDRESSES, ADDRESSES_BY_NAME


# def create_strategist(plf_env: PLFEnv) -> Strategist:
#     return super_strategist


def create_agents(
    plf_env: PLFEnv, initial_funds: Optional[dict] = None
) -> list[Union[Agent, Strategist]]:
    super_strategist = NaiveStrategist(plf_env=plf_env)

    if initial_funds is None:
        initial_funds = {
            ADDRESSES_BY_NAME["WETH"]: 9.999,
            ADDRESSES_BY_NAME["DAI"]: 9.999,
        }
    aggregate_agent = MarketAggregateAgent(
        address="AggregateAgent",
        plf_env=plf_env,
        initial_unsupplied_amounts={
            ADDRESSES_BY_NAME["WETH"]: 9e5,
            ADDRESSES_BY_NAME["USDT"]: 9e5,
            ADDRESSES_BY_NAME["DAI"]: 9e5,
        },
    )
    luigi = NaiveAgent(
        address="Luigi",
        plf_env=plf_env,
        initial_unsupplied_amounts=initial_funds,
    )
    daisy = SpiralBorrowAgent(
        address="daisy",
        plf_env=plf_env,
        initial_unsupplied_amounts=initial_funds,
    )
    mario = SpeculativeAgent(
        address="Mario",
        plf_env=plf_env,
        initial_unsupplied_amounts=initial_funds,
    )

    return [super_strategist, aggregate_agent, luigi, daisy, mario]
