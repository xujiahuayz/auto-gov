from unittest.mock import call, patch

import pandas as pd
from plf_env.constants import BLOCK_BREAKPOINT, DECIMALS
from plf_env.core import Event
from plf_env.env import PLFEnv
from plf_env.simulation_manager import SimulationManager

start_block = BLOCK_BREAKPOINT[0]


def test_execute_blocks(
    plf_env: PLFEnv,
    market_data: pd.DataFrame,
    oracle_feeds: pd.DataFrame,
    alice: str,
    dai_market_name: str,
    dai_address: str,
):
    event1 = Event(
        args={
            "reserve": dai_address,
            "onBehalfOf": alice,
            "referral": 0,
            "user": alice,
            "amount": 1 * 10 ** DECIMALS[dai_address],
        },
        event="Deposit",
        log_index=0,
        transaction_index=0,
        transaction_hash="event1",
        address=dai_market_name,
        block_hash="",
        block_number=start_block + 1,
    )

    event2 = Event(
        args={
            "reserve": dai_address,
            "onBehalfOf": alice,
            "referral": 0,
            "user": alice,
            "amount": 10 * 10 ** DECIMALS[dai_address],
        },
        event="Deposit",
        log_index=1,
        transaction_index=1,
        transaction_hash="event2",
        address=dai_market_name,
        block_hash="",
        block_number=start_block + 1,
    )

    event3 = Event(
        args={
            "reserve": dai_address,
            "onBehalfOf": alice,
            "referral": 0,
            "user": alice,
            "amount": 100 * 10 ** DECIMALS[dai_address],
        },
        event="Deposit",
        log_index=199,
        transaction_index=173,
        transaction_hash="",
        address=dai_market_name,
        block_hash="",
        block_number=start_block + 2,
    )

    event4 = Event(
        args={
            "reserve": dai_address,
            "onBehalfOf": alice,
            "referral": 0,
            "user": alice,
            "amount": 1000 * 10 ** DECIMALS[dai_address],
        },
        event="Deposit",
        log_index=199,
        transaction_index=173,
        transaction_hash="",
        address=dai_market_name,
        block_hash="",
        block_number=start_block + 5,
    )

    event5 = Event(
        args={
            "reserve": dai_address,
            "onBehalfOf": alice,
            "referral": 0,
            "user": alice,
            "amount": 10_000 * 10 ** DECIMALS[dai_address],
        },
        event="Deposit",
        log_index=199,
        transaction_index=173,
        transaction_hash="",
        address=dai_market_name,
        block_hash="",
        block_number=start_block + 6,
    )

    event6 = Event(
        args={
            "reserve": dai_address,
            "onBehalfOf": alice,
            "referral": 0,
            "user": alice,
            "amount": 100_000 * 10 ** DECIMALS[dai_address],
        },
        event="Deposit",
        log_index=199,
        transaction_index=173,
        transaction_hash="",
        address=dai_market_name,
        block_hash="",
        block_number=start_block + 6,
    )

    events = [event1, event2, event3, event4, event5, event6]
    manager = SimulationManager(plf_env, market_data, oracle_feeds)

    with patch.object(SimulationManager, "execute_event") as execute_event:
        manager.execute_blocks(
            start_block=start_block, end_block=start_block + 10, events=iter(events)
        )
        execute_event.assert_has_calls([call(event) for event in events])
        assert execute_event.call_count == len(events)
