import datetime as dt
import gzip
import json
import logging
from os import path
from typing import Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from plf_env import actions
from plf_env.agent import Agent
from plf_env.caching import cache
from plf_env.computations import round_down_hour
from plf_env.constants import (
    ADDRESSES,
    BLOCK_BREAKPOINT,
    BLOCK_TIMESTAMP_PATH,
    COIN_MAPPINGS,
    DATA_PATH,
    MARKET_PRICE_PATH,
    MARKETS,
    MAX_BLOCK,
    MIN_BLOCK,
    PRICE_FETCH_START,
)
from plf_env.core import Event
from plf_env.env import PLFEnv


@cache(-1)
def organize_market_data(file_path: str = MARKET_PRICE_PATH, window: int = 24 * 7):
    with open(file_path) as f:
        market_price = json.load(f)

    start_timestamp = int(
        dt.datetime.strptime(PRICE_FETCH_START, "%Y-%m-%d").timestamp()
    )
    end_timestamp = round_down_hour(int(dt.datetime.now().timestamp()))

    market_data = pd.DataFrame()
    for market in MARKETS:
        symbol = market["symbol"]
        market_address = market["address"]
        if symbol not in ["ETH", "WETH"]:
            if symbol in COIN_MAPPINGS:
                symbol = COIN_MAPPINGS[symbol]

            coin_price = (
                pd.json_normalize(market_price[symbol])
                .drop_duplicates()
                .sort_values("time")
            )

            # must avoid 0 price or volume being used
            coin_price = coin_price.applymap(lambda x: np.nan if x == 0 else x)
            coin_price["volatility"] = coin_price["close"].rolling(window).std()
            coin_price["volume"] = coin_price["volumeto"].rolling(window).mean()
        else:
            coin_price = pd.DataFrame(
                {
                    "time": np.arange(
                        start=start_timestamp, stop=end_timestamp, step=3600
                    )
                }
            )
            # price always 1
            coin_price["close"] = 1
            coin_price["volatility"] = 0

            usdt_coin_price = (
                pd.json_normalize(market_price["USDT"])
                .drop_duplicates()
                .sort_values("time")
            )

            coin_price["volume"] = usdt_coin_price["volumeto"].rolling(window).mean()

        coin_price["market"] = market_address
        market_data = market_data.append(
            coin_price[["time", "close", "volatility", "volume", "market"]],
            ignore_index=True,
        )
    return market_data.set_index(["time", "market"])


def get_oracle_df(start_no: int, end_no: int, addresses: list):
    oracle_prices = []
    with gzip.open(path.join(DATA_PATH, f"prices_{start_no}_{end_no}.jsonl.gz")) as f:
        for _, w in enumerate(f):
            this_block = json.loads(w)
            oracle_prices.append([this_block[0]] + this_block[1])

    return pd.DataFrame(oracle_prices, columns=["block"] + addresses)


@cache(-1)
def organize_oracle_price():
    df_1 = get_oracle_df(BLOCK_BREAKPOINT[0], BLOCK_BREAKPOINT[1] - 1, ADDRESSES[:20])
    df_2 = get_oracle_df(BLOCK_BREAKPOINT[1], BLOCK_BREAKPOINT[2], ADDRESSES)

    oracle_prices = (
        df_1.append(df_2, ignore_index=True)
        .set_index("block")
        .applymap(lambda x: np.nan if x == 0 else x / 1e18)
    )
    return oracle_prices


# window = 6500 * 7
# market_data = pd.DataFrame()
# for a in ADDRESSES:
#     coin_price = pd.DataFrame(oracle_prices["block"])
#     coin_price["price"] = oracle_prices[a]

#     coin_price["volatility"] = coin_price["price"].rolling(window).std()
#     # coin_price["volume"] = coin_price["volumeto"].rolling(window).mean()
#     market_data = market_data.append(coin_price, ignore_index=True)


class SimulationManager:
    def __init__(
        self,
        plf_env: PLFEnv,
        market_data: pd.DataFrame,
        oracle_feeds: pd.DataFrame,
        agents: List[Agent] = None,
    ):
        if agents is None:
            agents = []

        self.plf_env = plf_env
        self._last_block = MIN_BLOCK - 1
        self.market_data = market_data
        self.oracle_feeds = oracle_feeds
        self.agents = agents
        self._tqdm: Optional[tqdm] = None
        self._last_event: Optional[Event] = None

        with gzip.open(BLOCK_TIMESTAMP_PATH) as f:
            self.block_timestamps = json.load(f)["timestamp"]

    def run_agents(self):
        for agent in self.agents:
            for action in agent.generate_actions():
                action.execute(self.plf_env)
                logging.info(
                    f"executed: {agent.address} action {type(action).__name__}"
                )

    def execute_blocks(self, start_block: int, end_block: int, events: Iterator[Event]):
        if logging.getLogger().level != logging.DEBUG:
            self._tqdm = tqdm(total=end_block - start_block + 1)

        for block_number in range(start_block, end_block + 1):
            self.execute_block(block_number, events)

    def execute_block(self, block_number: int, events: Iterator[Event]):
        self.on_new_block(block_number)

        if self._last_event and self._last_event.block_number == block_number:
            self.execute_event(self._last_event)
            self._last_event = None

        if self._last_event is None:
            self.execute_events(block_number, events)

        self.run_agents()

    def execute_events(self, block_number: int, events: Iterator[Event]):
        for event in events:
            if event.block_number > block_number:
                self._last_event = event
                break

            if event.block_number == block_number:
                self.execute_event(event)

    def execute_event(self, event: Event):
        logging.debug(
            "processing %s %s@%s",
            event.event,
            event.transaction_hash,
            event.block_number,
        )
        action = actions.create_from_event(event)
        if action:
            action.execute(self.plf_env)

    def on_new_block(self, block_number: int):
        current_timestamp = self.block_timestamps[str(block_number)]
        last_block = self._last_block
        self.plf_env.timestamp = current_timestamp
        self.plf_env.block_number = block_number
        self._last_block = block_number

        # update price volatility and volume
        current_hour = round_down_hour(current_timestamp)
        external_states = self.market_data.loc[current_hour].to_dict()
        external_data = self.plf_env.external_data
        external_data.volume = external_states["volume"]
        external_data.volatility = external_states["volatility"]

        external_data.prices = self.oracle_feeds.loc[block_number].to_dict()

        if self._tqdm:
            self._tqdm.update(block_number - last_block)
        # external_states["close"]
