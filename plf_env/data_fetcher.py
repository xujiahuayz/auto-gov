import datetime as dt
import math

import requests
from eth_tools.event_fetcher import EventFetcher, FetchTask
from tqdm import tqdm
from web3 import Web3
from web3.providers.auto import load_provider_from_environment
from os import path
import pandas as pd


from plf_env.constants import (
    ABI_BASE_URL,
    COIN_MAPPINGS,
    CRYPTOCOMPARE_BASE_URL,
    LENDING_ORACLE_CREATION_BLOCK,
    LENDING_RATE_ORACLE_ADDRESS,
    MARKETS,
)


def fetch_initial_borrow_rates():
    web3 = Web3(load_provider_from_environment())
    event_fetcher = EventFetcher(web3)
    abi = requests.get(ABI_BASE_URL.format(address=LENDING_RATE_ORACLE_ADDRESS)).json()
    task = FetchTask(
        address=LENDING_RATE_ORACLE_ADDRESS,
        abi=abi,
        start_block=LENDING_ORACLE_CREATION_BLOCK,
    )
    events = event_fetcher.fetch_events(task)
    rates = {}
    for event in events:
        if event.get("event") != "MarketBorrowRateSet":
            continue
        rates[event["args"]["asset"]] = event["args"]["rate"] / 10 ** 27  # type: ignore
    return rates


# designed to fetch data from cryptocompare for atokens denominated in ETH
def fetch_price_history(
    from_coin: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    to_coin: str = "ETH",
):
    max_data_points = 2000
    hours_count = int((end_date - start_date).total_seconds()) // 3600
    batch_end = int(end_date.timestamp())
    prices = []
    for _ in range(math.ceil(hours_count / max_data_points)):
        params = {
            "limit": max_data_points,
            "tsym": to_coin,
            "fsym": from_coin,
            "toTs": batch_end,
        }
        price_history = requests.get(CRYPTOCOMPARE_BASE_URL, params=params).json()
        data = price_history["Data"]
        batch_end = data["TimeFrom"]
        prices.extend(data["Data"])
    return prices


def fetch_markets_price_histories(
    start_date: dt.datetime, end_date: dt.datetime, to_coin: str = "ETH"
):
    prices = {}
    for market in tqdm(MARKETS):
        symbol = market["symbol"]
        if to_coin == "ETH" and symbol in ["ETH", "WETH"]:
            continue
        if symbol in COIN_MAPPINGS:
            symbol = COIN_MAPPINGS[symbol]
        prices[symbol] = fetch_price_history(symbol, start_date, end_date, to_coin)
    return prices


def extract_block_timestamp(file_name: str):
    block_data = pd.read_csv(
        file_name,
        compression="gzip",
        header=0,
        usecols=["number", "timestamp"],
        index_col="number",
    )
    return block_data
