import datetime as dt
import json
import logging
import pickle
from argparse import ArgumentParser, ArgumentTypeError
from plf_env.plot import plot


from plf_env import data_fetcher
from plf_env.computations import EventFilesWrapper
from plf_env.constants import (
    BLOCK_INFO_PATH,
    BLOCK_TIMESTAMP_PATH,
    MARKET_PRICE_PATH,
    MAX_BLOCK,
    MIN_BLOCK,
    PRICE_FETCH_START,
)
from plf_env.env import PLFEnv
from plf_env.simulation_manager import (
    SimulationManager,
    organize_market_data,
    organize_oracle_price,
)
from plf_env.settings import LOG_FORMAT
from plf_env.simulation import create_agents


def date_type(date_string: str) -> dt.datetime:
    """Parses a date and returns it as a ``datetime.datetime```
    Raises a ``argparse.ArgumentTypeError`` if it fails
    """
    try:
        return dt.datetime.strptime(date_string, "%Y-%m-%d")
    except ValueError as ex:
        msg = f"{date_string} is not a valid date. Expected format, YYYY-MM-DD"
        raise ArgumentTypeError(msg) from ex


parser = ArgumentParser(prog="plf-env", description="CLI for plf-env")
parser.add_argument(
    "--debug", default=False, action="store_true", help="Enables debug logging"
)

subparsers = parser.add_subparsers(dest="command")

fetch_initial_borrow_rates_parser = subparsers.add_parser(
    "fetch-initial-borrow-rates", help="Fetches the initial borrow rates of all markets"
)
fetch_initial_borrow_rates_parser.add_argument(
    "-o", "--output", required=True, help="Python file where to save the rates"
)

fetch_market_data_parser = subparsers.add_parser(
    "fetch-market-prices", help="Fetch prices for markets from CryptoCompare"
)
fetch_market_data_parser.add_argument(
    "-o",
    "--output",
    default=MARKET_PRICE_PATH,
    help="JSON file where to save the prices",
)
fetch_market_data_parser.add_argument(
    "--start-date",
    type=date_type,
    help="Start date to fetch prices formatted as YYYY-MM-DD",
    default=PRICE_FETCH_START,
)
fetch_market_data_parser.add_argument(
    "--end-date",
    type=date_type,
    help="End date to fetch prices YYYY-MM-DD",
    default=dt.datetime.now(),
)
fetch_market_data_parser.add_argument(
    "--to-coin", default="ETH", help="Quote currency for prices"
)

fetch_block_timestamps_parser = subparsers.add_parser(
    "fetch-block-timestamps", help="Fetch timestamp of blocks included in a csv.gz file"
)


fetch_block_timestamps_parser.add_argument(
    "--file-name",
    type=str,
    help="Path to the block information file",
    default=BLOCK_INFO_PATH,
)

fetch_block_timestamps_parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="JSON file where to same the timestamps",
    default=BLOCK_TIMESTAMP_PATH,
)


process_events_parser = subparsers.add_parser(
    "process-events", help="Processes the events through the simulator"
)

process_events_parser.add_argument("inputs", nargs="*", help="Files containing events")
process_events_parser.add_argument(
    "--include-agents",
    default=False,
    action="store_true",
    help="Include external agents",
)
process_events_parser.add_argument("-s", "--start-block", type=int, default=MIN_BLOCK)
process_events_parser.add_argument("-e", "--end-block", type=int, default=MAX_BLOCK)

process_events_parser.add_argument(
    "-o", "--output", help="Output pickle file of agents' history"
)

plot_parser = subparsers.add_parser("plot", help="Plots results")
plot_parser.add_argument("-o", "--output", help="Output plot")
plot_parser.add_argument("-s", "--state", required=True, help="Path to state file")
plot_subparsers = plot_parser.add_subparsers(dest="plot_name")
agent_wealths_parser = plot_subparsers.add_parser("agent-wealths")


def run_fetch_initial_borrow_rates(args):
    rates = data_fetcher.fetch_initial_borrow_rates()
    with open(args["output"], "w") as f:
        output = "RATES = {0}".format(json.dumps(rates, indent=4))
        print(output, file=f)


def run_fetch_market_data(args):
    prices = data_fetcher.fetch_markets_price_histories(
        args["start_date"], args["end_date"], to_coin=args["to_coin"]
    )
    with open(args["output"], "w") as f:
        json.dump(prices, f)


def run_fetch_block_timestamps(args):
    block_data = data_fetcher.extract_block_timestamp(file_name=args["file_name"])
    block_data.to_json(args["output"], indent=4)


def run_process_events(args):
    market_data = organize_market_data()
    oracle_feeds = organize_oracle_price()

    plf_env = PLFEnv()
    agents = []
    if args["include_agents"]:
        agents = create_agents(plf_env)
    manager = SimulationManager(
        plf_env,
        market_data=market_data,
        oracle_feeds=oracle_feeds,
        agents=agents,
    )
    with EventFilesWrapper(args["inputs"]) as events:
        manager.execute_blocks(args["start_block"], args["end_block"], events)

    if args["output"]:
        with open(args["output"], "wb") as f:
            pickle.dump({"env": plf_env, "agents": agents}, f)


def run_plot(args):
    plot_name = args.pop("plot_name", None)
    if not plot_name:
        plot_parser.error("plot name not given")
    state = args.pop("state")
    output = args.pop("output")
    plot(plot_name, state, output, **args)


def run():
    logging_level = logging.INFO
    args = vars(parser.parse_args())
    if args.pop("debug", False):
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format=LOG_FORMAT)
    command = args.pop("command", None)
    if not command:
        parser.error("no command given")
    func_name = "run_{0}".format(command.replace("-", "_"))
    func = globals()[func_name]
    func(args)


# Run it only if called from the command line
if __name__ == "__main__":
    run()
