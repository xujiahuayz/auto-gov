import os
import gzip
import json

from web3 import main

os.environ.setdefault("WEB3_PROVIDER_URI", "http://localhost:8545")

from eth_tools.abi_fetcher import fetch_abi
from eth_tools.contract_caller import ContractCaller
from plf_env.constants import MARKETS, DATA_PATH, BLOCK_BREAKPOINT
from web3.auto.http import w3

ADDRESS = "0xA50ba011c48153De246E5192C8f9258A2ba79Ca9"

abi = fetch_abi(ADDRESS)

addresses = [a["address"] for a in MARKETS]

contract = w3.eth.contract(abi=abi, address=ADDRESS)
contract_caller = ContractCaller(contract)


def get_asset_prices(start_no: int, end_no: int, address_list: list):
    results = contract_caller.collect_results(
        "getAssetsPrices",
        start_block=start_no,
        end_block=end_no,
        block_interval=1,
        contract_args=[address_list],
    )

    with gzip.open(
        os.path.join(DATA_PATH, f"prices_{start_no}_{end_no}.jsonl.gz"), "wt"
    ) as f:
        for result in results:
            print(json.dumps(result), file=f)


if __name__ == "__main__":
    get_asset_prices(
        start_no=BLOCK_BREAKPOINT[0],
        end_no=BLOCK_BREAKPOINT[1] - 1,
        address_list=addresses[:20],
    )
    # get_asset_prices(
    #     start_no=BLOCK_BREAKPOINT[1],
    #     end_no=BLOCK_BREAKPOINT[2],
    #     address_list=addresses,
    # )