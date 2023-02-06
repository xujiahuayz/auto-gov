import json
import os
from decimal import Decimal
from os import path

from plf_env.settings import PROJECT_ROOT

os.environ.setdefault("WEB3_PROVIDER_URI", "http://localhost:8545")

from plf_env.constants import ADDRESSES, DATA_PATH
from web3.auto.http import w3

SCALE = Decimal(10) ** 27

ADDRESS = "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"

VARIABLES_TO_FETCH = [
    "EXCESS_UTILIZATION_RATE",
    "OPTIMAL_UTILIZATION_RATE",
    "baseVariableBorrowRate",
    "getMaxVariableBorrowRate",
    "stableRateSlope1",
    "stableRateSlope2",
    "variableRateSlope1",
    "variableRateSlope2",
]

with open(path.join(PROJECT_ROOT, "data/abis/aave-v2-lending-pool.json")) as f:
    pool_abi = json.load(f)

with open(path.join(PROJECT_ROOT, "data/abis/aave-interest-rate-strategy.json")) as f:
    interest_rate_model_abi = json.load(f)


pool_contract = w3.eth.contract(abi=pool_abi, address=ADDRESS)


reserves = {}
for address in ADDRESSES:
    print(f"fetching data for {address}")
    reserves[address] = {}
    data = pool_contract.functions.getReserveData(address).call()
    interest_strategy_address = data[-2]
    interest_strategy_contract = w3.eth.contract(
        abi=interest_rate_model_abi, address=interest_strategy_address
    )
    for variable_name in VARIABLES_TO_FETCH:
        func = getattr(interest_strategy_contract.functions, variable_name)
        result = func().call()
        reserves[address][variable_name] = str(Decimal(result) / SCALE)

with open(path.join(DATA_PATH, "interest-rate-params.json"), "w") as f:
    json.dump(reserves, f)
