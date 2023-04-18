# fetch price from cryptocompare

import requests
import json
from market_env.constants import DATA_PATH


# fetch price from cryptocompare and save to json
def fetch_price(asset_name: str):
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={asset_name}&tsym=ETH&allData=true"
    response = requests.get(url)
    data = json.loads(response.text)
    with open(DATA_PATH / f"{asset_name}.json", "w") as f:
        json.dump(data, f)
    return data


usdc_json = fetch_price("usdc")
link_json = fetch_price("link")
