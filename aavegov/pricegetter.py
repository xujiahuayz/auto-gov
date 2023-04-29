import json
from datetime import date
from os import path
from subprocess import PIPE, run
import numpy as np
import pandas as pd

import requests
from aavegov.utils import datapricefolder


INFURAURL = "https://mainnet.infura.io/v3/60fa4da89e6440aea04733cf913dc59a"
ABIFOLDER = "abis/ctoken.json"
rootcommand = "eth-tools call-contract"


def getContractOutput(
    addr: str, func: str, abipath: str = ABIFOLDER, endpoint: str = INFURAURL
):
    command = (
        rootcommand
        + " --abi "
        + abipath
        + " --web3-uri "
        + endpoint
        + " -f "
        + func
        + " "
        + addr
    )
    output = run(command, shell=True, stdout=PIPE).stdout
    return json.loads(output)


# designed to getch data from cryptocompare for atokens denominated in ETH
def getPriceHistory(
    fromcoin: str, baseurl: str, limit: int = 2000, tocoin: str = "ETH"
):
    url = baseurl + "limit=" + str(limit) + "&tsym=" + tocoin + "&fsym=" + fromcoin

    print(url)
    historyJSON = requests.get(url).json()
    with open(path.join(datapricefolder, fromcoin + "-price.json"), "w") as f:
        json.dump(historyJSON, f, indent=4)


def getPriceDF(symb: str):
    time_col = "time"
    if symb == "ETH":
        pricedata_pd = pd.DataFrame(
            # create a list of timestamps from 2015-07-30 to today, one per day
            {
                time_col: pd.date_range(
                    start="2015-07-30", end=date.today(), freq="D"
                ).astype(int)
                / 1e9,
            }
        )
        # make the timestamp int
        pricedata_pd[time_col] = pricedata_pd[time_col].astype(int)
        pricedata_pd["volumeto"] = np.nan
        pricedata_pd["close"] = 1
    else:
        with open(path.join(datapricefolder, symb + "-price.json")) as json_file:
            pricedata_pd = json.load(json_file)["Data"]["Data"]

        pricedata_pd = pd.json_normalize(pricedata_pd)
    pricedata_pd.index = [date.fromtimestamp(w) for w in pricedata_pd[time_col]]
    return pricedata_pd
