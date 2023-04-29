import json
from datetime import date
from os import path
from subprocess import PIPE, run

import requests
from aavegov.utils import datapricefolder
from pandas import json_normalize

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
    with open(path.join(datapricefolder, symb + "-price.json")) as json_file:
        pricedata = json.load(json_file)["Data"]["Data"]

    pricedata_pd = json_normalize(pricedata)
    pricedata_pd.index = [date.fromtimestamp(w) for w in pricedata_pd["time"]]
    return pricedata_pd
