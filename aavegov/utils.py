from os import path

import numpy as np
import pandas as pd
from scipy.special import logit

INFURAURL = "https://mainnet.infura.io/v3/60fa4da89e6440aea04733cf913dc59a"
ABIFOLDER = "abis/ctoken.json"
rootcommand = "eth-tools call-contract"
datafolder = path.join(path.dirname(__file__), "data/")
datapricefolder = path.join(datafolder, "price")

EPSILON = 1e-8

ratings = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+"]


config_fields = [
    "reserveLiquidationBonus",
    "reserveLiquidationThreshold",
    "baseLTVasCollateral",
]

# define corresponding legend label for plot
config_legend = pd.Series(
    ["liquidation incentive", "liquidation threshold", "collateral factor"],
    index=config_fields,
)


def safe_logit(value, epsilon=EPSILON):
    return logit(np.clip(value, epsilon, 1 - epsilon))
