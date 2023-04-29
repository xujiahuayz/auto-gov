from decimal import Decimal
import json
import pickle
from calendar import timegm
from datetime import datetime
from os import path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from aavegov.analyzer import getConfig, getMarketMovement
from aavegov.fetcher import series
from aavegov.pricegetter import getPriceDF
from aavegov.utils import datafolder, ratings, safe_logit

# from scipy.stats import beta

# def plotConfig(entry, figdim=(7, 5)):
with open(datafolder + "aavedata.json") as json_file:
    data = json.load(json_file)

reserve_hist = pickle.load(open(datafolder + "reservepara.pkl", "rb"))


reserve_data = data[series]

# entrys = [x['symbol'] for x in reserve_data]


# entry = reserve_data[14]


def dateToInt(date):
    return datetime.utcfromtimestamp(timegm(date.timetuple()))


def get_entry(symbol):
    return [e for e in reserve_data if e["symbol"] == symbol][0]


def normalize_amount(df, col):
    return df.apply(
        lambda row: Decimal(row[f"{col}Amount"])
        / Decimal(10 ** row[f"{col}Reserve.decimals"]),
        axis=1,
    ).astype("float64")


def create_liquidations_df(filepath=path.join(datafolder, "liquidation.pkl")):
    with open(filepath, "rb") as f:
        liquidations = pickle.load(f)

    liquidations_df = pd.json_normalize(liquidations)
    liquidations_df["collateralAmountNormalized"] = normalize_amount(
        liquidations_df, "collateral"
    )
    liquidations_df["principalAmountNormalized"] = normalize_amount(
        liquidations_df, "principal"
    )

    liquidations_df["date"] = pd.to_datetime(
        liquidations_df.timestamp, unit="s"
    ).dt.floor("d")

    return liquidations_df


# getMarketMovement("USDC", window=30)

liquidations_df = create_liquidations_df()

# liquidations_df[liquidations_df["collateralReserve.symbol"] == "USDC"].groupby(
#     "date"
# ).agg({"collateralAmountNormalized": "sum"})
# liquidations_df[liquidations_df["principalReserve.symbol"] == "USDC"].groupby(
#     "date"
# ).agg({"principalAmountNormalized": "sum"})


def buildRegTable(entry, window=30):
    symb = entry["symbol"]

    on_col = "day"
    config_pd = getConfig(entry)
    config_pd[on_col] = [dateToInt(w) for w in config_pd.index]
    start_date = min(config_pd.index)
    pricedata_pd = getMarketMovement(symb, window=window).loc[start_date:]
    pricedata_pd[on_col] = [dateToInt(w) for w in pricedata_pd.index]

    # pricedata_pd['Date'] = pricedata_pd.index

    collateral_liquidated = (
        liquidations_df[liquidations_df["collateralReserve.symbol"] == symb]
        .groupby("date")
        .agg({"collateralAmountNormalized": "sum"})
    )
    loan_repaid = (
        liquidations_df[liquidations_df["principalReserve.symbol"] == symb]
        .groupby("date")
        .agg({"principalAmountNormalized": "sum"})
    )

    # liquidations_df.groupby(["date", "collateralReserve.symbol"]).agg(
    #     {"collateralAmountNormalized": "sum"}
    # ).reset_index().set_index("date")

    reg_table_temp = pd.merge_asof(
        left=pricedata_pd.set_index(on_col),
        right=config_pd.set_index(on_col),
        left_index=True,
        right_index=True,
        direction="backward",
    )  # type: ignore

    pricedata_pd = getPriceDF(symb)
    pricedata_pd[on_col] = [dateToInt(w) for w in pricedata_pd.index]
    # print(pricedata_pd)
    reg_table_temp = (
        pd.merge_asof(
            left=reg_table_temp,
            right=pricedata_pd.set_index(on_col),
            left_index=True,
            right_index=True,
            direction="forward",
        )  # type: ignore
        .merge(loan_repaid, how="left", left_index=True, right_index=True)
        .merge(collateral_liquidated, how="left", left_index=True, right_index=True)
        .fillna(0)
    )

    reg_table_temp["principalrepaid"] = (
        reg_table_temp["principalAmountNormalized"] * reg_table_temp["close"]
    )

    reg_table_temp["collateralslashed"] = (
        reg_table_temp["collateralAmountNormalized"] * reg_table_temp["close"]
    )

    reg_table_temp["principalrepaid"] = (
        reg_table_temp["principalrepaid"].rolling(window).mean()
    )

    reg_table_temp["collateralslashed"] = (
        reg_table_temp["collateralslashed"].rolling(window).mean()
    )

    downfactor = 10 ** entry["decimals"]
    reserve_hist_coin = (
        pd.DataFrame(
            [
                {
                    "Date": datetime.fromtimestamp(w["timestamp"]),
                    "utilization": float(w["utilizationRate"]),
                    "total borrows": float(w["totalBorrows"]) / downfactor,
                    "total liquidity": float(w["totalLiquidity"]) / downfactor,
                }
                for w in reserve_hist
                if w["reserve"]["symbol"] == symb
            ]
        )
        .sort_values("Date")
        .set_index("Date")
    )

    reg_table = pd.merge_asof(
        left=reg_table_temp,
        right=reserve_hist_coin,
        left_index=True,
        right_index=True,
        direction="forward",
    )  # type: ignore

    reg_table["symbol"] = symb

    return reg_table


reg_table_nonstable = pd.DataFrame()

reg_table_stable = pd.DataFrame()
stablecoin_list = ["USDT", "USDC", "DAI", "TUSD", "BUSD", "GUSD", "sUSD"]
# and any(x > 0 for x in buildRegTable(entry)['baseLTVasCollateral'])


def log_monetary_value(series):
    return [np.log(w) if w > 0 else None for w in series]


def organizeRegTable(reg_table):
    reg_table["totalLiquidity"] = reg_table["total liquidity"] * reg_table["close"]

    reg_table["collateral_factor"] = reg_table["baseLTVasCollateral"]
    reg_table["unbounded_collateral_factor"] = safe_logit(
        reg_table["baseLTVasCollateral"] / 100
    )

    reg_table["LTminusLTV"] = (
        reg_table["reserveLiquidationThreshold"] - reg_table["baseLTVasCollateral"]
    )
    reg_table["unbounded_LTminusLTV"] = safe_logit(reg_table["LTminusLTV"] / 100)

    reg_table["LDminusLT"] = 10_000 / reg_table["reserveLiquidationBonus"] - (
        reg_table["reserveLiquidationThreshold"]
    )
    reg_table["unbounded_LDminusLT"] = safe_logit(reg_table["LDminusLT"] / 100)

    reg_table["log_liquidity"] = np.log(reg_table["totalLiquidity"])

    reg_table["unbounded_utilization"] = safe_logit(reg_table["utilization"])

    reg_table["log_vol"] = log_monetary_value(reg_table["vol"])
    # reg_table['timeString'] = reg_table['time'].astype(str)

    reg_table["log_principal_repaid"] = log_monetary_value(reg_table["principalrepaid"])
    reg_table["log_collateral_slashed"] = log_monetary_value(
        reg_table["collateralslashed"]
    )

    return reg_table


def create_df():
    reg_table = pd.DataFrame()

    for entry in reserve_data:
        symb = entry["symbol"]
        print(symb)
        if symb == "ETH" or any(
            x == 0 for x in buildRegTable(entry)["baseLTVasCollateral"]
        ):
            continue
        new_entries = buildRegTable(entry)
        new_entries["is_stablecoin"] = int(symb in stablecoin_list)
        reg_table = reg_table.append(new_entries, ignore_index=True)
    return organizeRegTable(reg_table)


if __name__ == "__main__":
    reg_table = create_df()
    risk_df = pd.read_excel(
        path.join(datafolder, "aaveRiskmatrix.xlsx"), header=0, index_col=0
    )
    reg_table = reg_table.join(risk_df, on="symbol")

    for col in reg_table.columns:
        if "risk" in col.lower():
            reg_table[col.replace(" ", "_").lower()] = reg_table[col].apply(
                lambda x: ratings.index(x) if isinstance(x, str) else x
            )

    reg_table.to_pickle(path.join(datafolder, "regtable.pkl"))
