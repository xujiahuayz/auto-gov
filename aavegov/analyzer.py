import json
from datetime import date, datetime
from pickle import load

import matplotlib.pyplot as plt
import pandas as pd
from numpy import eye, log
from pandas import DataFrame, Series, json_normalize
from scipy.stats import spearmanr
from seaborn import diverging_palette, heatmap
from sklearn.linear_model import LinearRegression

from aavegov.pricegetter import getPriceDF
from aavegov.utils import EPSILON, datafolder
from market_env.constants import DATA_PATH, FIGURE_PATH

index_field = ["date"]
WINDOW = 7


def getConfig_history_pd(entry):
    config_history = json_normalize(entry["configurationHistory"])
    config_history[index_field[0]] = [
        date.fromtimestamp(w) for w in config_history["timestamp"]
    ]
    config_history_pd = (
        config_history[index_field + config_fields]
        .drop_duplicates(subset=index_field, keep="first")
        .set_index(index_field)
        .astype(int)
    )
    return config_history_pd


def getConfig(entry):
    config_history_pd = getConfig_history_pd(entry)
    config_current = json_normalize(entry)[config_fields].astype(int)
    config_current.index = [date.fromtimestamp(entry["lastUpdateTimestamp"])]

    config_pd = pd.concat([config_current, config_history_pd]).sort_index(
        ascending=True
    )
    return config_pd


def plotConfig(entry, figdim=(7, 5)):
    symb = entry["symbol"]
    config_pd = getConfig(entry)
    # try to catch KeyError
    if symb == "ETH" and (not entry["usageAsCollateralEnabled"]):
        return
    try:
        pricedata_pd = getPriceDF(symb)
    except KeyError:
        return
    # plotting starts
    fig, (ax2, ax1, ax4) = plt.subplots(3, sharex=True, figsize=figdim)

    ax2.set_xlim(date(2019, 12, 1), date(2023, 4, 28))
    ax2.set_title("$\\tt " + symb + "$")

    # upper figure is daily volume + daily volatility
    # draw risk config history
    ax1.set_ylabel("Risk parameters")
    for w in config_fields:
        datapoints = config_pd[w]
        ax1.plot(
            datapoints,
            drawstyle="steps-post",
            label=config_legend[w],
            marker="o",
            alpha=0.8,
        )
        valign = "top" if w == "baseLTVasCollateral" else "bottom"
        for i in range(len(datapoints)):
            ax1.annotate(
                datapoints[i],
                (config_pd.index[i], datapoints[i]),
                va=valign,
                ha="right" if i == 0 else "left",
            )

    ax1.set_ylim(-5, 125)
    # ax1.xaxis.set_ticks_position('top')
    ax1.legend(
        ncol=3,
        bbox_to_anchor=(0.5, 1),
        # title='Risk parameters',
        loc="lower center",
        frameon=False,
    )
    # draw daily volatility on the right y-axis

    ax2.plot(
        log(pricedata_pd["close"] + EPSILON).diff(),
        label="daily log return",
        # color='orange',
        alpha=0.8,
    )
    ax2.set_ylabel("Log return of price in $\\tt ETH$")
    ax2.set_ylim([-0.65, 0.65])
    ax2.legend(bbox_to_anchor=(0, 1), loc="lower left", frameon=False)

    ax3 = ax2.twinx()
    # draw first daily volume
    ax3.bar(
        pricedata_pd.index,
        pricedata_pd["volumeto"],
        label="daily volume",
        color="grey",
        alpha=0.8,
    )
    ax3.set_ylabel("Volume in $\\tt ETH$")
    ax3.legend(bbox_to_anchor=(1, 1), loc="lower right", frameon=False)

    downfactor = 10 ** entry["decimals"]
    reserve_hist_coin = DataFrame(
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

    ax4.plot(
        reserve_hist_coin["Date"],
        reserve_hist_coin["utilization"],
        label="actual utilization",
        alpha=0.7,
        c="red",
    )
    ax4.axhline(
        y=int(entry["optimalUtilisationRate"]) / 1e27,
        label="optimal utilization",
        alpha=0.7,
        c="olive",
    )
    ax4.set_ylim([0, 1])
    ax4.set_ylabel("Utilization ratio")
    ax4.legend(bbox_to_anchor=(0, 1), loc="lower left", frameon=False)

    ax5 = ax4.twinx()

    reserve_hist_coin.plot.area(
        x="Date",
        y=["total borrows", "total liquidity"],
        ax=ax5,
        stacked=False,
        legend=False,
    )
    ax5.legend(bbox_to_anchor=(0.5, 1), loc="lower left", frameon=False)
    ax5.set_ylabel("Quantity in token units")

    # make sure x-axis label is not overlapped
    ax4.xaxis.set_tick_params(rotation=15)

    plt.tight_layout()
    plt.savefig(FIGURE_PATH / f"ps_{symb}.pdf")
    # show plot and then close it
    plt.show()
    plt.close()


def getMarketMovement(symbol: str, window: int = WINDOW):
    pricedata_pd = getPriceDF(symbol)
    monthly = DataFrame()
    # log price return standard deviation
    monthly["std"] = log(pricedata_pd["close"] + EPSILON).diff().rolling(window).std()
    monthly["vol"] = pricedata_pd["volumeto"].rolling(window).mean()
    return monthly


def getRiskrelation(entry, window: int = WINDOW):
    symb = entry["symbol"]
    config_history_pd = getConfig_history_pd(entry)
    monthly = getMarketMovement(symbol=symb, window=window)
    risk_relation_rows = config_history_pd.join(monthly)
    risk_relation_rows["symbol"] = symb
    return risk_relation_rows


if __name__ == "__main__":
    with open(datafolder + "aavedata.json") as json_file:
        data = json.load(json_file)

    reserve_data = data["reserves"]
    config_fields = [
        "reserveLiquidationBonus",
        "reserveLiquidationThreshold",
        "baseLTVasCollateral",
    ]

    # define corresponding legend label for plot
    config_legend = Series(
        ["liquidation incentive", "liquidation threshold", "collateral factor"],
        index=config_fields,
    )

    reserve_hist = load(open(datafolder + "reservepara.pkl", "rb"))

    for entry in reserve_data:
        symb = entry["symbol"]
        plotConfig(entry=entry, figdim=(7, 5))

    risk_relation_df = DataFrame()
    for entry in reserve_data:
        symb = entry["symbol"]
        if symb != "ETH":
            try:
                risk_relation_df_rows = getRiskrelation(entry)
            except KeyError:
                # next entry
                continue
            risk_relation_df = risk_relation_df.append(
                risk_relation_df_rows, ignore_index=True
            )

    colname = config_legend.to_dict()
    colname.update(
        {"std": f"{WINDOW}-day volatility", "vol": f"{WINDOW}-day average volume"}
    )
    risk_relation_df = risk_relation_df.rename(colname, axis=1)
    # run a simple regression to get coefficient of 7-day volatility for collateral factor
    reg = LinearRegression().fit(
        risk_relation_df[["collateral factor"]] / 100 - 0.75,
        risk_relation_df["7-day volatility"],
    )
    reg.coef_[0]
    # 0.08016361406433781
    risk_relation_corr = risk_relation_df.corr(method="spearman")
    # save risk_relation_corr to excel
    risk_relation_corr.to_excel(DATA_PATH / "risk_relation_corr.xlsx")

    pval = risk_relation_df.corr(method=lambda x, y: spearmanr(x, y)[1]) - eye(
        *risk_relation_corr.shape
    )
    p = pval.applymap(lambda x: "".join(["*" for t in [0.01, 0.05, 0.1] if x <= t]))

    heatmap(
        risk_relation_corr,
        vmin=-1,
        vmax=1,
        cmap=diverging_palette(220, 20, as_cmap=True),
        annot="$" + risk_relation_corr.round(3).astype(str) + "$" + p,
        fmt="",
    )

    plt.tight_layout()
    plt.show()

    # plt.savefig("./figures/aave_corr.pdf")
