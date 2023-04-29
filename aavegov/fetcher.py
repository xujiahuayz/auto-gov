import json
from pickle import dump


from aavegov.graphqueries import graphdata, query_structurer
from aavegov.pricegetter import getPriceHistory
from aavegov.utils import datafolder


BATCH_SIZE = 1000
queryurl = "https://api.thegraph.com/subgraphs/name/aave/protocol-multy-raw"

series = "reserves"
specs = """
        aToken {
        id
        # underlyingAssetAddress
        underlyingAssetDecimals
        }
        id
        symbol
        name
        decimals
        usageAsCollateralEnabled
        borrowingEnabled
        isActive
        reserveInterestRateStrategy
        optimalUtilisationRate
        variableRateSlope1
        variableRateSlope2
        stableRateSlope1
        stableRateSlope2
        baseVariableBorrowRate
        baseLTVasCollateral
        reserveLiquidationThreshold
        reserveLiquidationBonus
        lastUpdateTimestamp
        utilizationRate
        totalLiquidity
        totalLiquidityAsCollateral
        availableLiquidity
        totalBorrows
        totalBorrowsStable
        totalBorrowsVariable
        liquidityRate
        variableBorrowRate
        stableBorrowRate
        averageStableBorrowRate
        lifetimeLiquidity
        lifetimeBorrows
        lifetimeBorrowsStable
        lifetimeBorrowsVariable
        lifetimeRepayments
        lifetimeWithdrawals
        lifetimeLiquidated
        lifetimeFeeOriginated
        lifetimeFeeCollected
        lifetimeFlashLoans
        lifetimeFlashloanDepositorsFee
        lifetimeFlashloanProtocolFee
        configurationHistory (orderBy: timestamp, orderDirection: desc) {
            id
            usageAsCollateralEnabled
            borrowingEnabled
            stableBorrowRateEnabled
            isActive
            reserveInterestRateStrategy
            baseLTVasCollateral
            reserveLiquidationThreshold
            reserveLiquidationBonus
            timestamp
        }
    """
reserves_query = query_structurer(
    series,
    specs
    # , arg='first: 1'
)

series2 = "atokens"
specs2 = """
        id
        underlyingAssetAddress
        underlyingAssetDecimals
    """

atoken_query = query_structurer(series2, specs2)


series3 = "reserveParamsHistoryItems"
specs3 = """
    reserve{
      symbol
    }
    variableBorrowRate
    variableBorrowIndex
    utilizationRate
    stableBorrowRate
    averageStableBorrowRate
    liquidityIndex
    liquidityRate
    totalLiquidity
    totalLiquidityAsCollateral
    availableLiquidity
    totalBorrows
    totalBorrowsVariable
    totalBorrowsStable
    timestamp
"""

series4 = "liquidationCalls"
specs4 = """
id
pool {
  lendingPool
}
user { id }
collateralReserve { symbol, decimals }
collateralAmount
principalReserve { symbol, decimals }
principalAmount
liquidator
timestamp
"""


# "The `first` argument must be between 0 and 1000, but is 2000
if __name__ == "__main__":
    data = graphdata(reserves_query, atoken_query, url=queryurl)["data"]

    # save json data just in case
    with open(datafolder + "aavedata.json", "w") as f:
        json.dump(data, f, indent=4)

    # fetch token price from crypto compare
    cryptocompare_baseurl = "https://min-api.cryptocompare.com/data/v2/histoday?"
    reserve_data = data[series]
    for entry in reserve_data:
        getPriceHistory(
            fromcoin=entry["symbol"],
            baseurl=cryptocompare_baseurl,
            limit=2000,
            tocoin="ETH",
        )

    last_ts = 0
    data_liquidation = []
    while True:
        liquidation_query = query_structurer(
            series4,
            specs4,
            arg=f"first: {BATCH_SIZE}, orderBy: timestamp, orderDirection: asc, where: {{ timestamp_gt: {last_ts}}}",
        )
        res = graphdata(liquidation_query, url=queryurl)
        if "data" in set(res) and res["data"][series4]:
            rows = res["data"][series4]
            data_liquidation.extend(rows)
            last_ts = rows[-1]["timestamp"]
        else:
            break

    dump(data_liquidation, open(datafolder + "liquidation.pkl", "wb"))

    last_ts = 0
    data_reservepara = []
    while True:
        reservepara_query = query_structurer(
            series3,
            specs3,
            arg=f"first: {BATCH_SIZE}, orderBy: timestamp, orderDirection: asc, where: {{ timestamp_gt: {last_ts}}}",
        )
        res = graphdata(reservepara_query, url=queryurl)
        if "data" in set(res) and res["data"][series3]:
            rows = res["data"][series3]
            data_reservepara.extend(rows)
            last_ts = rows[-1]["timestamp"]
        else:
            break

    dump(data_reservepara, open(datafolder + "reservepara.pkl", "wb"))
