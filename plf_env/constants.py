import json
from os import path

from dotenv import load_dotenv

from plf_env.settings import PROJECT_ROOT

load_dotenv()

INITIAL_TIMESTAMP = 1603279128
SECONDS_PER_YEAR = 86400 * 365

MIN_BLOCK = 11_275_902
MAX_BLOCK = 11_976_336

EPSILON = 5e-2
ATOKEN_EPSILON = 1e-2
ATOKEN_ABS_EPSILON = 2e-1

PRICE_FETCH_START = "2020-09-01"

NULL_ADDRESS = "0x0000000000000000000000000000000000000000"

LENDING_RATE_ORACLE_ADDRESS = "0x8A32f49FFbA88aba6EFF96F45D8BD1D4b3f35c7D"
LENDING_ORACLE_CREATION_BLOCK = 11_360_094

ABI_BASE_URL = "http://api.etherscan.io/api?module=contract&action=getabi&address={address}&format=raw"
CRYPTOCOMPARE_BASE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"

DATA_PATH = path.join(PROJECT_ROOT, "data")
BLOCK_INFO_PATH = path.join(DATA_PATH, "blocks-10862180-12168759.csv.gz")
BLOCK_TIMESTAMP_PATH = path.join(DATA_PATH, "block_timestamp.json.gz")
MARKET_PRICE_PATH = path.join(DATA_PATH, "prices_cryptocompare.json")
LENDING_POOL_EVENTS = path.join(DATA_PATH, "aave-v2-events/lending-pool.jsonl.gz")

COIN_MAPPINGS = {"xSUSHI": "SUSHI"}

with open(path.join(DATA_PATH, "interest-rate-params.json")) as f:
    INTEREST_RATES_PARAMS = json.load(f)

MARKETS = [
    {
        "aTokenAddress": "0x3Ed3B47Dd13EC9a98b44e6204A523E766B225811",
        "aTokenSymbol": "aUSDT",
        "stableDebtTokenAddress": "0xe91D55AB2240594855aBd11b3faAE801Fd4c4687",
        "variableDebtTokenAddress": "0x531842cEbbdD378f8ee36D171d6cC9C4fcf475Ec",
        "symbol": "USDT",
        "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "decimals": 6,
    },
    {
        "aTokenAddress": "0x9ff58f4fFB29fA2266Ab25e75e2A8b3503311656",
        "aTokenSymbol": "aWBTC",
        "stableDebtTokenAddress": "0x51B039b9AFE64B78758f8Ef091211b5387eA717c",
        "variableDebtTokenAddress": "0x9c39809Dec7F95F5e0713634a4D0701329B3b4d2",
        "symbol": "WBTC",
        "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "decimals": 8,
    },
    {
        "aTokenAddress": "0x030bA81f1c18d280636F32af80b9AAd02Cf0854e",
        "aTokenSymbol": "aWETH",
        "stableDebtTokenAddress": "0x4e977830ba4bd783C0BB7F15d3e243f73FF57121",
        "variableDebtTokenAddress": "0xF63B34710400CAd3e044cFfDcAb00a0f32E33eCf",
        "symbol": "WETH",
        "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0x5165d24277cD063F5ac44Efd447B27025e888f37",
        "aTokenSymbol": "aYFI",
        "stableDebtTokenAddress": "0xca823F78C2Dd38993284bb42Ba9b14152082F7BD",
        "variableDebtTokenAddress": "0x7EbD09022Be45AD993BAA1CEc61166Fcc8644d97",
        "symbol": "YFI",
        "address": "0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xDf7FF54aAcAcbFf42dfe29DD6144A69b629f8C9e",
        "aTokenSymbol": "aZRX",
        "stableDebtTokenAddress": "0x071B4323a24E73A5afeEbe34118Cd21B8FAAF7C3",
        "variableDebtTokenAddress": "0x85791D117A392097590bDeD3bD5abB8d5A20491A",
        "symbol": "ZRX",
        "address": "0xE41d2489571d322189246DaFA5ebDe1F4699F498",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xB9D7CB55f463405CDfBe4E90a6D2Df01C2B92BF1",
        "aTokenSymbol": "aUNI",
        "stableDebtTokenAddress": "0xD939F7430dC8D5a427f156dE1012A56C18AcB6Aa",
        "variableDebtTokenAddress": "0x5BdB050A92CADcCfCDcCCBFC17204a1C9cC0Ab73",
        "symbol": "UNI",
        "address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xFFC97d72E13E01096502Cb8Eb52dEe56f74DAD7B",
        "aTokenSymbol": "aAAVE",
        "stableDebtTokenAddress": "0x079D6a3E844BcECf5720478A718Edb6575362C5f",
        "variableDebtTokenAddress": "0xF7DBA49d571745D9d7fcb56225B05BEA803EBf3C",
        "symbol": "AAVE",
        "address": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0x05Ec93c0365baAeAbF7AefFb0972ea7ECdD39CF1",
        "aTokenSymbol": "aBAT",
        "stableDebtTokenAddress": "0x277f8676FAcf4dAA5a6EA38ba511B7F65AA02f9F",
        "variableDebtTokenAddress": "0xfc218A6Dfe6901CB34B1a5281FC6f1b8e7E56877",
        "symbol": "BAT",
        "address": "0x0D8775F648430679A709E98d2b0Cb6250d2887EF",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xA361718326c15715591c299427c62086F69923D9",
        "aTokenSymbol": "aBUSD",
        "stableDebtTokenAddress": "0x4A7A63909A72D268b1D8a93a9395d098688e0e5C",
        "variableDebtTokenAddress": "0xbA429f7011c9fa04cDd46a2Da24dc0FF0aC6099c",
        "symbol": "BUSD",
        "address": "0x4Fabb145d64652a948d72533023f6E7A623C7C53",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0x028171bCA77440897B824Ca71D1c56caC55b68A3",
        "aTokenSymbol": "aDAI",
        "stableDebtTokenAddress": "0x778A13D3eeb110A4f7bb6529F99c000119a08E92",
        "variableDebtTokenAddress": "0x6C3c78838c761c6Ac7bE9F59fe808ea2A6E4379d",
        "symbol": "DAI",
        "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xaC6Df26a590F08dcC95D5a4705ae8abbc88509Ef",
        "aTokenSymbol": "aENJ",
        "stableDebtTokenAddress": "0x943DcCA156b5312Aa24c1a08769D67FEce4ac14C",
        "variableDebtTokenAddress": "0x38995F292a6E31b78203254fE1cdd5Ca1010A446",
        "symbol": "ENJ",
        "address": "0xF629cBd94d3791C9250152BD8dfBDF380E2a3B9c",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0x39C6b3e42d6A679d7D776778Fe880BC9487C2EDA",
        "aTokenSymbol": "aKNC",
        "stableDebtTokenAddress": "0x9915dfb872778B2890a117DA1F35F335eb06B54f",
        "variableDebtTokenAddress": "0x6B05D1c608015Ccb8e205A690cB86773A96F39f1",
        "symbol": "KNC",
        "address": "0xdd974D5C2e2928deA5F71b9825b8b646686BD200",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xa06bC25B5805d5F8d82847D191Cb4Af5A3e873E0",
        "aTokenSymbol": "aLINK",
        "stableDebtTokenAddress": "0xFB4AEc4Cc858F2539EBd3D37f2a43eAe5b15b98a",
        "variableDebtTokenAddress": "0x0b8f12b1788BFdE65Aa1ca52E3e9F3Ba401be16D",
        "symbol": "LINK",
        "address": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xa685a61171bb30d4072B338c80Cb7b2c865c873E",
        "aTokenSymbol": "aMANA",
        "stableDebtTokenAddress": "0xD86C74eA2224f4B8591560652b50035E4e5c0a3b",
        "variableDebtTokenAddress": "0x0A68976301e46Ca6Ce7410DB28883E309EA0D352",
        "symbol": "MANA",
        "address": "0x0F5D2fB29fb7d3CFeE444a200298f468908cC942",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xc713e5E149D5D0715DcD1c156a020976e7E56B88",
        "aTokenSymbol": "aMKR",
        "stableDebtTokenAddress": "0xC01C8E4b12a89456a9fD4e4e75B72546Bf53f0B5",
        "variableDebtTokenAddress": "0xba728eAd5e496BE00DCF66F650b6d7758eCB50f8",
        "symbol": "MKR",
        "address": "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xCC12AbE4ff81c9378D670De1b57F8e0Dd228D77a",
        "aTokenSymbol": "aREN",
        "stableDebtTokenAddress": "0x3356Ec1eFA75d9D150Da1EC7d944D9EDf73703B7",
        "variableDebtTokenAddress": "0xcd9D82d33bd737De215cDac57FE2F7f04DF77FE0",
        "symbol": "REN",
        "address": "0x408e41876cCCDC0F92210600ef50372656052a38",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0x35f6B052C598d933D69A4EEC4D04c73A191fE6c2",
        "aTokenSymbol": "aSNX",
        "stableDebtTokenAddress": "0x8575c8ae70bDB71606A53AeA1c6789cB0fBF3166",
        "variableDebtTokenAddress": "0x267EB8Cf715455517F9BD5834AeAE3CeA1EBdbD8",
        "symbol": "SNX",
        "address": "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0x6C5024Cd4F8A59110119C56f8933403A539555EB",
        "aTokenSymbol": "aSUSD",
        "stableDebtTokenAddress": "0x30B0f7324feDF89d8eff397275F8983397eFe4af",
        "variableDebtTokenAddress": "0xdC6a3Ab17299D9C2A412B0e0a4C1f55446AE0817",
        "symbol": "sUSD",
        "address": "0x57Ab1ec28D129707052df4dF418D58a2D46d5f51",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0x101cc05f4A51C0319f570d5E146a8C625198e636",
        "aTokenSymbol": "aTUSD",
        "stableDebtTokenAddress": "0x7f38d60D94652072b2C44a18c0e14A481EC3C0dd",
        "variableDebtTokenAddress": "0x01C0eb1f8c6F1C1bF74ae028697ce7AA2a8b0E92",
        "symbol": "TUSD",
        "address": "0x0000000000085d4780B73119b644AE5ecd22b376",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xBcca60bB61934080951369a648Fb03DF4F96263C",
        "aTokenSymbol": "aUSDC",
        "stableDebtTokenAddress": "0xE4922afAB0BbaDd8ab2a88E0C79d884Ad337fcA6",
        "variableDebtTokenAddress": "0x619beb58998eD2278e08620f97007e1116D5D25b",
        "symbol": "USDC",
        "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "decimals": 6,
    },
    {
        "aTokenAddress": "0x8dAE6Cb04688C62d939ed9B68d32Bc62e49970b1",
        "aTokenSymbol": "aCRV",
        "stableDebtTokenAddress": "0x9288059a74f589C919c7Cf1Db433251CdFEB874B",
        "variableDebtTokenAddress": "0x00ad8eBF64F141f1C81e9f8f792d3d1631c6c684",
        "symbol": "CRV",
        "address": "0xD533a949740bb3306d119CC777fa900bA034cd52",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xD37EE7e4f452C6638c96536e68090De8cBcdb583",
        "aTokenSymbol": "aGUSD",
        "stableDebtTokenAddress": "0xf8aC64ec6Ff8E0028b37EB89772d21865321bCe0",
        "variableDebtTokenAddress": "0x279AF5b99540c1A3A7E3CDd326e19659401eF99e",
        "symbol": "GUSD",
        "address": "0x056Fd409E1d7A124BD7017459dFEa2F387b6d5Cd",
        "decimals": 2,
    },
    {
        "aTokenAddress": "0x272F97b7a56a387aE942350bBC7Df5700f8a4576",
        "aTokenSymbol": "aBAL",
        "stableDebtTokenAddress": "0xe569d31590307d05DA3812964F1eDd551D665a0b",
        "variableDebtTokenAddress": "0x13210D4Fe0d5402bd7Ecbc4B5bC5cFcA3b71adB0",
        "symbol": "BAL",
        "address": "0xba100000625a3754423978a60c9317c58a424e3D",
        "decimals": 18,
    },
    {
        "aTokenAddress": "0xF256CC7847E919FAc9B808cC216cAc87CCF2f47a",
        "aTokenSymbol": "aXSUSHI",
        "stableDebtTokenAddress": "0x73Bfb81D7dbA75C904f430eA8BAe82DB0D41187B",
        "variableDebtTokenAddress": "0xfAFEDF95E21184E3d880bd56D4806c4b8d31c69A",
        "symbol": "xSUSHI",
        "address": "0x8798249c2E607446EfB7Ad49eC89dD1865Ff4272",
        "decimals": 18,
    },
]

DECIMALS = {market["address"]: market["decimals"] for market in MARKETS}
RESERVE_BY_ATOKEN = {market["aTokenAddress"]: market["address"] for market in MARKETS}

ADDRESSES = [a["address"] for a in MARKETS]
ADDRESSES_BY_NAME = {a["symbol"]: a["address"] for a in MARKETS}

BLOCK_BREAKPOINT = [
    11275902,  # https://etherscan.io/tx/0x55609beb6701b676d116a6fc424534e20dc4cbc941ff25d13b4141dd3c403aab
    11371997,  # seems to be able to take all markets starting here
    11999999,
]
