import matplotlib.pyplot as plt
from fetcher import *
from datetime import datetime
from pandas import json_normalize
from numpy import log
import matplotlib.pyplot as plt

# not looking at compound for now
ctokens = requests.get("https://api.compound.finance/api/v2/ctoken").json()
ctokens_json = ctokens["cToken"]

# ctokens_db = pd.json_normalize(ctokens_json)


def getdec(add) -> int:
    if add == None:
        dec = 18
    else:
        dec = getContractOutput(add, "decimals")["result"]
    return dec


totalBorrowsCurrent = [
    {
        "underlying": w["underlying_symbol"],
        "totalBorrows": w["total_borrows"]["value"],
        # 'totalBorrowsCurrent': getContractOutput(
        #     w['token_address'], 'totalBorrowsCurrent'
        # )['result']/(10 ** getdec(w['underlying_address']))  # need be careful with decimals, not all are 18
    }
    for w in ctokens_json
]


# for i in range(len(ctokens_db)):
#     tkn = ctokens_json[0]['underlying_symbol']
#     tkn_add = ctokens_json[0]['token_address']
#     getContractOutput(tkn_add, 'totalBorrowsCurrent')
