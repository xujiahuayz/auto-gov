import requests
import json
from pprint import pprint


def query_structurer(series: str, spec: str, arg: str = "") -> str:
    # format query arguments
    if arg != "":
        arg = "(" + arg + ")"

    # format query content
    q = series + arg + "{" + spec + "}"
    return q


def graphdata(*q, url: str):
    # pack all subqueries into one big query concatenated with linebreak '\n'
    query = "{" + "\n".join(q) + "}"

    # pretty print out query
    pprint(query)

    r = requests.post(url, json={"query": query})

    response_json = json.loads(r.text)
    return response_json


# # doesn't seem to need this function anymore
# def confighistory(parname: 'str', jsondata, rootparaname: 'str' = None):
#     if rootparaname is None:
#         rootparaname = parname
#     series = [int(jsondata[rootparaname])]
#     series.extend(int(w[parname]) for w in jsondata['configurationHistory'])
#     return series
