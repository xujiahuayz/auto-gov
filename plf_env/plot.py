import datetime as dt
import gzip
import json
import pickle
from typing import Optional

import matplotlib.pyplot as plt

from plf_env.agent import Agent
from plf_env.constants import BLOCK_TIMESTAMP_PATH
from os import path
from plf_env.settings import PROJECT_ROOT


def output_plot(output: Optional[str]):
    if output:
        plt.savefig(output)
    else:
        plt.show()


def plot_agent_wealths(persisted_state: dict, output: Optional[str] = None):
    with gzip.open(BLOCK_TIMESTAMP_PATH) as f:
        block_timestamps = json.load(f)["timestamp"]

    agents: list[Agent] = persisted_state["agents"]
    for agent in agents:
        sorted_wealths = sorted(agent.wealths.items(), key=lambda x: x[0])
        xs = [
            dt.datetime.fromtimestamp(block_timestamps[str(b)])
            for b, _ in sorted_wealths
        ]
        ys = [v for _, v in sorted_wealths]
        plt.plot(xs, ys, label=agent.address)

    plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Wealth (ETH)")
    plt.tight_layout()

    output_plot(output)


def plot(
    name: str,
    state_filename: str,
    output: Optional[str] = path.join(PROJECT_ROOT, "figures/agents.pdf"),
    **kwargs
):
    func_name = "plot_{0}".format(name.replace("-", "_"))
    func = globals()[func_name]
    with open(state_filename, "rb") as f:
        state = pickle.load(f)
    func(state, output=output, **kwargs)
