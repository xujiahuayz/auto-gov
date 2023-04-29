import pickle
from os import path
import matplotlib.pyplot as plt
import numpy as np

from aavegov.utils import datafolder

with open(path.join(datafolder, "stable-vol-params-results.pkl"), "rb") as f:
    results = pickle.load(f)


volumes = [v[0] for v in results]
collateral_factors = [v[1][1]["collateral_factor"] for v in results]
results[1]

plt.plot(volumes, collateral_factors)
plt.ylim((0, 100))
plt.xlabel("Volume")
plt.ylabel("Collateral factor (%)")
plt.xscale("log", base=10)
plt.show()
