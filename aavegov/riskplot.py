from riskmodel import *
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

params = {
    "legend.fontsize": "large",
    "figure.figsize": (4.5, 5),
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
pylab.rcParams.update(params)


def draw_lines_fron_origin():
    plt.axhline(y=0, color="grey", ls="--", lw=0.7)
    plt.axvline(x=0, color="grey", ls="--", lw=0.7)


# collateral factor
downRisk_values = np.arange(0, 9, 0.01)

for k_value in [-0.5, -1, -2, -3]:
    cf = np.vectorize(collateralFactor)(downRisk_values, k=k_value)
    plt.plot(downRisk_values, cf, label="$" + str(k_value) + "$")

plt.title(
    "$\mathcal{CF} = e^{k \cdot \sigma_{down}}$"
    + ", where "
    + "$k < 0$"
    + " and "
    + "$\sigma_{down} \geq 0$"
)
plt.xlabel("Downside risk of collateral, $\sigma_{down}$")
plt.ylabel("Collateral factor, $\mathcal{CF}$")
plt.legend(title="$k$")
draw_lines_fron_origin()

plt.tight_layout()
plt.savefig("./figures/aave_cf.pdf")
plt.close()

# liquidity bonus
marketVolume_values = np.arange(0, 99, 0.5)

for k_value in [-0.1, -0.5, -1, -2]:
    lb = np.vectorize(liquidityBonus)(marketVolume_values, k=k_value)
    plt.plot(marketVolume_values, lb, label="$" + str(k_value) + "$")

plt.title(
    "$\mathcal{LB} = 0.4 \cdot e^{k \cdot V}$"
    + ", where "
    + "$k < 0$"
    + " and "
    + "$V \geq 0$"
)
plt.xlabel("Daily market volume in ETH, $V$")
plt.ylabel("Liquidity bonus, $\mathcal{LB}$")
plt.legend(title="$k$")
draw_lines_fron_origin()

plt.tight_layout()
plt.savefig("./figures/aave_lb.pdf")
plt.close()

# reserve factor
risk_values = np.arange(0, 9, 0.1)

for k_value in [-0.1, -0.5, -1, -2]:
    rf = np.vectorize(reserveFactor)(risk_values, k=k_value)
    plt.plot(risk_values, rf, label="$" + str(k_value) + "$")

plt.title(
    "$\mathcal{RF} = 0.35 \cdot (1 - e^{k \cdot R})$"
    + ", where "
    + "$k < 0$"
    + " and "
    + "$R \geq 0$"
)
plt.xlabel("Risk, $R$")
plt.ylabel("Reserve factor, $\mathcal{RF}$")
plt.legend(title="$k$")
draw_lines_fron_origin()

plt.tight_layout()
plt.savefig("./figures/aave_rf.pdf")
plt.close()
