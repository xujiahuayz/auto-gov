import pickle
from os import path

import statsmodels.formula.api as smf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from aavegov.utils import datafolder

COLUMNS_TO_REGRESS = [
    "log_liquidity",
    "unbounded_utilization",
    "log_principal_repaid",
    "log_collateral_slashed",
]


def run_regression(reg_table, formula, filename):
    lm = smf.ols(formula=formula, data=reg_table)
    lm_results = lm.fit()
    print(lm_results.summary())
    lm_results.save(path.join(datafolder, "models", filename))
    return lm_results


def regress_series(reg_table, table_type, target_variable, main_formula, res_formula):
    res_liqd = f"{target_variable} ~ {main_formula}"
    filename = f"{target_variable}-estimate-results-{table_type}.pkl"
    lm_results = run_regression(reg_table, res_liqd, filename)

    target_residuals = f"{target_variable}_residuals"
    reg_table[target_residuals] = lm_results.resid

    target_formula = f"{target_residuals} ~ {res_formula}"
    filename = f"{target_variable}-residuals-estimate-results-{table_type}.pkl"
    run_regression(reg_table, target_formula, filename)


def compute_regressions(reg_table, table_type, main_formula, res_formula):
    print(f"computing regressions for {table_type}")

    for column in COLUMNS_TO_REGRESS:
        print("<>" * 50)
        print(f"Regressing {column}")
        regress_series(reg_table, table_type, column, main_formula, res_formula)
        print()


def describe_table(reg_table, reg_table_nonstable, reg_table_stable):
    reg_table["vol"].describe()

    reg_table_nonstable["vol"].describe()
    reg_table_nonstable["std"].describe()
    reg_table_nonstable["totalLiquidity"].describe()
    reg_table_nonstable["utilization"].describe()

    reg_table_stable["vol"].describe()
    reg_table_stable["std"].describe()


def main():
    reg_table = pickle.load(open(path.join(datafolder, "regtable.pkl"), "rb"))

    reg_table_stable = reg_table[reg_table["is_stablecoin"] == 1]
    reg_table_nonstable = reg_table[reg_table["is_stablecoin"] == 0]

    # describe_table(reg_table, reg_table_nonstable, reg_table_stable)

    # compute_regressions(
    #     reg_table.copy(),
    #     "full",
    #     "log_vol + std",
    #     "collateral_factor + collateral_factor:is_stablecoin + LDminusLT + LTminusLTV - 1",
    # )

    input_columns = [
        "log_vol",
        "std",
        "collateral_factor",
        "is_stablecoin",
        "LDminusLT",
        "LTminusLTV",
    ]

    filtered_table = reg_table[input_columns + COLUMNS_TO_REGRESS]
    filtered_table = filtered_table.dropna()
    X = filtered_table[input_columns].to_numpy()

    pipelines = {}
    for column in COLUMNS_TO_REGRESS:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", MLPRegressor(max_iter=1000, hidden_layer_sizes=(100,))),
            ]
        )
        y = filtered_table[column].to_numpy()
        pipeline.fit(X, y)
        print(f"R2 for {column}", pipeline.score(X, y))
        pipelines[column] = pipeline

    with open(path.join(datafolder, "models", "mlp-overfitting.pkl"), "wb") as f:
        pickle.dump(pipelines, f)


if __name__ == "__main__":
    main()

# # reg_util = "utilization ~ std + vol + baseLTVasCollateral + reserveLiquidationBonus + reserveLiquidationThreshold -1"
# reg_unbounded_util = "unbounded_utilization ~ std + vol + is_stablecoin + baseLTVasCollateral + LDminusLT + LTminusLTV - 1"
# lm = smf.ols(formula=reg_unbounded_util, data=reg_table)
# lm_results = lm.fit()
# lm_results.summary()

# reg_util = "utilization ~ std + vol + is_stablecoin + baseLTVasCollateral + LDminusLT + LTminusLTV"
# binom_glm = smf.glm(formula=reg_util, family=sm.families.Binomial(), data=reg_table)
# binom_results = binom_glm.fit()
# binom_results.summary()


# reg_liqd = "log_liquidity ~ std + vol + baseLTVasCollateral + reserveLiquidationBonus + reserveLiquidationThreshold - 1"
# lm = smf.ols(formula=reg_liqd, data=reg_table)
# lm_results = lm.fit()
# lm_results.summary()


# compute_regressions(
#     reg_table_stable.copy(),
#     "stable",
#     "log_vol",
#     "unbounded_collateral_factor + unbounded_LTminusLTV - 1",
# )
# compute_regressions(reg_table_nonstable.copy(), "non-stable")
