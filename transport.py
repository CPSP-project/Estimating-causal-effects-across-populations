import csv
from pathlib import Path
from typing import Union, List, Dict, Any
import pandas as pd

from functions import (
    compute_marginal_probability,
    compute_conditional_probability,
    create_key,
    extract_columns,
)

# ------------------------------------------------------------------------------
# Global in-memory cache to store all computed transport results for plotting
# Maps: x_key → DataFrame
# ------------------------------------------------------------------------------
TRANSPORT_DATA: Dict[str, pd.DataFrame] = {}

# ------------------------------------------------------------------------------
# Robust CSV loader with automatic delimiter detection
# ------------------------------------------------------------------------------
def read_csv_auto(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    path = str(path)
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            sniffed = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        except csv.Error:
            sniffed = type("SniffResult", (), {"delimiter": ","})()
    return pd.read_csv(path, sep=sniffed.delimiter, encoding="utf-8", **kwargs)

# ------------------------------------------------------------------------------
# Helper: format and store a result for later plotting
# ------------------------------------------------------------------------------
def _add_result(container: list, distr: pd.DataFrame, x_row: pd.Series, tag: str):
    """Store transport distribution result and cache it for plotting."""
    table_html = distr[["Y", "P*(Y | do(X))"]].to_html(
        index=False, border=0, float_format="{:.6f}".format
    )
    x_key = f"transport_{tag}_X_{'_'.join(map(str, x_row.values))}"
    TRANSPORT_DATA[x_key] = distr.copy()
    container.append({
        "condition": "first" if tag == "cond1" else "second",
        "x": x_row.to_dict(),
        "x_key": x_key,
        "table": table_html,
    })

# ------------------------------------------------------------------------------
# Transport formula under Condition 1: m-separation
# ------------------------------------------------------------------------------
def _formula_cond1(study_int: pd.DataFrame,
                   target_pop: pd.DataFrame,
                   X_cols: List[str], Z_cols: List[str], Y_cols: List[str],
                   out_list: list):
    p_y_given_xz = (
        study_int.groupby(X_cols + Z_cols + Y_cols).size().reset_index(name="count")
    )
    p_y_given_xz["total"] = p_y_given_xz.groupby(X_cols + Z_cols)["count"].transform("sum")
    p_y_given_xz["P_cond"] = p_y_given_xz["count"] / p_y_given_xz["total"]
    p_y_given_xz = create_key(p_y_given_xz, Z_cols, key_name="Z_key")

    p_z = compute_marginal_probability(target_pop, Z_cols[0])
    p_z = create_key(p_z, Z_cols, key_name="Z_key")
    p_z["P_cond"] = p_z["P"]

    for _, x_row in study_int[X_cols].drop_duplicates().iterrows():
        mask = (p_y_given_xz[X_cols] == x_row.values).all(axis=1)
        p_y_xz = p_y_given_xz[mask]
        if p_y_xz.empty:
            continue
        merged = pd.merge(p_y_xz, p_z, on="Z_key", suffixes=("_y", "_z"))
        merged["term"] = merged["P_cond_y"] * merged["P_cond_z"]
        tot = merged["term"].sum()
        if tot == 0:
            continue
        merged["term"] /= tot

        distr = (
            merged.groupby(Y_cols)["term"].sum()
            .reset_index()
            .rename(columns={"term": "P*(Y | do(X))"})
            .sort_values("Y")
        )
        distr["P_est"] = distr["P*(Y | do(X))"]
        _add_result(out_list, distr, x_row, "cond1")

# ------------------------------------------------------------------------------
# Transport formula under Condition 2: P(Z|X) stability
# ------------------------------------------------------------------------------
def _formula_cond2(study_int: pd.DataFrame,
                   target_pop: pd.DataFrame,
                   X_cols: List[str], Z_cols: List[str], Y_cols: List[str],
                   out_list: list):
    p_y_given_xz = (
        study_int.groupby(X_cols + Z_cols + Y_cols).size().reset_index(name="count")
    )
    p_y_given_xz["total"] = p_y_given_xz.groupby(X_cols + Z_cols)["count"].transform("sum")
    p_y_given_xz["P_cond"] = p_y_given_xz["count"] / p_y_given_xz["total"]
    p_y_given_xz = create_key(p_y_given_xz, Z_cols, key_name="Z_key")

    p_z_given_x = compute_conditional_probability(target_pop, X_cols[0], Z_cols[0])
    p_z_given_x = create_key(p_z_given_x, Z_cols, key_name="Z_key")

    for _, x_row in study_int[X_cols].drop_duplicates().iterrows():
        mz = (p_z_given_x[X_cols] == x_row.values).all(axis=1)
        my = (p_y_given_xz[X_cols] == x_row.values).all(axis=1)
        p_z_x, p_y_xz = p_z_given_x[mz], p_y_given_xz[my]
        if p_z_x.empty or p_y_xz.empty:
            continue
        merged = pd.merge(p_y_xz, p_z_x, on="Z_key", suffixes=("_y", "_z"))
        merged["term"] = merged["P_cond_y"] * merged["P_cond_z"]
        tot = merged["term"].sum()
        if tot == 0:
            continue
        merged["term"] /= tot

        distr = (
            merged.groupby(Y_cols)["term"].sum()
            .reset_index()
            .rename(columns={"term": "P*(Y | do(X))"})
            .sort_values("Y")
        )
        distr["P_est"] = distr["P*(Y | do(X))"]
        _add_result(out_list, distr, x_row, "cond2")

# ------------------------------------------------------------------------------
# Compute true P(Y | do(X)) from the study intervention dataset
# ------------------------------------------------------------------------------
def _store_true_distribution(study_int: pd.DataFrame, X_cols: List[str], Y_cols: List[str]):
    true_df = compute_conditional_probability(study_int, X_cols[0], Y_cols[0])
    for _, x_row in study_int[X_cols].drop_duplicates().iterrows():
        x_key = f"true_X_{'_'.join(map(str, x_row.values))}"
        mask = (true_df[X_cols] == x_row.values).all(axis=1)
        df = true_df[mask][Y_cols + ["P_cond"]].copy().sort_values("Y")
        df = df.rename(columns={"P_cond": "P_true"})
        TRANSPORT_DATA[x_key] = df

# ------------------------------------------------------------------------------
# Main entry point: run transportability analysis and return results
# ------------------------------------------------------------------------------
def run_transport_analysis(
    study_path: Union[str, Path],
    study_intervention_path: Union[str, Path],
    target_path: Union[str, Path],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:

    study_pop = read_csv_auto(study_path)
    study_int = read_csv_auto(study_intervention_path)
    target_pop = read_csv_auto(target_path)

    X_cols, Z_cols, Y_cols = extract_columns(study_pop)
    X, Z, Y = X_cols[0], Z_cols[0], Y_cols[0]

    # Check Condition 1: m-separation
    cond1_df = create_key(
        compute_conditional_probability(study_int, X, Z), Z_cols, key_name="Z_key"
    )
    cond1_ok = all(
        abs(v[:, None] - v).max() <= tolerance
        for v in (
            cond1_df.loc[cond1_df["Z_key"] == z, "P_cond"].values
            for z in cond1_df["Z_key"].unique()
        )
    )

    # Check Condition 2: stability of P(Z|X)
    cond_obs = create_key(
        compute_conditional_probability(study_pop, X, Z), Z_cols, key_name="Z_key"
    )
    cond2_ok = True
    for _, x_row in cond_obs[X_cols].drop_duplicates().iterrows():
        mo = (cond_obs[X_cols] == x_row.values).all(axis=1)
        me = (cond1_df[X_cols] == x_row.values).all(axis=1)
        diff = pd.merge(cond_obs[mo], cond1_df[me], on="Z_key", suffixes=("_o", "_e"))
        if not diff.empty and not (diff["P_cond_o"] - diff["P_cond_e"]).abs().lt(tolerance).all():
            cond2_ok = False
            break

    results_list: List[Dict[str, Any]] = []

    if cond1_ok:
        _formula_cond1(study_int, target_pop, X_cols, Z_cols, Y_cols, results_list)
    if cond2_ok:
        _formula_cond2(study_int, target_pop, X_cols, Z_cols, Y_cols, results_list)

    _store_true_distribution(study_int, X_cols, Y_cols)

    results: Dict[str, Any] = {
        "X_cols": X_cols,
        "Z_cols": Z_cols,
        "Y_cols": Y_cols,
        "conditions": {"first": cond1_ok, "second": cond2_ok},
        "transport_results": results_list,
    }


    if not results_list:
        results["message"] = "❌  Neither condition holds; transport not possible."
    elif cond1_ok and cond2_ok:
        results["message"] = "✅  At least one condition holds."
    elif cond1_ok:
        results["message"] = "✅  At least one condition holds."
    else:
        results["message"] = "✅  At least one condition holds."

    return results
