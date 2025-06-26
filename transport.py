import os
import csv
from pathlib import Path
from typing import Union
import pandas as pd

# Import utility functions for probability computations and column extraction
from functions import (
    compute_joint_probability,
    compute_marginal_probability,
    compute_conditional_probability,
    create_key,
    extract_columns
)

def read_csv_auto(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read a CSV file while automatically detecting the delimiter from a sample.
    Falls back to comma if detection fails.
    """
    path = str(path)
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            sniffed = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        except csv.Error:
            sniffed = type("SniffResult", (), {"delimiter": ","})()
    encoding = kwargs.pop("encoding", "utf-8")
    return pd.read_csv(path, sep=sniffed.delimiter, encoding=encoding, **kwargs)

def run_transport_analysis(study_path, study_intervention_path, target_path, tolerance):
    # Load datasets from input paths
    study_pop = read_csv_auto(study_path)
    study_intervention = read_csv_auto(study_intervention_path)
    target_pop = read_csv_auto(target_path)

    # Extract column names for X, Z, and Y from the study dataset
    X_cols, Z_cols, Y_cols = extract_columns(study_pop)
    X = X_cols[0]
    Z = Z_cols[0]
    Y = Y_cols[0]

    # Initialize result structure
    results = {
        "X_cols": X_cols,
        "Z_cols": Z_cols,
        "Y_cols": Y_cols,
        "conditions": {},
        "transport_results": [],
        "message": ""
    }

    # === CONDITION 1: m-separation check ===
    # Compute P(Z | X) in both observational and experimental datasets
    conditional_Z_obs = compute_conditional_probability(study_pop, X, Z)
    conditional_Z_target = compute_conditional_probability(study_intervention, X, Z)

    # Add Z_key to both for merging/joining purposes
    conditional_Z_obs = create_key(conditional_Z_obs, Z_cols, key_name="Z_key")
    conditional_Z_target = create_key(conditional_Z_target, Z_cols, key_name="Z_key")

    # Also compute P(Z | X) again using the experimental dataset for m-separation test
    conditional_Z_interv = compute_conditional_probability(study_intervention, X, Z)
    conditional_Z_interv = create_key(conditional_Z_interv, Z_cols, key_name="Z_key")

    # Check maximum variation in P(Z | X) across different values of X
    all_match_msep = True
    for z_key in conditional_Z_interv["Z_key"].unique():
        subset = conditional_Z_interv[conditional_Z_interv["Z_key"] == z_key]
        probs = subset["P_cond"].values
        max_diff = abs(probs[:, None] - probs).max()
        if max_diff > tolerance:
            all_match_msep = False
            break
    results["conditions"]["first"] = all_match_msep

    # === CONDITION 2: stability of P(Z | X) between datasets ===
    all_match_2 = True
    unique_X = study_intervention[X_cols].drop_duplicates()
    for _, x_row in unique_X.iterrows():
        # Match rows for a fixed X value
        mask_obs = (conditional_Z_obs[X_cols] == x_row.values).all(axis=1)
        obs_for_x = conditional_Z_obs[mask_obs]
        mask_study = (conditional_Z_target[X_cols] == x_row.values).all(axis=1)
        study_for_x = conditional_Z_target[mask_study]

        if obs_for_x.empty or study_for_x.empty:
            continue

        # Compare P(Z | X) in the two datasets
        comp_df = pd.merge(obs_for_x, study_for_x, on="Z_key", suffixes=('_obs', '_study'))
        comp_df["diff"] = (comp_df["P_cond_obs"] - comp_df["P_cond_study"]).abs()
        if not comp_df["diff"].lt(tolerance).all():
            all_match_2 = False
            break
    results["conditions"]["second"] = all_match_2

    # Exit early if neither condition is satisfied
    if not (all_match_msep or all_match_2):
        results["message"] = "❌ Neither condition is satisfied. Cannot apply transport formula."
        return results

    results["message"] = "✅ At least one condition is satisfied."

    # === Compute P(Y | X, Z) in the experimental dataset ===
    p_y_given_xz = (
        study_intervention
        .groupby(X_cols + Z_cols + Y_cols)
        .size()
        .reset_index(name="count")
    )
    p_y_given_xz["total"] = p_y_given_xz.groupby(X_cols + Z_cols)["count"].transform("sum")
    p_y_given_xz["P_cond"] = p_y_given_xz["count"] / p_y_given_xz["total"]
    p_y_given_xz = create_key(p_y_given_xz, Z_cols, key_name="Z_key")

    # === Compute marginal P*(Z) in the target population ===
    p_z_given_x_all = compute_marginal_probability(target_pop, Z)
    p_z_given_x_all = create_key(p_z_given_x_all, Z_cols, key_name="Z_key")
    p_z_given_x_all["P_cond"] = p_z_given_x_all["P"]

    # === Transport formula: combine P(Y | X, Z) with P*(Z) ===
    for _, x_row in unique_X.iterrows():
        p_z_x = p_z_given_x_all.copy()
        for col in X_cols:
            p_z_x[col] = x_row[col]

        # Extract matching rows for X = x_row
        mask_y = (p_y_given_xz[X_cols] == x_row.values).all(axis=1)
        p_y_xz = p_y_given_xz[mask_y]
        if p_z_x.empty or p_y_xz.empty:
            continue

        # Join and compute weighted terms
        merged = pd.merge(p_y_xz, p_z_x, on="Z_key", suffixes=('_y', '_z'), how="inner")
        merged["term"] = merged["P_cond_y"] * merged["P_cond_z"]

        print("Z keys in P(Y|X,Z):", set(p_y_xz["Z_key"]))
        print("Z keys in P*(Z):", set(p_z_x["Z_key"]))
        print("MERGED for X =", x_row.values)
        print(merged)

        # Normalize if needed
        total = merged["term"].sum()
        if total > 0:
            merged["term"] = merged["term"] / total

        # Aggregate by Y to obtain P*(Y | do(X))
        result = (
            merged.groupby(Y_cols)["term"].sum()
            .reset_index()
            .rename(columns={"term": "P*(Y | do(X))"})
        )

        # Save results to CSV and HTML
        filename = f"transport_result_X_{'_'.join(map(str, x_row.values))}.csv"
        filepath = os.path.join("uploads", filename)
        result.to_csv(filepath, index=False)
        results["transport_results"].append({
            "x": x_row.to_dict(),
            "file": filename,
            "table": result.to_html(index=False, border=0)
        })

    return results
