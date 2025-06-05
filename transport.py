# transport.py
import pandas as pd
import os
from functions import (
    compute_joint_probability,
    compute_marginal_probability,
    compute_conditional_probability,
    create_key,
    extract_columns
)

def run_transport_analysis(study_path, study_intervention_path, target_path, num_x, num_z, num_y, tolerance):
    # Dataset assignments based on corrected naming
    study_pop = pd.read_csv(study_path)                # SP_observed.csv → known population with Y
    study_intervention = pd.read_csv(study_intervention_path)  # SP_control_test.csv → intervention on known pop with Y
    target_pop = pd.read_csv(target_path)              # TP_observed.csv → target population (without Y)

    X_cols, Z_cols, Y_cols = extract_columns(study_pop, num_x, num_z, num_y)

    results = {
        "X_cols": X_cols,
        "Z_cols": Z_cols,
        "Y_cols": Y_cols,
        "conditions": {},
        "transport_results": [],
        "message": ""
    }

    # Conditional probabilities for conditions
    conditional_Z_obs = compute_conditional_probability(study_pop, X_cols, Z_cols)
    conditional_Z_target = compute_conditional_probability(study_intervention, X_cols, Z_cols)
    conditional_Z_obs = create_key(conditional_Z_obs, Z_cols, key_name="Z_key")
    conditional_Z_target = create_key(conditional_Z_target, Z_cols, key_name="Z_key")

    # First condition: P(Z|do(X)) ≈ P(Z|X) from study_intervention
    conditional_Z_interv = compute_conditional_probability(study_intervention, X_cols, Z_cols)
    conditional_Z_interv = create_key(conditional_Z_interv, Z_cols, key_name="Z_key")

    all_match_msep = True
    for z_key in conditional_Z_interv["Z_key"].unique():
        subset = conditional_Z_interv[conditional_Z_interv["Z_key"] == z_key]
        probs = subset["P_cond"].values
        max_diff = abs(probs[:, None] - probs).max()
        if max_diff > tolerance:
            all_match_msep = False
            break
    results["conditions"]["first"] = all_match_msep

    # Second condition: P(Z|X) observed vs study_intervention
    all_match_2 = True
    unique_X = study_intervention[X_cols].drop_duplicates()
    for _, x_row in unique_X.iterrows():
        mask_obs = (conditional_Z_obs[X_cols] == x_row.values).all(axis=1)
        obs_for_x = conditional_Z_obs[mask_obs]
        mask_study = (conditional_Z_target[X_cols] == x_row.values).all(axis=1)
        study_for_x = conditional_Z_target[mask_study]
        if obs_for_x.empty or study_for_x.empty:
            continue
        comp_df = pd.merge(obs_for_x, study_for_x, on="Z_key", suffixes=('_obs', '_study'))
        comp_df["diff"] = (comp_df["P_cond_obs"] - comp_df["P_cond_study"]).abs()
        if not comp_df["diff"].lt(tolerance).all():
            all_match_2 = False
            break
    results["conditions"]["second"] = all_match_2

    if all_match_msep or all_match_2:
        results["message"] = "✅ At least one condition is satisfied."
        # Transport formula: estimate P*(Y | do(X))
        p_y_given_xz = compute_conditional_probability(study_intervention, X_cols + Z_cols, Y_cols)
        p_y_given_xz = create_key(p_y_given_xz, Z_cols, key_name="Z_key")

        p_z_given_x = compute_conditional_probability(target_pop, X_cols, Z_cols)
        p_z_given_x = create_key(p_z_given_x, Z_cols, key_name="Z_key")

        for _, x_row in unique_X.iterrows():
            mask_z = (p_z_given_x[X_cols] == x_row.values).all(axis=1)
            p_z_x = p_z_given_x[mask_z]
            mask_y = (p_y_given_xz[X_cols] == x_row.values).all(axis=1)
            p_y_xz = p_y_given_xz[mask_y]
            if p_z_x.empty or p_y_xz.empty:
                continue
            merged = pd.merge(p_y_xz, p_z_x, on="Z_key", suffixes=('_y', '_z'))
            merged["term"] = merged["P_cond_y"] * merged["P_cond_z"]
            result = (
                merged.groupby(Y_cols)["term"].sum()
                .reset_index()
                .rename(columns={"term": "P*(Y | do(X))"})
            )
            filename = f"transport_result_X_{'_'.join(map(str, x_row.values))}.csv"
            filepath = os.path.join("uploads", filename)
            result.to_csv(filepath, index=False)
            results["transport_results"].append({
                "x": x_row.to_dict(),
                "file": filename,
                "table": result.to_html(index=False, border=0)
            })
    else:
        results["message"] = "❌ Neither condition is satisfied. Cannot apply transport formula."

    return results
