from functions import (
    compute_joint_probability,
    compute_marginal_probability,
    compute_conditional_probability,
    create_key,
    extract_columns
)
import pandas as pd

# Read datasets
study_pop            = pd.read_csv("case_a/SP_observed.csv")
study_intervention         = pd.read_csv("case_a/SP_control_test.csv")
target_pop = pd.read_csv("case_a/TP_observed.csv") #-----we don't have Y here!!!!!!!

# User specifies how many columns are X and Z
num_x = int(input("Enter the number of columns for X: "))
num_z = int(input("Enter the number of columns for Z: ")) 
num_y = int(input("Enter the number of columns for Y: "))

# Automatically extract the column names
X_cols, Z_cols, Y_cols = extract_columns(study_pop, num_x, num_z, num_y)

print("X columns:", X_cols)
print("Z columns:", Z_cols)
print("Y columns:", Y_cols)

###############################################################################
# Compute joint / marginal / conditional for observational and target populations
###############################################################################

# Observational population
joint_obs           = compute_joint_probability(study_pop,    X_cols + Y_cols + Z_cols)
marginal_Z_obs      = compute_marginal_probability(study_pop, Z_cols)
conditional_Z_obs   = compute_conditional_probability(study_pop, X_cols, Z_cols)

# Target (study) population
joint_target        = compute_joint_probability(study_intervention  ,        X_cols + Y_cols + Z_cols)
marginal_Z_target   = compute_marginal_probability(study_intervention  ,     Z_cols)
conditional_Z_target = compute_conditional_probability(study_intervention  , X_cols, Z_cols)

# Intervention population (needed later for transport formula)
joint_interv        = compute_joint_probability(target_pop , X_cols + Z_cols)
marginal_Z_interv   = compute_marginal_probability(target_pop , Z_cols)
conditional_Z_interv = compute_conditional_probability(target_pop , X_cols, Z_cols)

# Create composite key for all Z-based tables
marginal_Z_obs      = create_key(marginal_Z_obs,      Z_cols, key_name="Z_key")
conditional_Z_obs   = create_key(conditional_Z_obs,   Z_cols, key_name="Z_key")
marginal_Z_target   = create_key(marginal_Z_target,   Z_cols, key_name="Z_key")
conditional_Z_target = create_key(conditional_Z_target, Z_cols, key_name="Z_key")
marginal_Z_interv   = create_key(marginal_Z_interv,   Z_cols, key_name="Z_key")
conditional_Z_interv = create_key(conditional_Z_interv, Z_cols, key_name="Z_key")

# Sanity checks
assert "Z_key"  in conditional_Z_obs.columns
assert "P_cond" in conditional_Z_obs.columns
assert "P"      in marginal_Z_target.columns

# ###############################################################################
# # FIRST CONDITION:  P(z_i|do(x_j)) ≃ P(z_i do(x_k)) in the target population
# ###############################################################################

print("\n--- First condition (m-separation): testing if P(Z|do(X)) is invariant across X ---")

# Unici valori di X
unique_X_interv = target_pop[X_cols].drop_duplicates()
tolerance = 0.01
all_match_msep = True

# Calcola P(Z | do(X)) = P(Z | X) nei dati post-intervento
p_z_given_x_df = compute_conditional_probability(study_intervention , X_cols, Z_cols)
p_z_given_x_df = create_key(p_z_given_x_df, Z_cols, key_name="Z_key")

# Per ogni valore di Z, confronta le distribuzioni al variare di X
for z_key in p_z_given_x_df["Z_key"].unique():
    subset = p_z_given_x_df[p_z_given_x_df["Z_key"] == z_key]
    probs = subset["P_cond"].values
    max_diff = abs(probs[:, None] - probs).max()
    
    print(f"Z_key = {z_key} → max diff across X: {max_diff:.4f}")

    if max_diff > tolerance:
        all_match_msep = False

if all_match_msep:
    print("\n✅ First condition holds: P(Z | do(X)) is invariant across all X — m-separation confirmed.")
else:
    print("\n❌ First condition fails: P(Z | do(X)) changes with X — m-separation not confirmed.")


# ###############################################################################
# # SECOND CONDITION: compare P(Z|X)_obs vs P(Z|X)_study
# ###############################################################################

print("\n--- Second condition: compare P(Z|X) observed vs study ---")
all_match_2 = True
unique_X   = study_intervention  [X_cols].drop_duplicates()
for _, x_row in unique_X.iterrows():
    # extract observational and study conditionals for this X
    mask_obs    = (conditional_Z_obs[X_cols]    == x_row.values).all(axis=1)
    obs_for_x   = conditional_Z_obs[mask_obs]
    mask_study  = (conditional_Z_target[X_cols] == x_row.values).all(axis=1)
    study_for_x = conditional_Z_target[mask_study]

    if obs_for_x.empty or study_for_x.empty:
        continue

    print(f"\nFor X = {x_row.to_dict()}:")
    print("  Observed P(Z|X):")
    print(obs_for_x[[*X_cols, "Z_key", "P_cond"]])
    print("  Study   P(Z|X):")
    print(study_for_x[[*X_cols, "Z_key", "P_cond"]])

    comp_df = pd.merge(
        obs_for_x,
        study_for_x,
        on="Z_key",
        suffixes=('_obs', '_study')
    )
    comp_df["diff"]  = (comp_df["P_cond_obs"] - comp_df["P_cond_study"]).abs()
    comp_df["match"] = comp_df["diff"] < tolerance

    # reattach X columns
    for col in X_cols:
        comp_df[col] = x_row[col]

    print("  Comparison (obs vs study):")
    print(comp_df[[*X_cols, "Z_key", "P_cond_obs", "P_cond_study", "diff", "match"]])

    if not comp_df["match"].all():
        all_match_2 = False

if all_match_2:
    print("\n✅ Second condition holds: observed and study P(Z|X) match.")
else:
    print("\n❌ Second condition fails: some observed P(Z|X) differ from study beyond tolerance.")


###############################################################################
# APPLY TRANSPORT FORMULA if one of the two conditions is satisfied
###############################################################################

if all_match_msep or all_match_2:
    print("\n✅ At least one of the two conditions is satisfied.")
    print("→ Computing P*(Y | do(X)) using Equation 11 (Jeffrey conditionalization)...")

    # 1. Calcola P(Y | do(X), Z) ≈ P(Y | X, Z) nei dati studio (study_intervention)
    p_y_given_xz = compute_conditional_probability(study_intervention  , X_cols + Z_cols, Y_cols)
    p_y_given_xz = create_key(p_y_given_xz, Z_cols, key_name="Z_key")

    # 2. Calcola P*(Z | do(X)) ≈ P(Z | X) nei dati target_pop  
    p_z_given_x = compute_conditional_probability(target_pop , X_cols, Z_cols)
    p_z_given_x = create_key(p_z_given_x, Z_cols, key_name="Z_key")

    # 3. Per ogni combinazione di X
    unique_X = study_intervention  [X_cols].drop_duplicates()
    for _, x_row in unique_X.iterrows():
        print(f"\n→ Estimating P*(Y | do(X={x_row.to_dict()}))")

        # Filtra P(Z | X=x)
        mask_z = (p_z_given_x[X_cols] == x_row.values).all(axis=1)
        p_z_x = p_z_given_x[mask_z]

        # Filtra P(Y | X=x, Z=z)
        mask_y = (p_y_given_xz[X_cols] == x_row.values).all(axis=1)
        p_y_xz = p_y_given_xz[mask_y]

        if p_z_x.empty or p_y_xz.empty:
            print("  ⚠️  Skipped: no matching data for this X.")
            continue

        # Merge su Z_key per combinare le due probabilità
        merged = pd.merge(p_y_xz, p_z_x, on="Z_key", suffixes=('_y', '_z'))
        merged["term"] = merged["P_cond_y"] * merged["P_cond_z"]

        # Somma su Z per ogni configurazione di Y
        result = (
            merged.groupby(Y_cols)["term"].sum()
            .reset_index()
            .rename(columns={"term": "P*(Y | do(X))"})
        )

        print("  Estimated P*(Y | do(X)):")
        print(result)

        # Opzionale: salva ogni risultato su file
        filename = f"transport_result_X_{'_'.join(map(str, x_row.values))}.csv"
        result.to_csv(filename, index=False)
        print(f"  🔽 Saved to {filename}")
else:
    print("\n❌ Neither condition is satisfied: transport formula cannot be applied.")

   