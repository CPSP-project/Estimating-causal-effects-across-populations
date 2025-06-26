# app.py
import os
from io import BytesIO

from flask import Flask, render_template, request, Response
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt

from transport import run_transport_analysis
from functions import extract_columns, compute_marginal_probability
from transport import read_csv_auto  # Utility to auto-detect CSV encoding and read as DataFrame

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload directory exists

# ---------------------------------------------------------------------------
# Home route - renders the upload form
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ---------------------------------------------------------------------------
# Upload route - saves the uploaded CSVs and runs the analysis
# ---------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    # Save uploaded files to the local uploads directory
    paths = {
        "study":   os.path.join(UPLOAD_FOLDER, "study_population.csv"),
        "intv":    os.path.join(UPLOAD_FOLDER, "study_population_with_intervention.csv"),
        "target":  os.path.join(UPLOAD_FOLDER, "target_population.csv"),
    }
    request.files["study_population"].save(paths["study"])
    request.files["study_population_with_intervention"].save(paths["intv"])
    request.files["target_population"].save(paths["target"])

    # Run the transportability analysis
    tolerance = float(request.form["tolerance"])
    results = run_transport_analysis(
        study_path=paths["study"],
        study_intervention_path=paths["intv"],
        target_path=paths["target"],
        tolerance=tolerance,
    )

    # Get the X value key of the first result for dynamic plot rendering
    first_x_key = (
        results["transport_results"][0]["file"]
        .replace("transport_result_X_", "")
        .replace(".csv", "")
    )

    return render_template(
        "result.html",
        results=results,
        first_x_key=first_x_key,  # Passed to the frontend template for initial plot
    )

# ---------------------------------------------------------------------------
# Endpoint to plot the marginal distribution of Z in study and target datasets
# ---------------------------------------------------------------------------

@app.route("/plot_z_distribution")
def plot_z_distribution():
    study_df  = read_csv_auto(os.path.join(UPLOAD_FOLDER, "study_population.csv"))
    target_df = read_csv_auto(os.path.join(UPLOAD_FOLDER, "target_population.csv"))

    _, Z_cols, _ = extract_columns(study_df)
    z_col = request.args.get("z_col", Z_cols[0])

    # Compute marginal probabilities for Z
    study_pz  = compute_marginal_probability(study_df,  z_col).rename(columns={"P": "P_study"})
    target_pz = compute_marginal_probability(target_df, z_col).rename(columns={"P": "P_target"})
    # Merge the marginal distributions of Z from study and target datasets.
    # - `on=z_col` joins the two DataFrames by the common values of the Z variable.
    # - `how="outer"` ensures that all unique Z values from both datasets are included (even if they are missing in one).
    # - `.fillna(0)` replaces missing probability values (NaNs) with 0, assuming zero probability where data is absent.
    # - `.sort_values(by=z_col)` orders the result by the values of Z for consistent plotting.
    merged = (pd.merge(study_pz, target_pz, on=z_col, how="outer") 
                .fillna(0)
                .sort_values(by=z_col))

    # Plot study vs. target distribution of Z
    fig, ax = plt.subplots()
    ax.plot(merged[z_col], merged["P_study"],  marker="o", lw=2, label="Pr(Z) study")
    ax.plot(merged[z_col], merged["P_target"], marker="x", lw=2, linestyle="--", label="Pr*(Z) target")
    ax.set_xlabel(z_col)
    ax.legend()
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")

# ---------------------------------------------------------------------------
# Endpoint to plot the conditional distribution of Y given do(X=x)
# ---------------------------------------------------------------------------

@app.route("/plot_y_distribution")
def plot_y_distribution():
    x_key = request.args.get("x_key", "1")  # Value of X for which to plot do(X=x)
    path_csv = os.path.join(UPLOAD_FOLDER, f"transport_result_X_{x_key}.csv")

    # Read relevant datasets
    intv_df = read_csv_auto(os.path.join(UPLOAD_FOLDER, "study_population_with_intervention.csv"))
    transport_df = read_csv_auto(path_csv)

    X_cols, Z_cols, Y_cols = extract_columns(intv_df)
    X_col, y_col = X_cols[0], Y_cols[0]

    # Filter rows for selected X = x_key
    try:
        x_val = pd.to_numeric(x_key)
    except ValueError:
        x_val = x_key
    intv_subset = intv_df[intv_df[X_col] == x_val]

    # Estimate P(Z | X = x_key) from the subset of the study-intervention dataset
    # - Group by the Z variable and count occurrences
    # - Divide by the total number of rows in the subset (i.e., conditioned on X = x_key)
    # - Convert the resulting series to a dictionary for easy lookup
    p_z_study = (intv_subset.groupby(Z_cols[0]).size()
                            .div(len(intv_subset))
                            .to_dict())

    # Compute P(Y | X = x_key, Z):
    # - Group by both Z and Y to count occurrences
    # - Divide by the group count per Z to get P(Y | Z, X = x_key)
    # - Reset index to create a clean DataFrame with columns: Z, Y, P_cond
    cond = (intv_subset.groupby([Z_cols[0], y_col]).size()
                        .div(intv_subset.groupby(Z_cols[0]).size())
                        .reset_index(name="P_cond"))

    # Marginalize over Z to get P(Y | do(X = x_key)) using the transport formula:
    # P(Y | do(X)) = sum_Z P(Y | X, Z) * P(Z | X)
    # - Assign weight column using the P(Z | X = x_key) values from p_z_study
    # - Multiply conditional probability by weight to get the joint term
    # - Group by Y and sum to obtain the final marginal distribution
    true_py = (cond.assign(weight=lambda d: d[Z_cols[0]].map(p_z_study))
                    .assign(P_true=lambda d: d["P_cond"] * d["weight"])
                    .groupby(y_col, as_index=False)["P_true"].sum())


    # Rename transport distribution column for clarity
    transport_df = transport_df.rename(columns={"P*(Y | do(X))": "P_est"})

    # Convert types for merge
    true_py[y_col]      = true_py[y_col].astype(str)
    transport_df[y_col] = transport_df[y_col].astype(str)

    merged = (true_py
              .merge(transport_df, on=y_col, how="inner")
              .fillna(0)
              .sort_values(by=y_col, key=lambda s: s.astype(float)))

    # Debug logging for verification
    print("\n=== DEBUG  X =", x_key, "===")
    print("true_py   dataframe:")
    print(true_py)

    print("\ntransport_df dataframe:")
    print(transport_df)

    print("\nSum of probabilities (should be ~1):")
    print("Study estimate (blue):", true_py['P_true'].sum())
    print("Target estimate (orange):", transport_df['P_est'].sum())
    print("=====================================\n")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(merged[y_col], merged["P_true"],
            marker="o", lw=2,
            label="Pr(Y | do(X)) study")  # Blue curve

    ax.plot(merged[y_col], merged["P_est"],
            marker="x", lw=2, linestyle="--",
            label="Pr*(Y | do(X)) target")  # Orange curve

    ax.set_xlabel(y_col)
    ax.legend()
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")

# ---------------------------------------------------------------------------
# Run the Flask application
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
