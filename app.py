import os
import csv
from pathlib import Path
from typing import Union
import pandas as pd

# Global in-memory transport result cache
from transport import TRANSPORT_DATA     

# Utility functions for computing probabilities and extracting variable roles
from functions import (
    compute_joint_probability,
    compute_marginal_probability,
    compute_conditional_probability,
    create_key,
    extract_columns
)

from transport import run_transport_analysis

# Flask setup and plotting libraries
from flask import Flask, render_template, request, Response
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server rendering
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Initialize Flask app and upload folder
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------------------------------------------------------
# CSV Reader with automatic delimiter detection (comma, semicolon, tab, pipe)
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
    encoding = kwargs.pop("encoding", "utf-8")
    return pd.read_csv(path, sep=sniffed.delimiter, encoding=encoding, **kwargs)

# ------------------------------------------------------------------------------
# Homepage route - displays the file upload form
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ------------------------------------------------------------------------------
# Upload route - saves uploaded files, runs transportability analysis, and shows results
# ------------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    paths = {
        "study":   os.path.join(UPLOAD_FOLDER, "study_population.csv"),
        "intv":    os.path.join(UPLOAD_FOLDER, "study_population_with_intervention.csv"),
        "target":  os.path.join(UPLOAD_FOLDER, "target_population.csv"),
    }

    # Save uploaded CSVs to disk
    request.files["study_population"].save(paths["study"])
    request.files["study_population_with_intervention"].save(paths["intv"])
    request.files["target_population"].save(paths["target"])

    # Extract tolerance and run the main analysis
    tolerance = float(request.form["tolerance"])
    results = run_transport_analysis(
        study_path=paths["study"],
        study_intervention_path=paths["intv"],
        target_path=paths["target"],
        tolerance=tolerance,
    )

    transport_results = results.get("transport_results", [])
    first_x_key = transport_results[0]["x_key"] if transport_results else None

    return render_template(
        "result.html",
        results=results,
        first_x_key=first_x_key,
    )

# ------------------------------------------------------------------------------
# Plot route for the marginal distribution of Z in study vs target
# ------------------------------------------------------------------------------
@app.route("/plot_z_distribution")
def plot_z_distribution():
    study_df  = read_csv_auto(os.path.join(UPLOAD_FOLDER, "study_population.csv"))
    target_df = read_csv_auto(os.path.join(UPLOAD_FOLDER, "target_population.csv"))

    _, Z_cols, _ = extract_columns(study_df)
    z_col = request.args.get("z_col", Z_cols[0])

    # Compute marginal probabilities of Z
    study_pz  = compute_marginal_probability(study_df,  z_col).rename(columns={"P": "P_study"})
    target_pz = compute_marginal_probability(target_df, z_col).rename(columns={"P": "P_target"})

    # Merge and sort for plotting
    merged = (
        pd.merge(study_pz, target_pz, on=z_col, how="outer")
        .fillna(0)
        .sort_values(by=z_col)
    )

    # Create bar plot
    fig, ax = plt.subplots()
    bar_width = 0.2
    x = range(len(merged[z_col]))

    ax.bar(
        [i - bar_width/2 for i in x],
        merged["P_study"],
        width=bar_width,
        color="cornflowerblue",
        label="Pr(Z) study",
        align='center'
    )
    ax.bar(
        [i + bar_width/2 for i in x],
        merged["P_target"],
        width=bar_width,
        color="orange",
        label="Pr*(Z) target",
        align='center'
    )

    # Axis and legend
    ax.set_xticks(x)
    ax.set_xticklabels(merged[z_col])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(z_col)
    ax.set_ylabel("Probability")
    ax.legend()
    fig.tight_layout()

    # Return image as PNG
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")

# ------------------------------------------------------------------------------
# Plot route for conditional distribution of Y given do(X = x)
# ------------------------------------------------------------------------------
@app.route("/plot_y_distribution")
def plot_y_distribution():
    x_key = request.args["x_key"]

    try:
        df_est = TRANSPORT_DATA[x_key].sort_values("Y")

        # Recover correct key for true distribution
        x_suffix = "_".join(x_key.split("_")[3:])         # e.g., "1"
        true_key = f"true_X_{x_suffix}"
        df_true = TRANSPORT_DATA[true_key].sort_values("Y")
    except KeyError:
        return Response(f"Unknown x_key: {x_key}", status=404)

    # Debug print to verify values in server log
    print("=== DEBUG  X =", x_suffix, "===")
    print("true_py   dataframe:")
    print(df_true)

    print("transport_df dataframe:")
    print(df_est)

    # Extract data to plot
    y_vals = df_est["Y"].values
    p_true = df_true["P_true"].values
    p_est  = df_est["P*(Y | do(X))"].values

    # Create bar plot
    bar_width = 0.25
    fig, ax = plt.subplots()

    ax.bar(
        y_vals - bar_width/2,
        p_true,
        width=bar_width,
        color="cornflowerblue",
        label="P(Y | do(X)) study",
    )

    ax.bar(
        y_vals + bar_width/2,
        p_est,
        width=bar_width,
        color="orange",
        label="P*(Y | do(X)) target",
    )

    ax.set_xticks(y_vals)
    ax.set_xlabel("Y")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    # Return plot as PNG response
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")

# ------------------------------------------------------------------------------
# Launch Flask server
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
