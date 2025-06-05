import os
from io import BytesIO
from flask import Flask, render_template, request, Response
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usa backend non-interattivo per server
import matplotlib.pyplot as plt
from transport import run_transport_analysis
from functions import extract_columns, compute_marginal_probability  # include marginale

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creazione della cartella uploads se non esiste
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Salva i file caricati
    obs_file = request.files['study_population']
    intv_file = request.files['study_population_with_intervention']
    targ_file = request.files['target_population']

    obs_path = os.path.join(app.config['UPLOAD_FOLDER'], 'study_population.csv')
    intv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'study_population_with_intervention.csv')
    targ_path = os.path.join(app.config['UPLOAD_FOLDER'], 'target_population.csv')

    obs_file.save(obs_path)
    intv_file.save(intv_path)
    targ_file.save(targ_path)

    # Parametri fissi: 1 colonna per X, Z e Y
    tolerance = float(request.form['tolerance'])
    results = run_transport_analysis(
        study_path=obs_path,
        study_intervention_path=intv_path,
        target_path=targ_path,
        num_x=1,
        num_z=1,
        num_y=1,
        tolerance=tolerance
    )

    # Passa anche le colonne Z per i plot
    return render_template('result.html', results=results, Z_cols=results['Z_cols'])

@app.route('/plot_z_distribution')
def plot_z_distribution():
    """Plot della probabilità marginale discreta di Z (P(Z))"""
    study_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'study_population.csv'))
    target_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'target_population.csv'))

    _, Z_cols, _ = extract_columns(study_df, num_x=1, num_z=1, num_y=1)
    z_col = request.args.get('z_col', Z_cols[0])

    # Calcola le distribuzioni marginali
    study_pz_df = compute_marginal_probability(study_df, z_col).rename(columns={'P': 'P_study'})
    target_pz_df = compute_marginal_probability(target_df, z_col).rename(columns={'P': 'P_target'})

    # Merge e allineamento
    merged = pd.merge(study_pz_df, target_pz_df, on=z_col, how='outer').fillna(0)
    merged = merged.sort_values(by=z_col)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(merged[z_col], merged['P_study'], label='Pr(Z) study population', marker='o', lw=2)
    ax.plot(merged[z_col], merged['P_target'], label='Pr*(Z) target population', marker='x', lw=2, linestyle='--')
    ax.set_xlabel(z_col)
    #ax.set_ylabel('P(Z)')
 
    ax.legend()
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/plot_y_distribution')
def plot_y_distribution():
    """Grafico generale della distribuzione marginale di Y: empirica vs stimata"""
    intv_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'study_population_with_intervention.csv'))
    transport_result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transport_result_X_0.csv')
    transport_df = pd.read_csv(transport_result_path)

    _, _, Y_cols = extract_columns(intv_df, num_x=1, num_z=1, num_y=1)
    y_col = Y_cols[0]

    true_py_df = compute_marginal_probability(intv_df, y_col).rename(columns={'P': 'P_true'})
    transport_df = transport_df.rename(columns={'P*(Y | do(X))': 'P_est'})

    merged = pd.merge(true_py_df, transport_df, on=y_col, how='outer').fillna(0)
    merged = merged.sort_values(by=y_col)

    fig, ax = plt.subplots()
    ax.plot(merged[y_col], merged['P_true'], label='Pr(Y|do(X)) study population', marker='o', lw=2)
    ax.plot(merged[y_col], merged['P_est'], label='Pr*(Y|do(X)) target population', marker='x', lw=2, linestyle='--')
    ax.set_xlabel(y_col)
    #ax.set_ylabel('P(Y)')
    
    ax.legend()
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Response(buf.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)

