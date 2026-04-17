import os
import sys
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

os.chdir(BASE_DIR)            # models/ and data/ resolve from repo root
sys.path.insert(0, BASE_DIR)  # makes `src` importable as a package

from src.meta_feature_extractor import extract_meta_features
from src.dataset_analyzer       import analyze_dataset, compute_complexity_score
from src.predictor              import predict_algorithms
from src.performance_predictor  import predict_performance

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'automl-flask-secret')
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


def generate_heatmap(df, target_column):
    """Generate correlation heatmap, save to static/, return filename."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if len(numeric_cols) < 2:
        return None

    cols = numeric_cols[:10]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()

    fname = f'heatmap_{uuid.uuid4().hex[:8]}.png'
    fig.savefig(os.path.join(STATIC_DIR, fname), dpi=120, bbox_inches='tight')
    plt.close(fig)
    return fname


# ── routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. validate upload
    if 'file' not in request.files:
        flash('No file selected.')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Only CSV files are supported.')
        return redirect(url_for('index'))

    # 2. save & load
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception:
        flash('Could not read the CSV file. Please check the file format.')
        return redirect(url_for('index'))

    if df.empty:
        flash('The uploaded dataset is empty.')
        return redirect(url_for('index'))

    if len(df.columns) < 2:
        flash('Dataset must have at least 2 columns.')
        return redirect(url_for('index'))

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) == 0:
        flash('Dataset must contain at least one numeric feature column.')
        return redirect(url_for('index'))

    # 3. target column
    target_column = request.form.get('target_column', '').strip()
    if not target_column or target_column not in df.columns:
        flash('Invalid or missing target column.')
        return redirect(url_for('index'))

    # 4. backend calls
    try:
        meta_features   = extract_meta_features(df, target_column)
        analysis        = analyze_dataset(df, target_column)
        complexity      = compute_complexity_score(meta_features)
        recommendations = predict_algorithms(meta_features, top_k=3)

        for rec in recommendations:
            rec['expected_accuracy'] = round(
                float(predict_performance(meta_features, rec['algorithm'])) * 100, 2
            )
            rec['confidence_pct'] = round(float(rec['confidence']) * 100, 2)

    except FileNotFoundError as e:
        flash(f'Model not found: {e}. Please train the model first.')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Analysis error: {e}')
        return redirect(url_for('index'))

    # 5. heatmap
    try:
        heatmap_file = generate_heatmap(df, target_column)
    except Exception:
        heatmap_file = None

    # 6. prepare template data
    preview_html = df.head(8).to_html(classes='preview-table', border=0, index=False)

    meta_dict = {
        col: round(float(meta_features[col].values[0]), 4)
        for col in meta_features.columns
    }

    class_dist = analysis.get('class_distribution', {})

    return render_template(
        'result.html',
        filename        = filename,
        target_column   = target_column,
        shape           = analysis['shape'],
        total_missing   = analysis['total_missing'],
        n_classes       = analysis.get('n_classes', 'N/A'),
        complexity      = complexity,
        preview_html    = preview_html,
        meta_features   = meta_dict,
        recommendations = recommendations,
        class_dist      = class_dist,
        heatmap_file    = heatmap_file,
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
