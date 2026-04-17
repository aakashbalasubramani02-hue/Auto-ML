import json
import os

METRICS_PATH = os.path.join('results', 'model_metrics.json')
OUTPUT_PATH  = os.path.join('results', 'model_comparison.txt')
METRICS      = ['accuracy', 'precision', 'recall', 'f1_score']


def load_metrics(path=METRICS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path) as f:
        return json.load(f)


def extract_table(data):
    """Return list of dicts with model name + four core metrics."""
    rows = []
    for model_name, values in data['model_results'].items():
        rows.append({
            'model':     model_name,
            'accuracy':  values['accuracy'],
            'precision': values['precision'],
            'recall':    values['recall'],
            'f1_score':  values['f1_score'],
            'cv_score':  values['best_cv_score'],
        })
    return rows


def find_best(rows, metric='accuracy'):
    return max(rows, key=lambda r: r[metric])


def format_table(rows):
    """Build a fixed-width text table."""
    col_model  = 22
    col_metric = 11

    header = (
        f"{'Model':<{col_model}}"
        f"{'Accuracy':>{col_metric}}"
        f"{'Precision':>{col_metric}}"
        f"{'Recall':>{col_metric}}"
        f"{'F1-Score':>{col_metric}}"
        f"{'CV Score':>{col_metric}}"
    )
    separator = '-' * len(header)

    lines = [header, separator]
    for r in rows:
        lines.append(
            f"{r['model']:<{col_model}}"
            f"{r['accuracy']:>{col_metric}.4f}"
            f"{r['precision']:>{col_metric}.4f}"
            f"{r['recall']:>{col_metric}.4f}"
            f"{r['f1_score']:>{col_metric}.4f}"
            f"{r['cv_score']:>{col_metric}.4f}"
        )
    lines.append(separator)
    return '\n'.join(lines)


def save_comparison(rows, best, output_path=OUTPUT_PATH):
    data   = load_metrics()
    info   = data['dataset_info']
    table  = format_table(rows)

    content = f"""
MODEL COMPARISON RESULTS
{'=' * 67}

Dataset Information
-------------------
  Total Samples : {info['total_samples']}
  Features      : {info['n_features']}
  Target Classes: {', '.join(info['target_classes'])}
  Train / Test  : {info['train_size']} / {info['test_size']}

{table}

Best Model  : {best['model']}
  Accuracy  : {best['accuracy']:.4f}
  Precision : {best['precision']:.4f}
  Recall    : {best['recall']:.4f}
  F1-Score  : {best['f1_score']:.4f}
  CV Score  : {best['cv_score']:.4f}
""".strip()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

    return content


def run_comparison():
    data = load_metrics()
    rows = extract_table(data)
    best = find_best(rows)
    content = save_comparison(rows, best)

    print(content)
    print(f"\nComparison saved to: {OUTPUT_PATH}")

    return rows, best


if __name__ == '__main__':
    run_comparison()
