import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Resolve data/ and models/ relative to this file so paths work from any cwd
_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.normpath(os.path.join(_SRC_DIR, '..'))
_MODELS_DIR = os.path.join(_ROOT_DIR, 'models')
_DATA_DIR   = os.path.join(_ROOT_DIR, 'data')

def train_performance_models():
    """Train regression models to predict algorithm performance."""

    meta_data_path = os.path.join(_DATA_DIR, 'raw', 'meta_dataset.csv')
    
    if not os.path.exists(meta_data_path):
        print("Meta dataset not found. Cannot train performance models.")
        return
    
    df = pd.read_csv(meta_data_path)
    
    # Create synthetic performance data
    df['accuracy'] = 0.0
    
    for idx, row in df.iterrows():
        base_acc = 0.75
        
        # Adjust based on meta-features
        if row['class_sep'] > 1.0:
            base_acc += 0.1
        if row['imbalance_ratio'] > 5:
            base_acc -= 0.05
        if row['avg_corr'] > 0.6:
            base_acc += 0.05
        
        # Add noise
        base_acc += np.random.uniform(-0.05, 0.05)
        df.at[idx, 'accuracy'] = np.clip(base_acc, 0.5, 0.99)
    
    # Train a general performance predictor
    X = df[['n_samples', 'n_features', 'imbalance_ratio', 'avg_corr', 'class_sep']]
    y = df['accuracy']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    os.makedirs(_MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(_MODELS_DIR, 'performance_model.pkl'))
    print("Performance model trained and saved!")

def predict_performance(meta_features, algorithm_name):
    """Predict expected accuracy for a given algorithm."""

    model_path = os.path.normpath(os.path.join(_MODELS_DIR, 'performance_model.pkl'))
    
    if not os.path.exists(model_path):
        # Return heuristic-based estimate
        return estimate_performance_heuristic(meta_features, algorithm_name)
    
    model = joblib.load(model_path)
    
    required_features = ['n_samples', 'n_features', 'imbalance_ratio', 'avg_corr', 'class_sep']
    X = meta_features[required_features]
    
    base_accuracy = model.predict(X)[0]
    
    # Adjust based on algorithm
    adjustments = {
        'XGBoost': 0.03,
        'RandomForest': 0.01,
        'SVM': -0.02,
        'LogisticRegression': -0.03,
        'KNN': -0.04
    }
    
    adjusted_accuracy = base_accuracy + adjustments.get(algorithm_name, 0)
    return np.clip(adjusted_accuracy, 0.5, 0.99)

def estimate_performance_heuristic(meta_features, algorithm_name):
    """Heuristic-based performance estimation."""
    
    base_acc = 0.80
    
    # Adjust based on meta-features
    imbalance = meta_features['imbalance_ratio'].values[0]
    class_sep = meta_features['class_sep'].values[0]
    avg_corr = meta_features['avg_corr'].values[0]
    
    if class_sep > 1.0:
        base_acc += 0.08
    elif class_sep < 0.5:
        base_acc -= 0.08
    
    if imbalance > 5:
        base_acc -= 0.05
    
    if avg_corr > 0.7:
        base_acc += 0.03
    
    # Algorithm-specific adjustments
    algo_adjustments = {
        'XGBoost': 0.04,
        'RandomForest': 0.02,
        'SVM': -0.01,
        'LogisticRegression': -0.02,
        'KNN': -0.03
    }
    
    base_acc += algo_adjustments.get(algorithm_name, 0)
    
    return np.clip(base_acc, 0.55, 0.98)

if __name__ == "__main__":
    train_performance_models()
