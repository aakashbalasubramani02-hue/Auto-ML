import joblib
import pandas as pd
import numpy as np
import os

# Resolve models/ relative to this file so paths work from any cwd
_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SRC_DIR, '..', 'models')

def load_meta_model():
    """Load trained meta-learning model and label encoder."""

    model_path   = os.path.normpath(os.path.join(_MODELS_DIR, 'meta_model.pkl'))
    encoder_path = os.path.normpath(os.path.join(_MODELS_DIR, 'label_encoder.pkl'))

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Model files not found at {model_path}. Please train the model first."
        )

    model         = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    return model, label_encoder

def predict_algorithms(meta_features, top_k=3):
    """Predict top K recommended algorithms with probabilities."""
    
    model, label_encoder = load_meta_model()
    
    # Ensure meta_features has the correct columns
    required_features = ['n_samples', 'n_features', 'imbalance_ratio', 'avg_corr', 'class_sep']
    meta_features_filtered = meta_features[required_features]
    
    # Predict probabilities
    probabilities = model.predict_proba(meta_features_filtered)[0]
    
    # Get top K algorithms
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    recommendations = []
    for idx in top_indices:
        algo_name = label_encoder.classes_[idx]
        confidence = probabilities[idx]
        recommendations.append({
            'algorithm': algo_name,
            'confidence': confidence
        })
    
    return recommendations
