import pandas as pd
import numpy as np

def analyze_dataset(df, target_column):
    """Analyze dataset and return summary statistics."""
    
    analysis = {}
    
    # Dataset summary
    analysis['shape'] = df.shape
    analysis['columns'] = df.columns.tolist()
    analysis['dtypes'] = df.dtypes.to_dict()
    
    # Missing values
    missing = df.isnull().sum()
    analysis['missing_values'] = missing[missing > 0].to_dict()
    analysis['total_missing'] = df.isnull().sum().sum()
    
    # Class distribution
    if target_column in df.columns:
        class_dist = df[target_column].value_counts()
        analysis['class_distribution'] = class_dist.to_dict()
        analysis['n_classes'] = len(class_dist)
    
    # Feature correlations (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        analysis['correlation_matrix'] = corr_matrix
    else:
        analysis['correlation_matrix'] = None
    
    return analysis

def compute_complexity_score(meta_features):
    """Compute dataset complexity score: Easy, Medium, or Hard."""
    
    score = 0
    
    # High imbalance increases complexity
    if meta_features['imbalance_ratio'].values[0] > 5:
        score += 2
    elif meta_features['imbalance_ratio'].values[0] > 2:
        score += 1
    
    # High correlation reduces complexity
    if meta_features['avg_corr'].values[0] < 0.3:
        score += 2
    elif meta_features['avg_corr'].values[0] < 0.5:
        score += 1
    
    # Low class separability increases complexity
    if meta_features['class_sep'].values[0] < 0.3:
        score += 2
    elif meta_features['class_sep'].values[0] < 0.7:
        score += 1
    
    # Many features increases complexity
    if meta_features['n_features'].values[0] > 50:
        score += 1
    
    # Determine complexity level
    if score <= 2:
        return "Easy"
    elif score <= 4:
        return "Medium"
    else:
        return "Hard"
