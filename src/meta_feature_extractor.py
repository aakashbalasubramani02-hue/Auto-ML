import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def extract_meta_features(df, target_column):
    """Extract meta-features from a dataset."""
    
    features = {}
    
    # Basic dimensions
    features['n_samples'] = len(df)
    features['n_features'] = len(df.columns) - 1
    
    # Missing value ratio
    features['missing_value_ratio'] = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    
    # Numeric and categorical feature ratios
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    features['numeric_feature_ratio'] = len(numeric_cols) / features['n_features']
    features['categorical_feature_ratio'] = 1 - features['numeric_feature_ratio']
    
    # Target column analysis
    y = df[target_column]
    
    # Imbalance ratio
    if y.dtype == 'object' or len(y.unique()) < 20:
        class_counts = y.value_counts()
        features['imbalance_ratio'] = class_counts.max() / class_counts.min() if len(class_counts) > 1 else 1.0
    else:
        features['imbalance_ratio'] = 1.0
    
    # Average correlation (only numeric features)
    numeric_df = df[numeric_cols]
    if len(numeric_cols) > 1:
        corr_matrix = numeric_df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        features['avg_corr'] = upper_triangle.stack().mean()
    else:
        features['avg_corr'] = 0.0
    
    # Class separability (simplified)
    if len(numeric_cols) > 0 and (y.dtype == 'object' or len(y.unique()) < 20):
        try:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Calculate between-class variance vs within-class variance
            class_means = []
            for cls in np.unique(y_encoded):
                class_data = numeric_df[y_encoded == cls].mean()
                class_means.append(class_data)
            
            overall_mean = numeric_df.mean()
            between_var = np.var([cm.mean() for cm in class_means])
            within_var = numeric_df.var().mean()
            
            features['class_sep'] = between_var / (within_var + 1e-10)
        except:
            features['class_sep'] = 0.5
    else:
        features['class_sep'] = 0.5
    
    return pd.DataFrame([features])
