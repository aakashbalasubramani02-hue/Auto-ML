import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainingPipeline:
    """
    Comprehensive ML training pipeline with hyperparameter tuning and evaluation.
    """
    
    def __init__(self, data_path='data/processed/clean_meta_dataset.csv', results_dir='results'):
        self.data_path = data_path
        self.results_dir = results_dir
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.evaluation_results = {}
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_data(self):
        """Load the cleaned dataset."""
        print("Loading cleaned dataset...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Cleaned dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully: {self.df.shape}")
        print(f"Target distribution:\n{self.df['best_algo'].value_counts()}")
        
        return self.df
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for training with train/test split."""
        print("\nPreparing data for training...")
        
        # Remove ID column if exists
        feature_cols = [col for col in self.df.columns if col not in ['did', 'best_algo']]
        X = self.df[feature_cols]
        y = self.df['best_algo']
        
        print(f"Features: {list(X.columns)}")
        print(f"Feature matrix shape: {X.shape}")
        
        # Encode target labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Target classes: {self.label_encoder.classes_}")
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def define_models(self):
        """Define models with their parameter grids for hyperparameter tuning."""
        print("\nDefining models and parameter grids...")
        
        # Model definitions with pipelines (StandardScaler + Model)
        models_config = {
            'LogisticRegression': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
                ]),
                'param_grid': {
                    'classifier__C': [0.1, 1.0, 10.0, 100.0],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                }
            },
            
            'RandomForest': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(random_state=42))
                ]),
                'param_grid': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [5, 10, 15, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            
            'XGBoost': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', XGBClassifier(random_state=42, eval_metric='mlogloss'))
                ]),
                'param_grid': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__subsample': [0.8, 0.9, 1.0]
                }
            }
        }
        
        self.models = models_config
        print(f"Defined {len(self.models)} models for training")
        
        return self.models
    
    def train_and_tune_models(self, cv_folds=5, scoring='accuracy', n_jobs=-1):
        """Train models with hyperparameter tuning using GridSearchCV."""
        print(f"\nStarting model training with {cv_folds}-fold cross-validation...")
        print("="*60)
        
        trained_models = {}
        
        for model_name, model_config in self.models.items():
            print(f"\nTraining {model_name}...")
            print("-" * 40)
            
            # Setup GridSearchCV
            grid_search = GridSearchCV(
                estimator=model_config['pipeline'],
                param_grid=model_config['param_grid'],
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
            
            # Train model
            print(f"Performing grid search with {len(model_config['param_grid'])} parameter combinations...")
            grid_search.fit(self.X_train, self.y_train)
            
            # Store results
            trained_models[model_name] = {
                'model': grid_search,
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_
            }
            
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            print(f"Best Parameters: {grid_search.best_params_}")
        
        self.trained_models = trained_models
        print(f"\nCompleted training {len(trained_models)} models")
        
        return trained_models
    
    def evaluate_models(self):
        """Evaluate all trained models on the test set."""
        print("\nEvaluating models on test set...")
        print("="*50)
        
        evaluation_results = {}
        
        for model_name, model_info in self.trained_models.items():
            print(f"\nEvaluating {model_name}...")
            print("-" * 30)
            
            # Get best model
            best_model = model_info['best_estimator']
            
            # Make predictions
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Store results
            evaluation_results[model_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'best_cv_score': float(model_info['best_cv_score']),
                'best_params': model_info['best_params'],
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                'classification_report': classification_report(
                    self.y_test, y_pred, 
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
            }
            
            # Print results
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
        
        self.evaluation_results = evaluation_results
        print(f"\nCompleted evaluation of {len(evaluation_results)} models")
        
        return evaluation_results
    
    def select_best_model(self, metric='accuracy'):
        """Select the best model based on specified metric."""
        print(f"\nSelecting best model based on {metric}...")
        print("="*40)
        
        best_score = -1
        best_model_name = None
        
        # Compare models
        print("Model Performance Summary:")
        print("-" * 40)
        
        for model_name, results in self.evaluation_results.items():
            score = results[metric]
            print(f"{model_name:15}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        # Set best model
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]['best_estimator']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best {metric}: {best_score:.4f}")
        print(f"Best Parameters: {self.trained_models[best_model_name]['best_params']}")
        
        return self.best_model, best_model_name
    
    def save_results(self):
        """Save evaluation results and metrics to JSON file."""
        print(f"\nSaving results to {self.results_dir}/...")
        
        # Prepare results for saving
        results_to_save = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'data_path': self.data_path,
                'total_samples': len(self.df),
                'n_features': len([col for col in self.df.columns if col not in ['did', 'best_algo']]),
                'target_classes': self.label_encoder.classes_.tolist(),
                'train_size': len(self.X_train),
                'test_size': len(self.X_test)
            },
            'best_model': {
                'name': self.best_model_name,
                'accuracy': self.evaluation_results[self.best_model_name]['accuracy'],
                'parameters': self.trained_models[self.best_model_name]['best_params']
            },
            'model_results': self.evaluation_results
        }
        
        # Save to JSON
        results_file = os.path.join(self.results_dir, 'model_metrics.json')
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        # Save detailed report
        report_file = os.path.join(self.results_dir, 'training_report.txt')
        with open(report_file, 'w') as f:
            f.write("MODEL TRAINING PIPELINE REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Features: {len([col for col in self.df.columns if col not in ['did', 'best_algo']])}\n")
            f.write(f"Target classes: {', '.join(self.label_encoder.classes_)}\n\n")
            
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-" * 30 + "\n")
            for model_name, results in self.evaluation_results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
                f.write(f"  Precision: {results['precision']:.4f}\n")
                f.write(f"  Recall:    {results['recall']:.4f}\n")
                f.write(f"  F1-Score:  {results['f1_score']:.4f}\n")
                f.write(f"  CV Score:  {results['best_cv_score']:.4f}\n")
            
            f.write(f"\nBEST MODEL: {self.best_model_name}\n")
            f.write(f"Best Accuracy: {self.evaluation_results[self.best_model_name]['accuracy']:.4f}\n")
        
        print(f"Detailed report saved to: {report_file}")
        
        return results_file, report_file
    
    def save_best_model(self, model_path='models/best_model.pkl'):
        """Save the best model and label encoder."""
        print(f"\nSaving best model to {model_path}...")
        
        # Create models directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, model_path)
        
        # Save label encoder
        encoder_path = model_path.replace('best_model.pkl', 'best_model_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Best model saved: {model_path}")
        print(f"Label encoder saved: {encoder_path}")
        
        return model_path, encoder_path
    
    def run_pipeline(self):
        """Execute the complete training pipeline."""
        print("STARTING MODEL TRAINING PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Prepare data
            self.prepare_data()
            
            # Step 3: Define models
            self.define_models()
            
            # Step 4: Train and tune models
            self.train_and_tune_models()
            
            # Step 5: Evaluate models
            self.evaluate_models()
            
            # Step 6: Select best model
            self.select_best_model()
            
            # Step 7: Save results
            self.save_results()
            
            # Step 8: Save best model
            self.save_best_model()
            
            print("\n" + "="*60)
            print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return self.best_model, self.best_model_name, self.label_encoder
            
        except Exception as e:
            print(f"Error in training pipeline: {str(e)}")
            raise

def train_models():
    """Main function to run the training pipeline."""
    pipeline = ModelTrainingPipeline()
    best_model, best_model_name, label_encoder = pipeline.run_pipeline()
    
    return best_model, best_model_name, label_encoder

if __name__ == "__main__":
    train_models()