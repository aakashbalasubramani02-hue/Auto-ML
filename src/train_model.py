import joblib
import os
from model_pipeline import ModelTrainingPipeline

def train_meta_model():
    """Train meta-learning model using the comprehensive training pipeline."""
    
    print("Starting Meta-Learning Model Training")
    print("="*50)
    
    # Check if cleaned dataset exists
    cleaned_data_path = os.path.join('data', 'processed', 'clean_meta_dataset.csv')
    
    if not os.path.exists(cleaned_data_path):
        print("Error: clean_meta_dataset.csv not found!")
        print("Please run EDA.py first to generate the cleaned dataset:")
        print("python src/EDA.py")
        return None, None
    
    try:
        # Initialize and run the training pipeline
        pipeline = ModelTrainingPipeline(data_path=cleaned_data_path)
        best_model, best_model_name, label_encoder = pipeline.run_pipeline()
        
        print(f"\nTraining completed successfully!")
        print(f"Best model: {best_model_name}")
        print(f"Model saved to: models/best_model.pkl")
        print(f"Label encoder saved to: models/best_model_encoder.pkl")
        
        # Also save in the original format for compatibility
        print("\nSaving models in original format for compatibility...")
        joblib.dump(best_model, 'models/meta_model.pkl')
        joblib.dump(label_encoder, 'models/label_encoder.pkl')
        print("Compatibility models saved!")
        
        return best_model, label_encoder
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None

if __name__ == "__main__":
    train_meta_model()
