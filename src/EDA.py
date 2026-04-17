import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

class DataProcessor:
    """
    Comprehensive data processing pipeline for meta-learning dataset.
    Performs EDA, cleaning, and visualization.
    """
    
    def __init__(self, input_path='data/raw/meta_dataset.csv', output_path='data/processed/clean_meta_dataset.csv'):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """Load the raw dataset."""
        print("Loading dataset...")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Dataset not found at {self.input_path}")
        
        self.df = pd.read_csv(self.input_path)
        print(f"Dataset loaded successfully: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def data_quality_check(self):
        """Perform comprehensive data quality assessment."""
        print("\n" + "="*50)
        print("DATA QUALITY CHECK")
        print("="*50)
        
        # Basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\nData types:")
        print(self.df.dtypes)
        
        # Missing values analysis
        print("\nMissing values:")
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
        else:
            print("No missing values found!")
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        # Unique values in categorical columns
        print(f"\nUnique algorithms in 'best_algo': {self.df['best_algo'].nunique()}")
        print(f"Algorithm distribution:\n{self.df['best_algo'].value_counts()}")
        
        # Basic statistics
        print("\nBasic statistics for numeric columns:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe())
        
        return missing_df, duplicates
    
    def clean_data(self):
        """Clean the dataset based on quality assessment."""
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        self.cleaned_df = self.df.copy()
        
        # Remove duplicate rows
        initial_rows = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(self.cleaned_df)
        print(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values in numeric columns
        numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.cleaned_df[col].isnull().sum() > 0:
                median_val = self.cleaned_df[col].median()
                missing_count = self.cleaned_df[col].isnull().sum()
                self.cleaned_df.loc[:, col] = self.cleaned_df[col].fillna(median_val)
                print(f"Filled {missing_count} missing values in '{col}' with median: {median_val:.4f}")
        
        # Handle missing values in categorical columns
        if self.cleaned_df['best_algo'].isnull().sum() > 0:
            mode_val = self.cleaned_df['best_algo'].mode()[0]
            missing_count = self.cleaned_df['best_algo'].isnull().sum()
            self.cleaned_df.loc[:, 'best_algo'] = self.cleaned_df['best_algo'].fillna(mode_val)
            print(f"Filled {missing_count} missing values in 'best_algo' with mode: {mode_val}")
        
        # Standardize algorithm names
        algo_mapping = {
            'SVC': 'SVM',
            'DecisionTree': 'RandomForest',
            'DecisionTreeClassifier': 'RandomForest',
            'LogisticRegression': 'LogisticRegression',
            'RandomForest': 'RandomForest',
            'RandomForestClassifier': 'RandomForest',
            'XGBoost': 'XGBoost',
            'XGBClassifier': 'XGBoost',
            'KNN': 'KNN',
            'KNeighborsClassifier': 'KNN',
            'SVM': 'SVM'
        }
        
        self.cleaned_df['best_algo'] = self.cleaned_df['best_algo'].replace(algo_mapping)
        print(f"Standardized algorithm names")
        
        # Remove rows with invalid values
        initial_rows = len(self.cleaned_df)
        
        # Remove rows where numeric values are negative (if they shouldn't be)
        positive_cols = ['n_samples', 'n_features']
        for col in positive_cols:
            if col in self.cleaned_df.columns:
                self.cleaned_df = self.cleaned_df[self.cleaned_df[col] > 0]
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for col in numeric_cols:
            if col not in ['did']:  # Skip ID columns
                mean_val = self.cleaned_df[col].mean()
                std_val = self.cleaned_df[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                outliers_removed = len(self.cleaned_df) - len(self.cleaned_df[
                    (self.cleaned_df[col] >= lower_bound) & (self.cleaned_df[col] <= upper_bound)
                ])
                
                if outliers_removed > 0:
                    self.cleaned_df = self.cleaned_df[
                        (self.cleaned_df[col] >= lower_bound) & (self.cleaned_df[col] <= upper_bound)
                    ]
                    print(f"Removed {outliers_removed} extreme outliers from '{col}'")
        
        final_rows = len(self.cleaned_df)
        total_removed = initial_rows - final_rows
        print(f"Total rows removed during cleaning: {total_removed}")
        print(f"Final dataset shape: {self.cleaned_df.shape}")
        
        return self.cleaned_df
    
    def outlier_inspection(self):
        """Generate boxplots to inspect outliers in numeric columns."""
        print("\n" + "="*50)
        print("OUTLIER INSPECTION")
        print("="*50)
        
        numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'did']  # Exclude ID columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns found for outlier inspection")
            return
        
        # Create boxplots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(data=self.cleaned_df, y=col)
            plt.title(f'Boxplot of {col}')
            plt.ylabel(col)
        
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/outlier_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid display issues
        
        # Print outlier statistics
        for col in numeric_cols:
            Q1 = self.cleaned_df[col].quantile(0.25)
            Q3 = self.cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.cleaned_df[(self.cleaned_df[col] < lower_bound) | 
                                     (self.cleaned_df[col] > upper_bound)]
            
            print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(self.cleaned_df)*100:.2f}%)")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations for EDA."""
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Count plot of best_algo
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.cleaned_df, x='best_algo', order=self.cleaned_df['best_algo'].value_counts().index)
        plt.title('Distribution of Best Algorithms', fontsize=16, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        ax = plt.gca()
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('plots/algorithm_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap
        numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'did']
        
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.cleaned_df[numeric_cols].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Correlation Heatmap of Numeric Features', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Histograms of numeric features
        if len(numeric_cols) > 0:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 5 * n_rows))
            
            for i, col in enumerate(numeric_cols, 1):
                plt.subplot(n_rows, n_cols, i)
                sns.histplot(data=self.cleaned_df, x=col, kde=True, bins=30)
                plt.title(f'Distribution of {col}', fontweight='bold')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('plots/feature_histograms.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Boxplot of n_samples grouped by best_algo
        if 'n_samples' in self.cleaned_df.columns:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=self.cleaned_df, x='best_algo', y='n_samples')
            plt.title('Distribution of Sample Sizes by Algorithm', fontsize=16, fontweight='bold')
            plt.xlabel('Algorithm', fontsize=12)
            plt.ylabel('Number of Samples', fontsize=12)
            plt.xticks(rotation=45)
            plt.yscale('log')  # Use log scale for better visualization
            plt.tight_layout()
            plt.savefig('plots/samples_by_algorithm.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Pairplot of numeric features (sample if too many)
        plot_cols = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        if len(plot_cols) > 1:
            plt.figure(figsize=(12, 10))
            
            # Create a subset for pairplot if dataset is large
            sample_size = min(1000, len(self.cleaned_df))
            sample_df = self.cleaned_df.sample(n=sample_size, random_state=42)
            
            sns.pairplot(sample_df[plot_cols + ['best_algo']], hue='best_algo', diag_kind='hist')
            plt.suptitle('Pairplot of Numeric Features', y=1.02, fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('plots/feature_pairplot.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("All visualizations saved to 'plots/' directory")
    
    def save_cleaned_data(self):
        """Save the cleaned dataset."""
        print("\n" + "="*50)
        print("SAVING CLEANED DATASET")
        print("="*50)
        
        # Create output directory
        output_dir = os.path.dirname(self.output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cleaned dataset
        self.cleaned_df.to_csv(self.output_path, index=False)
        print(f"Cleaned dataset saved to: {self.output_path}")
        print(f"Final shape: {self.cleaned_df.shape}")
        
        # Save summary report
        report_path = os.path.join(output_dir, 'cleaning_report.txt')
        with open(report_path, 'w') as f:
            f.write("DATA CLEANING REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Original dataset shape: {self.df.shape}\n")
            f.write(f"Cleaned dataset shape: {self.cleaned_df.shape}\n")
            f.write(f"Rows removed: {self.df.shape[0] - self.cleaned_df.shape[0]}\n\n")
            
            f.write("Algorithm distribution after cleaning:\n")
            f.write(str(self.cleaned_df['best_algo'].value_counts()) + "\n\n")
            
            f.write("Missing values after cleaning:\n")
            missing_after = self.cleaned_df.isnull().sum()
            f.write(str(missing_after[missing_after > 0]) + "\n")
            if missing_after.sum() == 0:
                f.write("No missing values remaining!\n")
        
        print(f"Cleaning report saved to: {report_path}")
    
    def run_pipeline(self):
        """Execute the complete data processing pipeline."""
        print("STARTING DATA PROCESSING PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Data quality check
            self.data_quality_check()
            
            # Step 3: Clean data
            self.clean_data()
            
            # Step 4: Outlier inspection
            self.outlier_inspection()
            
            # Step 5: Generate visualizations
            self.generate_visualizations()
            
            # Step 6: Save cleaned data
            self.save_cleaned_data()
            
            print("\n" + "="*60)
            print("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return self.cleaned_df
            
        except Exception as e:
            print(f"Error in data processing pipeline: {str(e)}")
            raise

def main():
    """Main function to run the EDA pipeline."""
    processor = DataProcessor()
    cleaned_data = processor.run_pipeline()
    return cleaned_data

if __name__ == "__main__":
    main()