"""
Data Preprocessing Module for Student Engagement Analytics
Prepares data from Merge.csv for clustering and feedback models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles data loading, cleaning, and feature engineering"""
    
    def __init__(self, data_path='Merge.csv'):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the Merge.csv dataset
        """
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.df = None
        
    def load_data(self):
        """Load the dataset from CSV file"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def clean_data(self):
        """Clean and handle missing values"""
        print("\nCleaning data...")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print(f"Missing values:\n{missing_values[missing_values > 0]}")
        
        # Fill missing numeric values with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        print("Data cleaning completed.")
        return self.df
    
    def engineer_features(self):
        """Create derived features for clustering"""
        print("\nEngineering features...")
        
        # Convert 'Is Correct' to binary
        self.df['Is_Correct_Binary'] = self.df['Is Correct'].apply(
            lambda x: 1 if str(x).lower() == 'yes' else 0
        )
        
        # Encode Engagement Level (Active=2, Moderate=1, Passive=0)
        engagement_mapping = {'Active': 2, 'Moderate': 1, 'Passive': 0}
        self.df['Engagement_Encoded'] = self.df['Engagement Level'].map(engagement_mapping)
        
        # Handle any unmapped values
        if self.df['Engagement_Encoded'].isnull().any():
            print("Warning: Some engagement levels not mapped. Filling with 1 (Moderate)")
            self.df['Engagement_Encoded'].fillna(1, inplace=True)
        
        # Encode Network Quality (Good=2, Fair=1, Poor=0)
        network_mapping = {'Good': 2, 'Fair': 1, 'Poor': 0}
        self.df['Network_Quality_Encoded'] = self.df['Network Quality'].map(network_mapping)
        
        # Handle any unmapped values
        if self.df['Network_Quality_Encoded'].isnull().any():
            print("Warning: Some network quality values not mapped. Filling with 1 (Fair)")
            self.df['Network_Quality_Encoded'].fillna(1, inplace=True)
        
        # Calculate accuracy per student
        student_accuracy = self.df.groupby('Student Name')['Is_Correct_Binary'].mean().reset_index()
        student_accuracy.columns = ['Student Name', 'Student_Accuracy']
        self.df = self.df.merge(student_accuracy, on='Student Name', how='left')
        
        # Calculate average response time per student
        student_response_time = self.df.groupby('Student Name')['Response Time (sec)'].mean().reset_index()
        student_response_time.columns = ['Student Name', 'Avg_Response_Time']
        self.df = self.df.merge(student_response_time, on='Student Name', how='left')
        
        print("Feature engineering completed.")
        return self.df
    
    def prepare_clustering_features(self):
        """
        Prepare features for the K-Means clustering model
        
        Returns:
            DataFrame with clustering features
        """
        print("\nPreparing features for clustering...")
        
        # Select features for clustering (as per SDS document)
        clustering_features = [
            'Response Time (sec)',     # Response time
            'RTT (ms)',                # Round Trip Time (network metric)
            'Jitter (ms)',             # Network jitter
            'Stability (%)',           # Network stability
            'Is_Correct_Binary',       # Answer correctness
            'Engagement_Encoded'       # Engagement level
        ]
        
        # Create feature matrix
        X = self.df[clustering_features].copy()
        
        # Handle any remaining NaN values
        X.fillna(X.median(), inplace=True)
        
        # Normalize features using MinMax scaling
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=clustering_features,
            index=X.index
        )
        
        print(f"Clustering features prepared. Shape: {X_scaled.shape}")
        return X_scaled
    
    def aggregate_student_features(self):
        """
        Aggregate features per student for clustering
        
        Returns:
            DataFrame with one row per student
        """
        print("\nAggregating features per student...")
        
        # Group by student and aggregate
        student_agg = self.df.groupby('Student Name').agg({
            'Response Time (sec)': 'mean',
            'RTT (ms)': 'mean',
            'Jitter (ms)': 'mean',
            'Stability (%)': 'mean',
            'Is_Correct_Binary': 'mean',
            'Engagement_Encoded': 'mean',
            'Admission No': 'first',
            'Email': 'first'
        }).reset_index()
        
        # Normalize features
        feature_cols = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 
                       'Stability (%)', 'Is_Correct_Binary', 'Engagement_Encoded']
        
        student_agg[feature_cols] = self.scaler.fit_transform(student_agg[feature_cols])
        
        print(f"Student aggregation completed. Shape: {student_agg.shape}")
        return student_agg
    
    def split_data(self, X, y=None, test_size=0.3, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test (if y provided) or X_train, X_test
        """
        if y is not None:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            return train_test_split(X, test_size=test_size, random_state=random_state)
    
    def get_data_summary(self):
        """Print summary statistics of the dataset"""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        print(f"\nEngagement Level Distribution:\n{self.df['Engagement Level'].value_counts()}")
        print(f"\nNetwork Quality Distribution:\n{self.df['Network Quality'].value_counts()}")
        print("="*60)


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor('../../Merge.csv')
    
    # Load and process data
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.engineer_features()
    
    # Get summary
    preprocessor.get_data_summary()
    
    # Prepare clustering features
    X_clustering = preprocessor.prepare_clustering_features()
    print(f"\nClustering features shape: {X_clustering.shape}")
    print(f"Sample features:\n{X_clustering.head()}")
    
    # Aggregate student features
    student_features = preprocessor.aggregate_student_features()
    print(f"\nStudent features shape: {student_features.shape}")
    print(f"Sample student features:\n{student_features.head()}")
