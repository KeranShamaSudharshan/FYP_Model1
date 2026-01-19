"""
Predictive Learner Clustering Model
K-Means clustering to group students based on engagement and performance
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


class PredictiveClusteringModel:
    """K-Means based student engagement clustering model"""
    
    def __init__(self, n_clusters=3, random_state=42):
        """
        Initialize the clustering model
        
        Args:
            n_clusters: Number of clusters (3 for Active, Moderate, Passive)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
        self.cluster_labels = {0: 'Passive', 1: 'Moderate', 2: 'Active'}
        self.is_fitted = False
        
    def fit(self, X):
        """
        Train the clustering model
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            
        Returns:
            self
        """
        print("\nTraining K-Means Clustering Model...")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Fit the model
        self.model.fit(X)
        self.is_fitted = True
        
        print("Model training completed.")
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def fit_predict(self, X):
        """
        Fit the model and predict cluster labels
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.predict(X)
    
    def map_clusters_to_engagement(self, X, labels):
        """
        Map cluster IDs to engagement levels based on cluster characteristics
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Mapped labels
        """
        # Calculate mean engagement metrics for each cluster
        cluster_means = {}
        for i in range(self.n_clusters):
            cluster_data = X[labels == i]
            # Higher engagement = lower response time + higher accuracy + better network
            engagement_score = cluster_data.mean(axis=0).mean()
            cluster_means[i] = engagement_score
        
        # Sort clusters by engagement score
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
        
        # Map: lowest score -> Passive (0), middle -> Moderate (1), highest -> Active (2)
        cluster_mapping = {
            sorted_clusters[0][0]: 0,  # Passive
            sorted_clusters[1][0]: 1,  # Moderate
            sorted_clusters[2][0]: 2   # Active
        }
        
        # Apply mapping
        mapped_labels = np.array([cluster_mapping[label] for label in labels])
        self.cluster_mapping = cluster_mapping
        
        return mapped_labels
    
    def evaluate(self, X, labels):
        """
        Evaluate clustering quality
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating Clustering Model...")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        # Inertia (within-cluster sum of squares)
        inertia = self.model.inertia_
        
        metrics = {
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': davies_bouldin,
            'Calinski-Harabasz Index': calinski_harabasz,
            'Inertia (WCSS)': inertia
        }
        
        print("\nClustering Quality Metrics:")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric:30s}: {value:.4f}")
        print("="*60)
        
        # Interpretation
        print("\nInterpretation:")
        print(f"- Silhouette Score ({silhouette:.4f}): ", end="")
        if silhouette > 0.5:
            print("Excellent clustering")
        elif silhouette > 0.3:
            print("Good clustering")
        elif silhouette > 0:
            print("Weak clustering structure")
        else:
            print("Poor clustering")
        
        print(f"- Davies-Bouldin Index ({davies_bouldin:.4f}): ", end="")
        if davies_bouldin < 1.0:
            print("Excellent separation")
        elif davies_bouldin < 2.0:
            print("Good separation")
        else:
            print("Poor separation")
        
        return metrics
    
    def elbow_method(self, X, max_clusters=10):
        """
        Perform elbow method to find optimal number of clusters
        
        Args:
            X: Feature matrix
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Dictionary with inertia values
        """
        print("\nPerforming Elbow Method Analysis...")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        inertias = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=self.random_state)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}")
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(K_range)
        plt.tight_layout()
        
        # Save plot
        output_dir = '../../data/processed'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/elbow_curve.png', dpi=300, bbox_inches='tight')
        print(f"\nElbow curve saved to {output_dir}/elbow_curve.png")
        plt.close()
        
        return dict(zip(K_range, inertias))
    
    def visualize_clusters(self, X, labels, feature_names=None):
        """
        Visualize clusters using the first two principal components
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Names of features
        """
        from sklearn.decomposition import PCA
        
        print("\nVisualizing clusters...")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_array)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        colors = ['#FF6B6B', '#FFD93D', '#6BCB77']  # Red, Yellow, Green
        cluster_names = ['Passive', 'Moderate', 'Active']
        
        for i in range(self.n_clusters):
            mask = labels == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[i], label=cluster_names[i], 
                       alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        # Plot centroids
        centroids_pca = pca.transform(self.model.cluster_centers_)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                   c='black', marker='X', s=300, 
                   edgecolors='white', linewidth=2, 
                   label='Centroids', zorder=10)
        
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.title('Student Engagement Clusters', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_dir = '../../data/processed'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/cluster_visualization.png', dpi=300, bbox_inches='tight')
        print(f"Cluster visualization saved to {output_dir}/cluster_visualization.png")
        plt.close()
    
    def get_cluster_statistics(self, X, labels, feature_names=None):
        """
        Get statistics for each cluster
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Names of features
            
        Returns:
            DataFrame with cluster statistics
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Create statistics DataFrame
        stats_list = []
        cluster_names = ['Passive', 'Moderate', 'Active']
        
        for i in range(self.n_clusters):
            cluster_data = X_array[labels == i]
            stats = {
                'Cluster': cluster_names[i],
                'Count': len(cluster_data),
                'Percentage': f"{len(cluster_data)/len(X_array)*100:.2f}%"
            }
            
            for j, feature in enumerate(feature_names):
                stats[f'{feature}_mean'] = cluster_data[:, j].mean()
                stats[f'{feature}_std'] = cluster_data[:, j].std()
            
            stats_list.append(stats)
        
        stats_df = pd.DataFrame(stats_list)
        
        print("\nCluster Statistics:")
        print("="*100)
        print(stats_df.to_string(index=False))
        print("="*100)
        
        return stats_df
    
    def save_model(self, filepath='../../data/processed/clustering_model.pkl'):
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='../../data/processed/clustering_model.pkl'):
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_fitted = True
        print(f"\nModel loaded from {filepath}")


if __name__ == "__main__":
    # Test the clustering model
    from utils.data_preprocessing import DataPreprocessor
    
    # Prepare data
    preprocessor = DataPreprocessor('../../Merge.csv')
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.engineer_features()
    
    # Get aggregated student features
    student_features = preprocessor.aggregate_student_features()
    feature_cols = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 
                   'Stability (%)', 'Is_Correct_Binary', 'Engagement_Encoded']
    X = student_features[feature_cols]
    
    # Train clustering model
    clustering_model = PredictiveClusteringModel(n_clusters=3)
    labels = clustering_model.fit_predict(X)
    
    # Map to engagement levels
    mapped_labels = clustering_model.map_clusters_to_engagement(X.values, labels)
    
    # Evaluate
    metrics = clustering_model.evaluate(X, mapped_labels)
    
    # Visualize
    clustering_model.visualize_clusters(X, mapped_labels, feature_cols)
    
    # Get statistics
    stats = clustering_model.get_cluster_statistics(X, mapped_labels, feature_cols)
    
    # Elbow method
    clustering_model.elbow_method(X, max_clusters=10)
    
    # Save model
    clustering_model.save_model()
