# Model 1: Clustering + Prediction - Full Implementation
# This script will be converted to Colab notebook

sections = {
    "setup": """
# Install and import required libraries
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.decomposition import PCA
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully")
""",
    
    "mount_drive": """
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted")
""",
    
    "load_data": """
# Load preprocessed data
DATA_PATH = '/content/drive/MyDrive/FYP_Data/'

print("Loading preprocessed data...")
X = pd.read_csv(DATA_PATH + 'Features_X.csv')
y = pd.read_csv(DATA_PATH + 'Target_y.csv').values.ravel()
student_data = pd.read_csv(DATA_PATH + 'Student_Aggregated_Data.csv')

print(f"✅ Data loaded successfully")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Student-level data shape: {student_data.shape}")
""",
    
    "clustering_features": """
# Select features for clustering
clustering_features = [
    'Response Time (sec)',
    'RTT (ms)',
    'Jitter (ms)',
    'Stability (%)',
    'Is_Correct_Binary',
    'Student_Avg_Accuracy'
]

X_clustering = X[clustering_features].copy()
print(f"Selected {len(clustering_features)} features for clustering")
""",
    
    "scale_clustering": """
# Scale features for clustering
scaler_clustering = StandardScaler()
X_clustering_scaled = scaler_clustering.fit_transform(X_clustering)
print("✅ Features scaled for clustering")
""",
    
    "elbow_method": """
# Elbow Method to find optimal K
print("Performing Elbow Method...")
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_clustering_scaled)
    inertias.append(kmeans.inertia_)
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal K')
plt.grid(True, alpha=0.3)
plt.show()
""",
    
    "train_kmeans": """
# Train K-Means with optimal K=3
print("Training K-Means Clustering Model...")
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42)
cluster_labels = kmeans.fit_predict(X_clustering_scaled)

print(f"✅ Clustering complete")
print(f"Cluster distribution: {np.bincount(cluster_labels)}")
"""
}

# Print the script structure
print("Model 1 Notebook Sections:")
for i, (key, value) in enumerate(sections.items(), 1):
    print(f"{i}. {key}")
    
print("\nTotal script length:", sum(len(v) for v in sections.values()), "characters")
