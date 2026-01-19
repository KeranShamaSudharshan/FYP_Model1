"""
Demonstration Script - Generates visualizations and example outputs
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.utils.data_preprocessing import DataPreprocessor
from src.models.clustering_model import PredictiveClusteringModel

# Set output directory
OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print(" "*20 + "GENERATING VISUALIZATIONS")
print("="*80)

# Load and preprocess data
print("\n1. Loading and preprocessing data...")
preprocessor = DataPreprocessor('Merge.csv')
preprocessor.load_data()
preprocessor.clean_data()
preprocessor.engineer_features()

# Prepare clustering data
print("\n2. Preparing clustering features...")
student_features = preprocessor.aggregate_student_features()
feature_cols = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 
               'Stability (%)', 'Is_Correct_Binary', 'Engagement_Encoded']
X = student_features[feature_cols]

# Train clustering model
print("\n3. Training clustering model...")
clustering_model = PredictiveClusteringModel(n_clusters=3)
labels = clustering_model.fit_predict(X)
mapped_labels = clustering_model.map_clusters_to_engagement(X.values, labels)

# Generate Elbow Curve
print("\n4. Generating elbow curve...")
inertias = []
K_range = range(2, 11)

for k in K_range:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X.values)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/elbow_curve.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/elbow_curve.png")
plt.close()

# Generate Cluster Visualization
print("\n5. Generating cluster visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.values)

plt.figure(figsize=(12, 8))
colors = ['#FF6B6B', '#FFD93D', '#6BCB77']  # Red, Yellow, Green
cluster_names = ['Passive', 'Moderate', 'Active']

for i in range(3):
    mask = mapped_labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors[i], label=cluster_names[i], 
               alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

# Plot centroids
centroids_pca = pca.transform(clustering_model.model.cluster_centers_)
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
plt.savefig(f'{OUTPUT_DIR}/cluster_visualization.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/cluster_visualization.png")
plt.close()

# Generate cluster distribution bar chart
print("\n6. Generating cluster distribution chart...")
cluster_counts = pd.Series(mapped_labels).map({0: 'Passive', 1: 'Moderate', 2: 'Active'}).value_counts()

plt.figure(figsize=(10, 6))
bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Engagement Cluster', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.title('Student Distribution Across Engagement Clusters', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(mapped_labels)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cluster_distribution.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/cluster_distribution.png")
plt.close()

# Generate engagement metrics comparison
print("\n7. Generating engagement metrics comparison...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Engagement Metrics by Cluster', fontsize=16, fontweight='bold')

metrics = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 
          'Stability (%)', 'Is_Correct_Binary', 'Engagement_Encoded']
metric_labels = ['Response Time', 'RTT', 'Jitter', 'Stability', 'Accuracy', 'Engagement']

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 3, idx % 3]
    
    data_by_cluster = []
    for cluster_id in range(3):
        mask = mapped_labels == cluster_id
        data_by_cluster.append(X[metric].values[mask])
    
    bp = ax.boxplot(data_by_cluster, labels=['Passive', 'Moderate', 'Active'],
                    patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel(label, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(label, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/engagement_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/engagement_metrics_comparison.png")
plt.close()

# Save model
print("\n8. Saving trained model...")
import pickle
with open(f'{OUTPUT_DIR}/clustering_model.pkl', 'wb') as f:
    pickle.dump(clustering_model.model, f)
print(f"   Saved: {OUTPUT_DIR}/clustering_model.pkl")

print("\n" + "="*80)
print(" "*20 + "VISUALIZATION GENERATION COMPLETE")
print("="*80)
print(f"\nAll files saved in: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  - elbow_curve.png")
print("  - cluster_visualization.png")
print("  - cluster_distribution.png")
print("  - engagement_metrics_comparison.png")
print("  - clustering_model.pkl")
print("  - student_clusters.csv")
print("  - student_feedback.csv")
