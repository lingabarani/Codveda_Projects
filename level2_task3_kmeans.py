"""
Level 2 - Task 3: Clustering Analysis (K-Means)
Dataset: Iris
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/uploads/1__iris.csv')
df['species'] = df['species'].str.strip().str.lower().str.capitalize()
numeric = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

print("=" * 60)
print("LEVEL 2 – TASK 3: CLUSTERING ANALYSIS (K-MEANS)")
print("Dataset: Iris")
print("=" * 60)

X = df[numeric].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Elbow Method ──────────────────────────────────────────────────────────────
inertias = []
sil_scores = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

print("\n▶ Elbow Method Results:")
for k, inertia, sil in zip(K_range, inertias, sil_scores):
    print(f"   k={k}  Inertia={inertia:.2f}  Silhouette={sil:.4f}")

# Optimal k = 3 (known from domain knowledge & silhouette)
optimal_k = 3
km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X_scaled)
print(f"\n▶ Optimal k = {optimal_k}")
print(f"▶ Cluster sizes:\n{df['Cluster'].value_counts().sort_index()}")

# ── PCA for 2D visualisation ──────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]
print(f"\n▶ PCA Explained Variance: {pca.explained_variance_ratio_.round(3)}")
print(f"   Total: {pca.explained_variance_ratio_.sum():.3f}")

# ── Figure 1: Elbow + Silhouette ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
ax.axvline(optimal_k, color='red', linestyle='--', linewidth=1.5, label=f'Optimal k={optimal_k}')
ax.set_xlabel('Number of Clusters (k)', fontsize=11)
ax.set_ylabel('Inertia (WCSS)', fontsize=11)
ax.set_title('Elbow Method', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(list(K_range), sil_scores, 'gs-', linewidth=2, markersize=8)
ax.axvline(optimal_k, color='red', linestyle='--', linewidth=1.5, label=f'Optimal k={optimal_k}')
ax.set_xlabel('Number of Clusters (k)', fontsize=11)
ax.set_ylabel('Silhouette Score', fontsize=11)
ax.set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

plt.suptitle("Level 2 – Task 3: K-Means Cluster Optimisation", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L2T3_elbow.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Cluster visualisation (PCA 2D) ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cluster_colors = ['#4C72B0', '#DD8452', '#55A868']
species_colors  = ['#E377C2', '#8C564B', '#BCBD22']

ax = axes[0]
for c_id, color in enumerate(cluster_colors):
    mask = df['Cluster'] == c_id
    ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'], c=color,
               label=f'Cluster {c_id}', s=70, edgecolors='white', alpha=0.85)
# Plot centroids in PCA space
centroids_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='*', s=300,
           c='red', zorder=5, label='Centroids', edgecolors='black')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
ax.set_title(f'K-Means Clusters (k={optimal_k})\nPCA Projection', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for sp, color in zip(df['species'].unique(), species_colors):
    mask = df['species'] == sp
    ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'], c=color,
               label=sp, s=70, edgecolors='white', alpha=0.85)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
ax.set_title('Actual Species Labels\n(Ground Truth)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle("Level 2 – Task 3: K-Means vs Ground Truth (Iris)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/L2T3_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ K-Means clustering complete – 2 plots saved")
