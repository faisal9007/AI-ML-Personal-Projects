"""
Clustering & Dimensionality Reduction - Iris Dataset
Sections: K-Means & Elbow Method | PCA | Hierarchical Clustering | DBSCAN

Run this in VS Code either as a plain script (python iris_clustering_assignment.py)
or cell-by-cell using the VS Code "Python Interactive" mode - each "# %%" marker
below is a Jupyter-style cell delimiter that VS Code recognizes automatically.
Plots are both shown (plt.show()) and saved as PNG files in the working directory.
"""

# %% [markdown]
# # Clustering & Dimensionality Reduction — Iris Dataset
# **Sections:** K-Means & Elbow Method · PCA · Hierarchical Clustering · DBSCAN
# 
# This notebook loads the Iris dataset and works through all four sections of the assignment, with code, plots, and interpretations after each result.

# %% [markdown]
# ## Setup — Imports & Data Loading

# %%
import os
import numpy as np
import pandas as pd
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (7, 5)

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)
df.head()


# %% [markdown]
# We standardize the features before clustering / PCA, since K-Means, hierarchical clustering, DBSCAN and PCA are all distance-based and sepal/petal measurements are on different scales (e.g. petal length ranges further than sepal width).

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pd.DataFrame(X_scaled, columns=feature_names).describe().round(2)


# %% [markdown]
# ---
# ## Section A: K-Means Clustering & Elbow Method
# 
# ### Task 1.1–1.2 — Load data & apply K-Means

# %%
kmeans_demo = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels_demo = kmeans_demo.fit_predict(X_scaled)
print("Cluster sizes:", np.bincount(kmeans_labels_demo))
print("Inertia (k=3):", round(kmeans_demo.inertia_, 2))


# %% [markdown]
# ### Task 1.3 — Elbow Method to choose optimal k

# %%
inertias = []
sil_scores = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    if k > 1:
        sil_scores.append(silhouette_score(X_scaled, km.labels_))
    else:
        sil_scores.append(np.nan)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(K_range), inertias, marker='o', color='#4C72B0')
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")
axes[0].axvline(3, color='red', linestyle='--', alpha=0.6, label='k=3')
axes[0].legend()

axes[1].plot(list(K_range)[1:], sil_scores[1:], marker='o', color='#55A868')
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette Score vs k")
axes[1].axvline(3, color='red', linestyle='--', alpha=0.6, label='k=3')
axes[1].legend()

plt.tight_layout()
plt.savefig('elbow_method.png', dpi=110)
plt.show()


# %% [markdown]
# **Interpretation:** The inertia (within-cluster sum of squares) drops sharply from k=1 to k=2 to k=3, then flattens out — the "elbow" sits at **k=3**. The silhouette score confirms this: it peaks (or is highest among small k) at k=3, matching the fact that Iris genuinely has 3 species. This is a good outcome since we know the ground truth has 3 classes, even though K-Means never saw the labels.

# %% [markdown]
# ### Task 1.4 — Final K-Means (k=3) and 2D / 3D cluster plots

# %%
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['kmeans_cluster'] = kmeans_labels

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 2D plot: first two original features
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans_labels, palette='Set1', ax=axes[0], s=60)
axes[0].set_xlabel(feature_names[0])
axes[0].set_ylabel(feature_names[1])
axes[0].set_title("K-Means Clusters (Sepal Length vs Sepal Width)")
axes[0].legend(title='Cluster')

# For comparison: colored by true species
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=[target_names[i] for i in y], palette='Set2', ax=axes[1], s=60)
axes[1].set_xlabel(feature_names[0])
axes[1].set_ylabel(feature_names[1])
axes[1].set_title("True Species (Sepal Length vs Sepal Width)")
axes[1].legend(title='Species')

plt.tight_layout()
plt.savefig('kmeans_2d.png', dpi=110)
plt.show()


# %%
from mpl_toolkits.mplot3d import Axes3D  # noqa

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans_labels, cmap='Set1', s=50)
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_zlabel(feature_names[2])
ax.set_title("K-Means Clusters in 3D (Sepal Length, Sepal Width, Petal Length)")
plt.savefig('kmeans_3d.png', dpi=110)
plt.show()

print("Agreement between K-Means clusters and true species (ARI):", round(adjusted_rand_score(y, kmeans_labels), 3))


# %% [markdown]
# **Interpretation:** In the sepal-only 2D view the *Setosa* cluster separates cleanly, but *Versicolor* and *Virginica* overlap somewhat on sepal measurements alone — sepal length/width alone don't fully separate them. The 3D plot (adding petal length) shows much cleaner separation between all three groups, since petal measurements are far more discriminative for Iris species than sepal measurements. The Adjusted Rand Index confirms K-Means recovers the true species structure quite well (typically ARI ≈ 0.62–0.73 depending on which features are used for the plot vs. all 4 features used for clustering).

# %% [markdown]
# ---
# ## Section B: Principal Component Analysis (PCA)
# 
# ### Task 2.1–2.2 — Apply PCA and plot the first two components

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=[target_names[i] for i in y], palette='Set2', s=70)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("PCA — First Two Principal Components (colored by true species)")
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('pca_2d.png', dpi=110)
plt.show()


# %% [markdown]
# ### Task 2.3 — Explained variance ratio

# %%
pca_full = PCA(n_components=4)
pca_full.fit(X_scaled)

evr = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr)

print("Explained variance ratio per component:", np.round(evr, 4))
print("Cumulative explained variance:", np.round(cum_evr, 4))

plt.figure(figsize=(7, 5))
plt.bar(range(1, 5), evr, alpha=0.7, label='Individual', color='#4C72B0')
plt.step(range(1, 5), cum_evr, where='mid', color='red', label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Component')
plt.xticks(range(1, 5))
plt.legend()
plt.tight_layout()
plt.savefig('pca_variance.png', dpi=110)
plt.show()


# %% [markdown]
# **Interpretation:** PC1 alone captures roughly 73% of the total variance and PC2 adds another ~23%, so the first two components together explain about **95–96%** of the variance in the 4-dimensional Iris data. This means the 2D PCA plot is a very faithful low-dimensional summary of the original 4 features — almost no information is lost by dropping to 2 dimensions. In the scatter plot, *Setosa* is clearly separated along PC1, while *Versicolor* and *Virginica* are closer together but still mostly distinguishable, consistent with petal features (which dominate PC1's loadings) being the most discriminative.

# %% [markdown]
# ### Task 2.4 — Compare clustering on PCA-reduced data vs. original (scaled) data

# %%
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_pca_labels = kmeans_pca.fit_predict(X_pca)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set1', ax=axes[0], s=60)
axes[0].set_title("K-Means (fit on original 4 features)\nplotted on PCA axes")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_pca_labels, palette='Set1', ax=axes[1], s=60)
axes[1].set_title("K-Means (fit directly on PCA-reduced data)")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

plt.tight_layout()
plt.savefig('pca_vs_original_clustering.png', dpi=110)
plt.show()

ari_original = adjusted_rand_score(y, kmeans_labels)
ari_pca = adjusted_rand_score(y, kmeans_pca_labels)
ari_between = adjusted_rand_score(kmeans_labels, kmeans_pca_labels)

sil_original = silhouette_score(X_scaled, kmeans_labels)
sil_pca = silhouette_score(X_pca, kmeans_pca_labels)

print(f"ARI vs true labels — original-feature clustering: {ari_original:.3f}")
print(f"ARI vs true labels — PCA-reduced clustering:      {ari_pca:.3f}")
print(f"ARI between the two clusterings:                   {ari_between:.3f}")
print(f"Silhouette — original features:                    {sil_original:.3f}")
print(f"Silhouette — PCA-reduced (2D):                      {sil_pca:.3f}")


# %% [markdown]
# **Interpretation:** The cluster assignments from K-Means on the full 4-feature (scaled) data and on the 2D PCA-reduced data are nearly identical (high ARI between the two, usually > 0.9) — because PC1+PC2 retain ~96% of the variance, almost no cluster-relevant information is lost. The silhouette score is typically slightly *higher* on the PCA-reduced data, since silhouette scores tend to look better in lower dimensions (less "curse of dimensionality" noise) even when the underlying grouping is the same. In short: PCA gives a much easier-to-visualize 2D view without materially changing which points get grouped together.

# %% [markdown]
# ---
# ## Section C: Hierarchical Clustering
# 
# ### Task 3.1 — Apply Agglomerative (Hierarchical) Clustering

# %%
hier = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_labels = hier.fit_predict(X_scaled)
df['hier_cluster'] = hier_labels
print("Cluster sizes:", np.bincount(hier_labels))


# %% [markdown]
# ### Task 3.2 — Dendrogram

# %%
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(11, 6))
dendrogram(Z, labels=y, leaf_rotation=90, leaf_font_size=6,
           color_threshold=7)
plt.title("Dendrogram — Ward Linkage (leaf labels = true species code 0/1/2)")
plt.xlabel("Sample index (true species code)")
plt.ylabel("Distance")
plt.axhline(y=7, color='red', linestyle='--', label='cut for 3 clusters')
plt.legend()
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=110)
plt.show()


# %% [markdown]
# **Interpretation:** Cutting the dendrogram at a height that yields 3 clusters shows one branch splitting off early and cleanly (corresponding to *Setosa*), while the other main branch splits later into two sub-branches that are more entangled (*Versicolor* and *Virginica*) — the same pattern K-Means found. This mirrors the biological fact that Setosa is the most morphologically distinct species, while Versicolor and Virginica are more similar to each other.

# %% [markdown]
# ### Task 3.3 — Compare hierarchical clusters with K-Means

# %%
ari_hier_vs_kmeans = adjusted_rand_score(kmeans_labels, hier_labels)
ari_hier_vs_true = adjusted_rand_score(y, hier_labels)

print(f"ARI — Hierarchical vs K-Means clusters: {ari_hier_vs_kmeans:.3f}")
print(f"ARI — Hierarchical vs true species:      {ari_hier_vs_true:.3f}")

ct = pd.crosstab(pd.Series(kmeans_labels, name='K-Means cluster'),
                  pd.Series(hier_labels, name='Hierarchical cluster'))
print("\nCross-tab: K-Means cluster vs Hierarchical cluster")
print(ct)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set1', ax=axes[0], s=60)
axes[0].set_title("K-Means clusters (PCA axes)")
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hier_labels, palette='Set1', ax=axes[1], s=60)
axes[1].set_title("Hierarchical clusters (PCA axes)")
plt.tight_layout()
plt.savefig('hier_vs_kmeans.png', dpi=110)
plt.show()


# %% [markdown]
# **Interpretation:** K-Means and Ward-linkage hierarchical clustering agree very closely (ARI typically ≈ 0.9+) — both isolate Setosa perfectly and only disagree on a handful of borderline points between Versicolor and Virginica. This is expected since Ward linkage, like K-Means, minimizes within-cluster variance, so the two algorithms tend to converge on similar partitions for well-separated, roughly spherical clusters like these.

# %% [markdown]
# ---
# ## Section D: DBSCAN
# 
# ### Task 4.1–4.2 — Apply DBSCAN and experiment with eps / min_samples

# %%
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 4])

plt.figure(figsize=(7, 5))
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel("5th nearest-neighbor distance")
plt.title("k-distance Graph (used to pick eps)")
plt.tight_layout()
plt.savefig('dbscan_kdistance.png', dpi=110)
plt.show()


# %%
results = []
eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
min_samples_values = [3, 5, 7]

for ms in min_samples_values:
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        if n_clusters > 1:
            mask = labels != -1
            sil = silhouette_score(X_scaled[mask], labels[mask]) if mask.sum() > n_clusters else np.nan
        else:
            sil = np.nan
        results.append({'eps': eps, 'min_samples': ms, 'n_clusters': n_clusters,
                         'n_noise': n_noise, 'silhouette': sil})

results_df = pd.DataFrame(results)
results_df


# %% [markdown]
# **Interpretation of the parameter sweep:** Small `eps` values (e.g. 0.3) produce many small clusters and lots of noise points, since few points are close enough to be considered neighbors. As `eps` increases (around 0.5–0.6 with `min_samples=5`), DBSCAN typically settles into 2 clusters (it tends to merge Versicolor and Virginica together, since they overlap in feature space, while still separating Setosa) with very few noise points. Very large `eps` collapses everything into a single cluster. Increasing `min_samples` makes the algorithm stricter about what counts as a "dense" region, which increases the noise count for the same `eps`.

# %% [markdown]
# ### Task 4.3 — Final DBSCAN model & cluster plot with noise highlighted

# %%
dbscan = DBSCAN(eps=0.6, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
df['dbscan_cluster'] = dbscan_labels

n_clusters_final = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_final = list(dbscan_labels).count(-1)
print(f"DBSCAN (eps=0.6, min_samples=5) -> clusters: {n_clusters_final}, noise points: {n_noise_final}")

plt.figure(figsize=(7, 6))
unique_labels = set(dbscan_labels)
palette = sns.color_palette('Set1', len(unique_labels))

for lbl, color in zip(unique_labels, palette):
    mask = dbscan_labels == lbl
    if lbl == -1:
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', marker='x', s=70, label='Noise')
    else:
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], s=60, label=f'Cluster {lbl}')

plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title(f"DBSCAN Clusters on PCA axes (eps=0.6, min_samples=5)\nNoise points marked with X")
plt.legend()
plt.tight_layout()
plt.savefig('dbscan_clusters.png', dpi=110)
plt.show()


# %% [markdown]
# ### Task 4.4 — Compare DBSCAN clusters with K-Means

# %%
ari_dbscan_vs_kmeans = adjusted_rand_score(kmeans_labels, dbscan_labels)
ari_dbscan_vs_true = adjusted_rand_score(y, dbscan_labels)

print(f"ARI — DBSCAN vs K-Means: {ari_dbscan_vs_kmeans:.3f}")
print(f"ARI — DBSCAN vs true species: {ari_dbscan_vs_true:.3f}")

ct2 = pd.crosstab(pd.Series(kmeans_labels, name='K-Means cluster'),
                   pd.Series(dbscan_labels, name='DBSCAN cluster (-1=noise)'))
print("\nCross-tab: K-Means cluster vs DBSCAN cluster")
print(ct2)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set1', ax=axes[0], s=60)
axes[0].set_title("K-Means (k=3)")

for lbl, color in zip(unique_labels, palette):
    mask = dbscan_labels == lbl
    if lbl == -1:
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', marker='x', s=70, label='Noise')
    else:
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], s=60, label=f'Cluster {lbl}')
axes[1].set_title("DBSCAN")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('dbscan_vs_kmeans.png', dpi=110)
plt.show()


# %% [markdown]
# **Interpretation:** Unlike K-Means (which is forced to produce exactly 3 clusters and always assigns every point to one), DBSCAN with these settings typically finds **2 density-based clusters** — it cleanly isolates Setosa as its own dense region, but merges Versicolor and Virginica into a single cluster because there's no clear density gap between them, plus it flags a handful of border/outlier points as **noise (-1)**. This is the key conceptual difference: K-Means partitions *all* points based on distance to centroids regardless of density, while DBSCAN only forms clusters where points are densely packed and is willing to call sparse/boundary points "noise" rather than force them into a cluster. As a result, DBSCAN's agreement with the true 3-species labels (ARI) is usually somewhat lower than K-Means' or hierarchical clustering's, since Iris's natural cluster shapes are closer to convex/spherical (which suits K-Means/Ward) rather than being separated by clear density gaps everywhere (which DBSCAN needs).

# %% [markdown]
# ---
# ## Overall Summary
# 
# | Method | # Clusters Found | Handles Noise? | Agreement with True Species (ARI) | Notes |
# |---|---|---|---|---|
# | K-Means (k=3, elbow-selected) | 3 (fixed) | No | High | Best match to ground truth; requires choosing k in advance |
# | Hierarchical (Ward, cut at 3) | 3 (fixed) | No | High, very close to K-Means | Dendrogram shows Setosa splits off first; no need to pre-specify k until cutting |
# | DBSCAN (eps=0.6, min_samples=5) | 2 (density-based) | Yes (flags noise) | Lower than K-Means/Hierarchical | Merges Versicolor & Virginica since they lack a density gap; useful for detecting outliers |
# | PCA (2 components) | — (dimensionality reduction, not clustering) | — | ~96% variance retained | Clustering on PCA-reduced data agrees almost perfectly with clustering on original scaled features |
# 
# **Key takeaways:**
# - The Iris dataset's 3 species are genuinely close to 3 convex clusters, which is why K-Means and hierarchical clustering (both variance-minimizing methods) recover the species structure well.
# - Setosa is consistently the easiest to separate across every method — it forms its own distinct cluster/branch/dense region everywhere.
# - Versicolor and Virginica overlap in feature space, so they are the main source of disagreement between methods and the main source of misclassification versus true labels.
# - PCA is an effective, near-lossless way to visualize this 4D dataset in 2D (95–96% variance retained), and clustering results are essentially unchanged whether performed on the original scaled features or the PCA-reduced data.
# - DBSCAN's density-based approach is fundamentally different from K-Means/hierarchical: it is better suited to finding arbitrarily-shaped clusters and identifying outliers, but here it under-splits the data relative to the true 3 classes because Versicolor/Virginica don't have a clean density gap between them.
#