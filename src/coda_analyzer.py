"""
Statistical analysis and clustering of sperm whale codas.

Finds structure in the four-feature space: clusters codas into types,
measures information capacity, and identifies social-unit signatures.
"""
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score


def build_feature_matrix(features_df, max_rhythm_len=9):
    rows = []
    for _, r in features_df.iterrows():
        rv = r["rhythm_vector"]
        padded = np.zeros(max_rhythm_len)
        padded[:min(len(rv), max_rhythm_len)] = rv[:max_rhythm_len]
        row = np.concatenate([
            [r["n_clicks"], r["tempo"], r["rubato"], r["ornamentation"], r["duration"]],
            padded,
        ])
        rows.append(row)
    return np.array(rows)


def cluster_codas(X, method="ward", n_clusters=None, distance_threshold=1.5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Z = linkage(X_scaled, method=method)
    if n_clusters:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    else:
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
    return labels, Z, scaler


def evaluate_clustering(X, labels, true_labels=None):
    X_scaled = StandardScaler().fit_transform(X)
    results = {
        "n_clusters": len(set(labels)),
        "silhouette": silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0,
    }
    if true_labels is not None:
        results["ari"] = adjusted_rand_score(true_labels, labels)
    return results


def reduce_dimensions(X, method="tsne", n_components=2, perplexity=30):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if method == "pca":
        reducer = PCA(n_components=n_components)
    else:
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return reducer.fit_transform(X_scaled)


def coda_type_statistics(features_df):
    type_stats = features_df.groupby("coda_type").agg(
        count=("tempo", "size"),
        mean_tempo=("tempo", "mean"),
        std_tempo=("tempo", "std"),
        mean_rubato=("rubato", "mean"),
        std_rubato=("rubato", "std"),
        mean_duration=("duration", "mean"),
        mean_clicks=("n_clicks", "mean"),
        mean_ornamentation=("ornamentation", "mean"),
        n_units=("unit", "nunique"),
        n_whales=("whale_id", "nunique"),
    ).reset_index()
    return type_stats.sort_values("count", ascending=False)


def unit_repertoire(features_df):
    repertoires = {}
    for unit, group in features_df.groupby("unit"):
        type_counts = Counter(group["coda_type"])
        total = sum(type_counts.values())
        repertoires[unit] = {
            "total_codas": total,
            "n_types": len(type_counts),
            "top_types": type_counts.most_common(10),
            "type_entropy": _entropy(list(type_counts.values())),
        }
    return repertoires


def _entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def information_capacity(features_df):
    type_counts = Counter(features_df["coda_type"])
    n_types = len(type_counts)
    entropy = _entropy(list(type_counts.values()))
    return {
        "n_distinguishable_types": n_types,
        "entropy_bits": entropy,
        "max_entropy_bits": np.log2(n_types) if n_types > 1 else 0,
        "efficiency": entropy / np.log2(n_types) if n_types > 1 else 0,
    }
