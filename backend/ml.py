"""
ml.py - Machine Learning Module for ProfitGuard AI
Implements from-scratch K-Means clustering using pure NumPy.
Features:
- Uniform random-point centroid initialization with multiple restarts
- Vectorized Euclidean distance assignment
- Centroid update step
- Convergence checking with tolerance
- Multiple random restarts (n_init) for global inertia minimization
- Sample-based Silhouette Score calculation
- RFM (Recency, Frequency, Monetary) feature extraction and scaling
"""

import numpy as np
import pandas as pd
from datetime import datetime


class StandardScalerFromScratch:
    """Standardizes features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero for constant columns
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KMeansFromScratch:
    """
    K-Means Clustering implemented from scratch using NumPy.
    """
    def __init__(self, n_clusters=4, max_iters=100, tol=1e-4, n_init=5, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _init_centroids(self, X, rng):
        """Forgy initialization: sample distinct observations uniformly."""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _fit_single(self, X, seed):
        rng = np.random.RandomState(seed)
        n_samples = X.shape[0]
        centroids = self._init_centroids(X, rng)

        for _ in range(self.max_iters):
            # Compute Euclidean distance to all centroids: shape (n_samples, n_clusters)
            # diff = X[:, None, :] - centroids[None, :, :]
            # dists = np.linalg.norm(diff, axis=2)
            dists = np.sqrt(np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    # Handle empty cluster by re-seeding with random sample
                    new_centroids[k] = X[rng.choice(n_samples)]

            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        # Compute inertia (within-cluster sum of squares)
        min_sq_dists = np.min(np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        inertia = np.sum(min_sq_dists)
        return centroids, labels, inertia

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        base_seed = self.random_state if self.random_state is not None else 42
        for i in range(self.n_init):
            seed = base_seed + i * 17
            centroids, labels, inertia = self._fit_single(X, seed)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        dists = np.sqrt(np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2))
        return np.argmin(dists, axis=1)


def compute_silhouette_score_sample(X, labels, sample_size=1000, random_state=42):
    """
    Computes approximate Silhouette Score using a random sample of the data.
    a(i): Mean distance to other points in the same cluster.
    b(i): Mean distance to points in the nearest neighboring cluster.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n_samples = len(X)
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    if n_samples > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, size=sample_size, replace=False)
        X_sub = X[idx]
        labels_sub = labels[idx]
    else:
        X_sub = X
        labels_sub = labels

    n_sub = len(X_sub)
    s_values = []

    for i in range(n_sub):
        same_mask = (labels_sub == labels_sub[i])
        other_clusters = [k for k in unique_labels if k != labels_sub[i]]

        # Intra-cluster distance a(i); explicitly remove this point from its
        # own cluster. A singleton's silhouette is defined as zero.
        intra_mask = same_mask.copy()
        intra_mask[i] = False
        if np.any(intra_mask):
            a_i = np.mean(np.linalg.norm(X_sub[intra_mask] - X_sub[i], axis=1))
        else:
            s_values.append(0.0)
            continue

        # Nearest cluster distance b(i)
        b_candidates = []
        for k in other_clusters:
            k_mask = (labels_sub == k)
            if np.sum(k_mask) > 0:
                b_k = np.mean(np.linalg.norm(X_sub[k_mask] - X_sub[i], axis=1))
                b_candidates.append(b_k)

        b_i = min(b_candidates) if b_candidates else 0.0

        denom = max(a_i, b_i)
        if denom > 0:
            s_values.append((b_i - a_i) / denom)
        else:
            s_values.append(0.0)

    return float(np.mean(s_values)) if s_values else 0.0


def extract_rfm_features(transactions_df, reference_date=None):
    """
    Extracts Recency, Frequency, Monetary (RFM) metrics per customer.
    Filters on status == 'Completed'.
    """
    completed = transactions_df[transactions_df['status'] == 'Completed'].copy()
    if 'timestamp' in completed.columns and not pd.api.types.is_datetime64_any_dtype(completed['timestamp']):
        completed['timestamp'] = pd.to_datetime(completed['timestamp'])

    if reference_date is None:
        reference_date = completed['timestamp'].max()

    # Calculate effective spend: quantity * unit_price * (1 - discount_pct)
    completed['effective_spend'] = (
        completed['quantity'] * completed['unit_price'] * (1.0 - completed['discount_pct'])
    )

    rfm = completed.groupby('customer_id').agg(
        recency=('timestamp', lambda dates: (reference_date - dates.max()).days),
        frequency=('transaction_id', 'count'),
        monetary=('effective_spend', 'sum')
    ).reset_index()

    return rfm
