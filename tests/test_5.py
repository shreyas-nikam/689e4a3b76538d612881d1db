import pytest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from definition_415659c2b07d40e88b1e8e6960f74c1c import apply_kmeans_clustering


def test_apply_kmeans_clustering_valid_input():
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1, 2, 1, 2, 1]})
    n_clusters = 2
    x_col = 'x'
    y_col = 'y'
    result_df = apply_kmeans_clustering(df.copy(), n_clusters, x_col, y_col)
    assert 'cluster_id' in result_df.columns
    assert len(result_df['cluster_id'].unique()) == n_clusters


def test_apply_kmeans_clustering_empty_dataframe():
    df = pd.DataFrame({'x': [], 'y': []})
    n_clusters = 2
    x_col = 'x'
    y_col = 'y'
    result_df = apply_kmeans_clustering(df.copy(), n_clusters, x_col, y_col)
    assert 'cluster_id' in result_df.columns
    assert len(result_df['cluster_id']) == 0


def test_apply_kmeans_clustering_one_cluster():
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1, 2, 1, 2, 1]})
    n_clusters = 1
    x_col = 'x'
    y_col = 'y'
    result_df = apply_kmeans_clustering(df.copy(), n_clusters, x_col, y_col)
    assert 'cluster_id' in result_df.columns
    assert len(result_df['cluster_id'].unique()) == n_clusters

def test_apply_kmeans_clustering_non_numeric_data():
    df = pd.DataFrame({'x': ['a', 'b', 'c'], 'y': ['d', 'e', 'f']})
    n_clusters = 2
    x_col = 'x'
    y_col = 'y'
    with pytest.raises(TypeError):
        apply_kmeans_clustering(df.copy(), n_clusters, x_col, y_col)

def test_apply_kmeans_clustering_missing_columns():
    df = pd.DataFrame({'z': [1, 2, 3], 'w': [4, 5, 6]})
    n_clusters = 2
    x_col = 'x'
    y_col = 'y'
    with pytest.raises(KeyError):
        apply_kmeans_clustering(df.copy(), n_clusters, x_col, y_col)
