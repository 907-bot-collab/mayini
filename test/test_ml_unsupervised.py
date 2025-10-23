import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import pytest


def test_kmeans():
    """Test KMeans"""
    from mayini.ml.unsupervised.clustering import KMeans
    
    X = np.random.randn(100, 5)
    
    model = KMeans(n_clusters=3, max_iter=100)
    model.fit(X)
    labels = model.predict(X)
    
    assert labels.shape == (100,)
    assert len(np.unique(labels)) <= 3
    print("✅ KMeans passed")


def test_dbscan():
    """Test DBSCAN"""
    from mayini.ml.unsupervised.clustering import DBSCAN
    
    X = np.random.randn(100, 5)
    
    model = DBSCAN(eps=0.5, min_samples=5)
    model.fit(X)
    labels = model.labels_
    
    assert labels.shape == (100,)
    print("✅ DBSCAN passed")


def test_agglomerative():
    """Test AgglomerativeClustering"""
    from mayini.ml.unsupervised.clustering import AgglomerativeClustering
    
    X = np.random.randn(50, 5)
    
    model = AgglomerativeClustering(n_clusters=3, linkage='average')
    model.fit(X)
    labels = model.labels_
    
    assert labels.shape == (50,)
    assert len(np.unique(labels)) == 3
    print("✅ AgglomerativeClustering passed")


def test_pca():
    """Test PCA"""
    from mayini.ml.unsupervised.decomposition import PCA
    
    X = np.random.randn(100, 10)
    
    model = PCA(n_components=5)
    X_transformed = model.fit_transform(X)
    
    assert X_transformed.shape == (100, 5)
    print("✅ PCA passed")


def test_lda():
    """Test LDA"""
    from mayini.ml.unsupervised.decomposition import LDA
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    
    model = LDA(n_components=2)
    X_transformed = model.fit_transform(X, y)
    
    assert X_transformed.shape == (100, 2)
    print("✅ LDA passed")


if __name__ == '__main__':
    print("\\n" + "="*60)
    print("Testing Unsupervised Learning Algorithms")
    print("="*60 + "\\n")
    
    test_kmeans()
    test_dbscan()
    test_agglomerative()
    test_pca()
    test_lda()
    
    print("\\n" + "="*60)
    print("✅ All unsupervised learning tests passed!")
    print("="*60)
