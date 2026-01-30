import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import pytest


import pytest
import numpy as np
from mayini.clustering import KMeans


class TestKMeans:
    """Test cases for KMeans clustering"""
    
    def test_kmeans_init(self):
        """Test KMeans initialization"""
        kmeans = KMeans(n_clusters=3)
        assert kmeans.n_clusters == 3
        assert kmeans.max_iter == 100
    
    def test_kmeans_fit(self, blob_data):
        """Test KMeans fit method"""
        X, _ = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        assert hasattr(kmeans, 'cluster_centers_')
        assert hasattr(kmeans, 'labels_')
        assert kmeans.cluster_centers_.shape == (3, 2)
        assert len(kmeans.labels_) == len(X)
    
    def test_kmeans_fit_predict(self, blob_data):
        """Test KMeans fit_predict method"""
        X, _ = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) == 3
        assert all(label in [0, 1, 2] for label in labels)
    
    def test_kmeans_predict(self, blob_data):
        """Test KMeans predict method"""
        X, _ = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        # Test prediction on new data
        X_new = X[:10]
        predictions = kmeans.predict(X_new)
        
        assert len(predictions) == 10
        assert all(pred in [0, 1, 2] for pred in predictions)

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
