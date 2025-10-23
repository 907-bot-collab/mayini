import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import pandas as pd
import pytest


def test_standard_scaler():
    """Test StandardScaler"""
    from mayini.preprocessing.numerical.scalers import StandardScaler
    
    X = np.random.randn(100, 5)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == X.shape
    assert np.abs(np.mean(X_scaled, axis=0)).max() < 0.1
    print("✅ StandardScaler passed")


def test_minmax_scaler():
    """Test MinMaxScaler"""
    from mayini.preprocessing.numerical.scalers import MinMaxScaler
    
    X = np.random.randn(100, 5)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == X.shape
    assert np.min(X_scaled) >= 0
    assert np.max(X_scaled) <= 1
    print("✅ MinMaxScaler passed")


def test_simple_imputer():
    """Test SimpleImputer"""
    from mayini.preprocessing.numerical.imputers import SimpleImputer
    
    X = np.random.randn(100, 5)
    X[0, 0] = np.nan
    X[1, 1] = np.nan
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()
    print("✅ SimpleImputer passed")


def test_knn_imputer():
    """Test KNNImputer"""
    from mayini.preprocessing.numerical.imputers import KNNImputer
    
    X = np.random.randn(50, 5)
    X[0, 0] = np.nan
    X[1, 1] = np.nan
    
    imputer = KNNImputer(n_neighbors=3)
    X_imputed = imputer.fit_transform(X)
    
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()
    print("✅ KNNImputer passed")


def test_normalizer():
    """Test Normalizer"""
    from mayini.preprocessing.numerical.normalizers import Normalizer
    
    X = np.random.randn(100, 5)
    
    normalizer = Normalizer(norm='l2')
    X_normalized = normalizer.fit_transform(X)
    
    assert X_normalized.shape == X.shape
    print("✅ Normalizer passed")


def test_label_encoder():
    """Test LabelEncoder"""
    from mayini.preprocessing.categorical.encoders import LabelEncoder
    
    X = np.array(['cat', 'dog', 'cat', 'bird']).reshape(-1, 1)
    
    encoder = LabelEncoder()
    X_encoded = encoder.fit_transform(X)
    
    assert X_encoded.shape == X.shape
    print("✅ LabelEncoder passed")


def test_onehot_encoder():
    """Test OneHotEncoder"""
    from mayini.preprocessing.categorical.encoders import OneHotEncoder
    
    X = np.array(['cat', 'dog', 'cat']).reshape(-1, 1)
    
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    
    assert X_encoded.shape[0] == 3
    print("✅ OneHotEncoder passed")


def test_target_encoder():
    """Test TargetEncoder"""
    from mayini.preprocessing.categorical.target_encoding import TargetEncoder
    
    X = np.array(['A', 'B', 'A', 'C', 'B']).reshape(-1, 1)
    y = np.array([1, 0, 1, 1, 0])
    
    encoder = TargetEncoder(smoothing=1.0)
    X_encoded = encoder.fit_transform(X, y)
    
    assert X_encoded.shape == X.shape
    print("✅ TargetEncoder passed")


def test_polynomial_features():
    """Test PolynomialFeatures"""
    from mayini.preprocessing.feature_engineering.polynomial import PolynomialFeatures
    
    X = np.random.randn(50, 3)
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    assert X_poly.shape[0] == 50
    assert X_poly.shape[1] > 3
    print("✅ PolynomialFeatures passed")


def test_feature_interactions():
    """Test FeatureInteractions"""
    from mayini.preprocessing.feature_engineering.interactions import FeatureInteractions
    
    X = np.random.randn(50, 3)
    
    interactions = FeatureInteractions(interaction_type='multiply')
    X_interact = interactions.fit_transform(X)
    
    assert X_interact.shape[0] == 50
    assert X_interact.shape[1] > 3
    print("✅ FeatureInteractions passed")


def test_count_vectorizer():
    """Test CountVectorizer"""
    from mayini.preprocessing.text.vectorizers import CountVectorizer
    
    docs = ["hello world", "hello python", "python world"]
    
    vectorizer = CountVectorizer(max_features=10)
    X = vectorizer.fit_transform(docs)
    
    assert X.shape[0] == 3
    assert X.shape[1] <= 10
    print("✅ CountVectorizer passed")


def test_tfidf_vectorizer():
    """Test TfidfVectorizer"""
    from mayini.preprocessing.text.vectorizers import TfidfVectorizer
    
    docs = ["hello world", "hello python", "python programming"]
    
    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform(docs)
    
    assert X.shape[0] == 3
    assert X.shape[1] <= 10
    print("✅ TfidfVectorizer passed")


def test_variance_threshold():
    """Test VarianceThreshold"""
    from mayini.preprocessing.selection.variance import VarianceThreshold
    
    X = np.random.randn(100, 5)
    X[:, 0] = 0  # Zero variance column
    
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    
    assert X_selected.shape[0] == 100
    assert X_selected.shape[1] < 5
    print("✅ VarianceThreshold passed")


def test_correlation_threshold():
    """Test CorrelationThreshold"""
    from mayini.preprocessing.selection.correlation import CorrelationThreshold
    
    X = np.random.randn(100, 5)
    X[:, 1] = X[:, 0] + np.random.randn(100) * 0.01  # Highly correlated
    
    selector = CorrelationThreshold(threshold=0.95)
    X_selected = selector.fit_transform(X)
    
    assert X_selected.shape[0] == 100
    print("✅ CorrelationThreshold passed")


def test_outlier_detector():
    """Test OutlierDetector"""
    from mayini.preprocessing.outlier_detection import OutlierDetector
    
    X = np.random.randn(100, 5)
    X[0, 0] = 100  # Outlier
    
    detector = OutlierDetector(method='iqr', action='cap')
    X_cleaned = detector.fit_transform(X)
    
    assert X_cleaned.shape[0] <= 100
    print("✅ OutlierDetector passed")


def test_pipeline():
    """Test Pipeline"""
    from mayini.preprocessing.pipeline import Pipeline
    from mayini.preprocessing.numerical.imputers import SimpleImputer
    from mayini.preprocessing.numerical.scalers import StandardScaler
    
    X = np.random.randn(100, 5)
    X[0, 0] = np.nan
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])
    
    X_processed = pipeline.fit_transform(X)
    
    assert X_processed.shape == X.shape
    assert not np.isnan(X_processed).any()
    print("✅ Pipeline passed")


if __name__ == '__main__':
    print("\\n" + "="*60)
    print("Testing Preprocessing Module")
    print("="*60 + "\\n")
    
    test_standard_scaler()
    test_minmax_scaler()
    test_simple_imputer()
    test_knn_imputer()
    test_normalizer()
    test_label_encoder()
    test_onehot_encoder()
    test_target_encoder()
    test_polynomial_features()
    test_feature_interactions()
    test_count_vectorizer()
    test_tfidf_vectorizer()
    test_variance_threshold()
    test_correlation_threshold()
    test_outlier_detector()
    test_pipeline()
    
    print("\\n" + "="*60)
    print("✅ All preprocessing tests passed!")
    print("="*60)
