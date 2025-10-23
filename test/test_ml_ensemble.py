import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import pytest


def test_bagging_classifier():
    """Test BaggingClassifier"""
    from mayini.ml.ensemble.bagging import BaggingClassifier
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = BaggingClassifier(n_estimators=5, max_samples=0.8)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ BaggingClassifier passed")


def test_bagging_regressor():
    """Test BaggingRegressor"""
    from mayini.ml.ensemble.bagging import BaggingRegressor
    
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    model = BaggingRegressor(n_estimators=5, max_samples=0.8)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ BaggingRegressor passed")


def test_adaboost():
    """Test AdaBoostClassifier"""
    from mayini.ml.ensemble.boosting import AdaBoostClassifier
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = AdaBoostClassifier(n_estimators=10, learning_rate=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ AdaBoostClassifier passed")


def test_gradient_boosting_classifier():
    """Test GradientBoostingClassifier"""
    from mayini.ml.ensemble.boosting import GradientBoostingClassifier
    
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ GradientBoostingClassifier passed")


def test_gradient_boosting_regressor():
    """Test GradientBoostingRegressor"""
    from mayini.ml.ensemble.boosting import GradientBoostingRegressor
    
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    
    model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ GradientBoostingRegressor passed")


def test_voting_classifier():
    """Test VotingClassifier"""
    from mayini.ml.ensemble.voting import VotingClassifier
    from mayini.ml.supervised.linear_models import LogisticRegression
    from mayini.ml.supervised.tree_models import DecisionTreeClassifier
    from mayini.ml.supervised.knn import KNeighborsClassifier
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    estimators = [
        ('lr', LogisticRegression(n_iterations=50)),
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('knn', KNeighborsClassifier(k=3))
    ]
    
    model = VotingClassifier(estimators=estimators, voting='hard')
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ VotingClassifier passed")


def test_voting_regressor():
    """Test VotingRegressor"""
    from mayini.ml.ensemble.voting import VotingRegressor
    from mayini.ml.supervised.linear_models import LinearRegression, Ridge
    
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    estimators = [
        ('lr', LinearRegression()),
        ('ridge', Ridge(alpha=1.0))
    ]
    
    model = VotingRegressor(estimators=estimators)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ VotingRegressor passed")


if __name__ == '__main__':
    print("\\n" + "="*60)
    print("Testing Ensemble Learning Methods")
    print("="*60 + "\\n")
    
    test_bagging_classifier()
    test_bagging_regressor()
    test_adaboost()
    test_gradient_boosting_classifier()
    test_gradient_boosting_regressor()
    test_voting_classifier()
    test_voting_regressor()
    
    print("\\n" + "="*60)
    print("✅ All ensemble learning tests passed!")
    print("="*60)
