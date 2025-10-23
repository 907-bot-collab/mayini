import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import pytest


def test_linear_regression():
    """Test LinearRegression"""
    from mayini.ml.supervised.linear_models import LinearRegression
    
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    score = model.score(X, y)
    
    assert predictions.shape == y.shape
    assert 0 <= score <= 1
    print("✅ LinearRegression passed")


def test_ridge_regression():
    """Test Ridge"""
    from mayini.ml.supervised.linear_models import Ridge
    
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ Ridge passed")


def test_lasso_regression():
    """Test Lasso"""
    from mayini.ml.supervised.linear_models import Lasso
    
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    model = Lasso(alpha=0.1, max_iter=500)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ Lasso passed")


def test_logistic_regression():
    """Test LogisticRegression"""
    from mayini.ml.supervised.linear_models import LogisticRegression
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression(learning_rate=0.1, n_iterations=100)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ LogisticRegression passed")


def test_decision_tree_classifier():
    """Test DecisionTreeClassifier"""
    from mayini.ml.supervised.tree_models import DecisionTreeClassifier
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ DecisionTreeClassifier passed")


def test_random_forest_classifier():
    """Test RandomForestClassifier"""
    from mayini.ml.supervised.tree_models import RandomForestClassifier
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    
    model = RandomForestClassifier(n_estimators=5, max_depth=5)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ RandomForestClassifier passed")


def test_knn_classifier():
    """Test KNeighborsClassifier"""
    from mayini.ml.supervised.knn import KNeighborsClassifier
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    
    model = KNeighborsClassifier(k=5)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ KNeighborsClassifier passed")


def test_gaussian_nb():
    """Test GaussianNB"""
    from mayini.ml.supervised.naive_bayes import GaussianNB
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    
    model = GaussianNB()
    model.fit(X, y)
    predictions = model.predict(X)
    proba = model.predict_proba(X)
    
    assert predictions.shape == y.shape
    assert proba.shape == (100, 3)
    print("✅ GaussianNB passed")


def test_multinomial_nb():
    """Test MultinomialNB"""
    from mayini.ml.supervised.naive_bayes import MultinomialNB
    
    X = np.random.randint(0, 10, (100, 5)).astype(float)
    y = np.random.randint(0, 2, 100)
    
    model = MultinomialNB(alpha=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ MultinomialNB passed")


def test_bernoulli_nb():
    """Test BernoulliNB"""
    from mayini.ml.supervised.naive_bayes import BernoulliNB
    
    X = np.random.randint(0, 2, (100, 5)).astype(float)
    y = np.random.randint(0, 2, 100)
    
    model = BernoulliNB(alpha=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ BernoulliNB passed")


def test_linear_svm():
    """Test LinearSVM"""
    from mayini.ml.supervised.svm import LinearSVM
    
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    model = LinearSVM(C=1.0, learning_rate=0.001, n_iterations=100)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ LinearSVM passed")


def test_svc():
    """Test SVC"""
    from mayini.ml.supervised.svm import SVC
    
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    model = SVC(kernel='rbf', C=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ SVC passed")


def test_svr():
    """Test SVR"""
    from mayini.ml.supervised.svm import SVR
    
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    
    model = SVR(kernel='rbf', C=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    print("✅ SVR passed")


if __name__ == '__main__':
    print("\\n" + "="*60)
    print("Testing Supervised Learning Algorithms")
    print("="*60 + "\\n")
    
    test_linear_regression()
    test_ridge_regression()
    test_lasso_regression()
    test_logistic_regression()
    test_decision_tree_classifier()
    test_random_forest_classifier()
    test_knn_classifier()
    test_gaussian_nb()
    test_multinomial_nb()
    test_bernoulli_nb()
    test_linear_svm()
    test_svc()
    test_svr()
    
    print("\\n" + "="*60)
    print("✅ All supervised learning tests passed!")
    print("="*60)
