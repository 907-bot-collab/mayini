"""Machine Learning module for mayini"""

# Supervised learning algorithms
from .supervised.knn import KNN, KNNClassifier, KNNRegressor
from .supervised.naive_bayes import NaiveBayes, MultinomialNB
from .supervised.svm import SVM, SVC, SVR
from .supervised.linear_models import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
    ElasticNet,
)
from .supervised.tree_models import (
    DecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

# Unsupervised learning algorithms
from .unsupervised.clustering import KMeans, DBSCAN, HierarchicalClustering
from .unsupervised.decomposition import PCA, LDA

# Ensemble methods
from .ensemble.bagging import BaggingClassifier, BaggingRegressor
from .ensemble.voting import VotingClassifier, VotingRegressor
from .ensemble.boosting import (
    #AdaBoost,
    #AdaBoostClassifier,
    GradientBoosting,
    RandomForest,
)

# Base classes
from .base import BaseClassifier, BaseRegressor
from .supervised.naive_bayes import GaussianNB

__all__ = [
    # Base classes
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    # Supervised - KNN
    "KNN",
    "KNNClassifier",
    "KNNRegressor",
    # Supervised - Naive Bayes
    "NaiveBayes",
    "GaussianNB",
    "MultinomialNB",
    # Supervised - SVM
    "SVM",
    "SVC",
    "SVR",
    # Supervised - Linear Models
    "LinearRegression",
    "Ridge",
    "Lasso",
    "LogisticRegression",
    "ElasticNet",
    # Supervised - Trees
    "DecisionTree",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    # Unsupervised - Clustering
    "KMeans",
    "DBSCAN",
    "HierarchicalClustering",
    # Unsupervised - Decomposition
    "PCA",
    "LDA",
    # Ensemble - Bagging
    "BaggingClassifier",
    "BaggingRegressor",
    # Ensemble - Voting
    "VotingClassifier",
    "VotingRegressor",
    # Ensemble - Boosting
    "AdaBoost",
    "AdaBoostClassifier",
    "GradientBoosting",
    "RandomForest",
]
