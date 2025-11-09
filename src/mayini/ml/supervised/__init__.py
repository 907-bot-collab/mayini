from .linear_models import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
    ElasticNet

)

from .naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB
)

from .tree_models import (
    DecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForest,
    RandomForestClassifier,
    RandomForestRegressor
)

from .knn import (
    KNN,
    KNNClassifier,
    KNNRegressor
)

from .svm import (
    SVC,
    SVR,
    SVM,
    LinearSVC,
    LinearSVR
)

# Define what gets imported with "from mayini.ml.supervised import *"
__all__ = [
    # Linear Models
    'LinearRegression',
    'Ridge',
    'Lasso',
    'LogisticRegression',

    
    # Naive Bayes
    'GaussianNB',
    'MultinomialNB',
    'BernoulliNB',
    
    # Tree Models
    'DecisionTree',
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    'RandomForest',
    'RandomForestClassifier',
    'RandomForestRegressor',
    
    # K-Nearest Neighbors
    'KNN',
    'KNNClassifier',
    'KNNRegressor',
    
    # Support Vector Machines
    'SVC',
    'SVR',
    'SVM',
    'LinearSVC',
    'LinearSVR',
]
