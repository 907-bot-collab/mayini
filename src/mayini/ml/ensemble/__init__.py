from .bagging import (
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor
)
from ..supervised.tree_models import (
    RandomForestClassifier,
    RandomForestRegressor,
)

# Boosting methods
from .boosting import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    XGBoostClassifier,
    XGBoostRegressor
)

# Voting methods
from .voting import (
    VotingClassifier,
    VotingRegressor,
    SoftVotingClassifier,
    HardVotingClassifier
)

# Stacking methods
from .stacking import (
    StackingClassifier,
    StackingRegressor,
    StackedEnsemble
)

# Utility functions
from .utils import (
    make_ensemble,
    compare_models,
    ensemble_weights_optimizer
)

# Define what gets imported with "from mayini.ml.ensemble import *"
__all__ = [
    # Bagging Methods
    'BaggingClassifier',
    'BaggingRegressor',
    'RandomForestClassifier',
    'RandomForestRegressor',
    'ExtraTreesClassifier',
    'ExtraTreesRegressor',
    
    # Boosting Methods
    'AdaBoostClassifier',
    'AdaBoostRegressor',
    'GradientBoostingClassifier',
    'GradientBoostingRegressor',
    'XGBoostClassifier',
    'XGBoostRegressor',
    
    # Voting Methods
    'VotingClassifier',
    'VotingRegressor',
    'SoftVotingClassifier',
    'HardVotingClassifier',
    
    # Stacking Methods
    'StackingClassifier',
    'StackingRegressor',
    'StackedEnsemble',
    
    # Utilities
    'make_ensemble',
    'compare_models',
    'ensemble_weights_optimizer',
]

DEFAULT_CONFIG = {
    'bagging': {
        'n_estimators': 10,
        'max_samples': 1.0,
        'max_features': 1.0,
        'bootstrap': True,
        'bootstrap_features': False,
        'random_state': None,
        'n_jobs': 1
    },
    'boosting': {
        'n_estimators': 50,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': None
    },
    'voting': {
        'voting': 'hard',  # 'hard' or 'soft'
        'weights': None,
        'n_jobs': 1
    },
    'stacking': {
        'cv': 5,
        'stack_method': 'auto',
        'n_jobs': 1,
        'passthrough': False
    }
}
