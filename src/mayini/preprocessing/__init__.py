from .base import BaseTransformer

# Numerical
from .numerical.scalers import StandardScaler, MinMaxScaler, RobustScaler
from .numerical.imputers import SimpleImputer, KNNImputer
from .numerical.normalizers import Normalizer, PowerTransformer

# Categorical
from .categorical.encoders import LabelEncoder, OneHotEncoder, OrdinalEncoder
from .categorical.target_encoding import TargetEncoder, FrequencyEncoder

# Feature Engineering
from .feature_engineering.polynomial import PolynomialFeatures
from .feature_engineering.interactions import FeatureInteractions

# Text
from .text.vectorizers import CountVectorizer, TfidfVectorizer

# Selection
from .selection.variance import VarianceThreshold
from .selection.correlation import CorrelationThreshold

# Outlier & Type Conversion
from .outlier_detection import OutlierDetector
from .type_conversion import DataTypeConverter

# Pipeline
from .pipeline import Pipeline
from .autopreprocessor import AutoPreprocessor

__all__ = [
    # Base
    'BaseTransformer',
    
    # Numerical
    'StandardScaler',
    'MinMaxScaler',
    'RobustScaler',
    'SimpleImputer',
    'KNNImputer',
    'Normalizer',
    'PowerTransformer',
    
    # Categorical
    'LabelEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',
    'TargetEncoder',
    'FrequencyEncoder',
    
    # Feature Engineering
    'PolynomialFeatures',
    'FeatureInteractions',
    
    # Text
    'CountVectorizer',
    'TfidfVectorizer',
    
    # Selection
    'VarianceThreshold',
    'CorrelationThreshold',
    
    # Outlier & Type
    'OutlierDetector',
    'DataTypeConverter',
    
    # Pipeline
    'Pipeline',
    'AutoPreprocessor',
]


# ============================================================================
# FILE: src/mayini/preprocessing/numerical/__init__.py
# ============================================================================

"""Numerical preprocessing transformers"""

from .scalers import StandardScaler, MinMaxScaler, RobustScaler
from .imputers import SimpleImputer, KNNImputer
from .normalizers import Normalizer, PowerTransformer

__all__ = [
    'StandardScaler',
    'MinMaxScaler',
    'RobustScaler',
    'SimpleImputer',
    'KNNImputer',
    'Normalizer',
    'PowerTransformer',
]


# ============================================================================
# FILE: src/mayini/preprocessing/categorical/__init__.py
# ============================================================================

"""Categorical preprocessing transformers"""

from .encoders import LabelEncoder, OneHotEncoder, OrdinalEncoder
from .target_encoding import TargetEncoder, FrequencyEncoder

__all__ = [
    'LabelEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',
    'TargetEncoder',
    'FrequencyEncoder',
]


# ============================================================================
# FILE: src/mayini/preprocessing/feature_engineering/__init__.py
# ============================================================================

"""Feature engineering transformers"""

from .polynomial import PolynomialFeatures
from .interactions import FeatureInteractions

__all__ = [
    'PolynomialFeatures',
    'FeatureInteractions',
]


# ============================================================================
# FILE: src/mayini/preprocessing/text/__init__.py
# ============================================================================

"""Text preprocessing transformers"""

from .vectorizers import CountVectorizer, TfidfVectorizer

__all__ = [
    'CountVectorizer',
    'TfidfVectorizer',
]
