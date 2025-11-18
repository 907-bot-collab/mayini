from .categorical.encoders import LabelEncoder, OneHotEncoder, OrdinalEncoder
from .categorical.target_encoding import TargetEncoder, FrequencyEncoder
from .numerical.scalers import StandardScaler, MinMaxScaler, RobustScaler
from .numerical.imputers import SimpleImputer, KNNImputer
from .numerical.normalizers import Normalizer, PowerTransformer
from .feature_engineering.polynomial import PolynomialFeatures
from .feature_engineering.interactions import FeatureInteractions
from .text.vectorizers import TfidfVectorizer, CountVectorizer
from .outlier_detection import IsolationForest, LocalOutlierFactor
from .pipeline import Pipeline
from .selection.variance import VarianceThreshold
from .selection.correlation import CorrelationSelector
from .autopreprocessor import AutoPreprocessor
from .text.tokenizer import (
    Tokenizer,
    CharacterTokenizer,
    WordPieceTokenizer,
    NGramTokenizer
)

from .text.cleaner import (
    TextCleaner,
    TextNormalizer
)

from .text.stemmer import (
    PorterStemmer,
    SimpleStemmer,
    LancasterStemmer
)

from .text.vectorizer import (
    TFIDFVectorizer,
    CountVectorizer,
    BinaryVectorizer
)

from .text.embeddings import (
    Word2Vec,
    FastTextEmbeddings,
    GloVeEmbeddings
)


# ============================================================================
# IMAGE PREPROCESSING IMPORTS
# ============================================================================

from .image.loader import (
    ImageLoader,
    ImageWriter,
    ImageGenerator
)

from .image.transforms import (
    ImageTransforms,
    Normalize
)

from .image.augmentation import (
    ImageAugmentation
)

from .image.filters import (
    ConvolutionFilters,
    EdgeDetection,
    MorphologicalOperations,
    ColorConversion
)

from .image.features import (
    HistogramFeatures,
    TextureFeatures,
    HOGFeatures,
    ShapeFeatures
)


# ============================================================================
# AUDIO PREPROCESSING IMPORTS
# ============================================================================

from .audio.loader import (
    AudioLoader,
    AudioWriter,
    AudioGenerator
)

from .audio.features import (
    Spectrogram,
    MFCCExtractor,
    AudioFeatures,
    AudioStatistics
)

from .audio.effects import (
    AudioEffects,
    AudioAugmentation as AudioAug
)

from .audio.analysis import (
    AudioAnalysis,
    AudioQuality,
    FeatureComparison
)


# ============================================================================
# VIDEO PREPROCESSING IMPORTS
# ============================================================================

from .video.loader import (
    VideoLoader,
    VideoWriter,
    VideoProcessor
)

from .video.motion import (
    OpticalFlow,
    MotionDetection
)

from .video.detection import (
    SceneDetection,
    ShotBoundaryDetection
)

from .video.features import (
    TemporalFeatures,
    VideoStatistics,
    VideoAugmentation as VideoAug
)


# ============================================================================
# AUTOMATED PREPROCESSOR & WIDGET
# ============================================================================

from .preprocess import AutomatedPreprocessor

from .widget import PreprocessorWidget, launch_widget


# ============================================================================
# PUBLIC API - ALL EXPORTED CLASSES
# ============================================================================

__all__ = [
    # ===== TEXT PREPROCESSING =====
    'Tokenizer',
    'CharacterTokenizer',
    'WordPieceTokenizer',
    'NGramTokenizer',
    'TextCleaner',
    'TextNormalizer',
    'PorterStemmer',
    'SimpleStemmer',
    'LancasterStemmer',
    'TFIDFVectorizer',
    'CountVectorizer',
    'BinaryVectorizer',
    'Word2Vec',
    'FastTextEmbeddings',
    'GloVeEmbeddings',
    
    # ===== IMAGE PREPROCESSING =====
    'ImageLoader',
    'ImageWriter',
    'ImageGenerator',
    'ImageTransforms',
    'Normalize',
    'ImageAugmentation',
    'ConvolutionFilters',
    'EdgeDetection',
    'MorphologicalOperations',
    'ColorConversion',
    'HistogramFeatures',
    'TextureFeatures',
    'HOGFeatures',
    'ShapeFeatures',
    
    # ===== AUDIO PREPROCESSING =====
    'AudioLoader',
    'AudioWriter',
    'AudioGenerator',
    'Spectrogram',
    'MFCCExtractor',
    'AudioFeatures',
    'AudioStatistics',
    'AudioEffects',
    'AudioAug',
    'AudioAnalysis',
    'AudioQuality',
    'FeatureComparison',
    
    # ===== VIDEO PREPROCESSING =====
    'VideoLoader',
    'VideoWriter',
    'VideoProcessor',
    'OpticalFlow',
    'MotionDetection',
    'SceneDetection',
    'ShotBoundaryDetection',
    'TemporalFeatures',
    'VideoStatistics',
    'VideoAug',
    
    # ===== AUTOMATED PREPROCESSOR & WIDGET =====
    'AutomatedPreprocessor',
    'PreprocessorWidget',
    'launch_widget'
]

__all__ = [
    # Categorical encoding
    "LabelEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "FrequencyEncoder",
    # Numerical scaling
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    # Imputation
    "SimpleImputer",
    "KNNImputer",
    # Normalization
    "Normalizer",
    "PowerTransformer",
    # Feature engineering
    "PolynomialFeatures",
    "FeatureInteractions",
    # Text processing
    "TfidfVectorizer",
    "CountVectorizer",
    # Outlier detection
    "IsolationForest",
    "LocalOutlierFactor",
    # Pipeline
    "Pipeline",
    # Feature selection
    "VarianceThreshold",
    "CorrelationSelector",
    # Auto preprocessing
    "AutoPreprocessor",
]
