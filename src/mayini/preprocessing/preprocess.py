import numpy as np
from typing import Union, Optional, Dict, List, Any, Tuple
from pathlib import Path
import json


class AutomatedPreprocessor:
    """
    Unified preprocessing interface for all modalities
    Automatically detects data type and applies appropriate preprocessing
    
    Supports: Text, Image, Audio, Video
    """
    
    def __init__(self, modality: str = 'auto', **kwargs):
        """
        Initialize automated preprocessor
        
        Parameters:
        -----------
        modality : str
            Data modality: 'text', 'image', 'audio', 'video', or 'auto'
            If 'auto', modality will be detected from file extension
        **kwargs : dict
            Additional configuration options
        """
        self.modality = modality.lower() if modality else 'auto'
        self.config = kwargs
        self.data = None
        self.processed_data = None
        self.metadata = {}
        self.history = []
        self.current_shape = None
        
        # Initialize module references
        self._text_modules = {}
        self._image_modules = {}
        self._audio_modules = {}
        self._video_modules = {}
        
        # Lazy load modules
        self._modules_loaded = False
    
    def _load_modules(self):
        """Lazily load all preprocessing modules"""
        if self._modules_loaded:
            return
        
        try:
            # Import text modules
            from .text.cleaner import TextCleaner, TextNormalizer
            from .text.tokenizer import Tokenizer, CharacterTokenizer, WordPieceTokenizer
            from .text.stemmer import PorterStemmer, SimpleStemmer, LancasterStemmer
            from .text.vectorizer import TFIDFVectorizer, CountVectorizer, BinaryVectorizer
            from .text.embeddings import Word2Vec, FastTextEmbeddings, GloVeEmbeddings
            
            self._text_modules = {
                'cleaner': TextCleaner,
                'normalizer': TextNormalizer,
                'tokenizer': Tokenizer,
                'char_tokenizer': CharacterTokenizer,
                'wordpiece_tokenizer': WordPieceTokenizer,
                'stemmer': PorterStemmer,
                'simple_stemmer': SimpleStemmer,
                'lancaster_stemmer': LancasterStemmer,
                'tfidf': TFIDFVectorizer,
                'count_vectorizer': CountVectorizer,
                'binary_vectorizer': BinaryVectorizer,
                'word2vec': Word2Vec,
                'fasttext': FastTextEmbeddings,
                'glove': GloVeEmbeddings
            }
        except ImportError as e:
            raise ImportError(f"Failed to import text modules: {e}")
        
        try:
            # Import image modules
            from .image.loader import ImageLoader, ImageWriter, ImageGenerator
            from .image.transforms import ImageTransforms, Normalize
            from .image.augmentation import ImageAugmentation
            from .image.filters import ConvolutionFilters, EdgeDetection, MorphologicalOperations, ColorConversion
            from .image.features import HistogramFeatures, TextureFeatures, HOGFeatures, ShapeFeatures
            
            self._image_modules = {
                'loader': ImageLoader,
                'writer': ImageWriter,
                'generator': ImageGenerator,
                'transforms': ImageTransforms,
                'normalize': Normalize,
                'augmentation': ImageAugmentation,
                'filters': ConvolutionFilters,
                'edge_detection': EdgeDetection,
                'morphological': MorphologicalOperations,
                'color_conversion': ColorConversion,
                'histogram': HistogramFeatures,
                'texture': TextureFeatures,
                'hog': HOGFeatures,
                'shape': ShapeFeatures
            }
        except ImportError as e:
            raise ImportError(f"Failed to import image modules: {e}")
        
        try:
            # Import audio modules
            from .audio.loader import AudioLoader, AudioWriter, AudioGenerator
            from .audio.features import Spectrogram, MFCCExtractor, AudioFeatures, AudioStatistics
            from .audio.effects import AudioEffects, AudioAugmentation
            from .audio.analysis import AudioAnalysis, AudioQuality, FeatureComparison
            
            self._audio_modules = {
                'loader': AudioLoader,
                'writer': AudioWriter,
                'generator': AudioGenerator,
                'spectrogram': Spectrogram,
                'mfcc': MFCCExtractor,
                'features': AudioFeatures,
                'statistics': AudioStatistics,
                'effects': AudioEffects,
                'augmentation': AudioAugmentation,
                'analysis': AudioAnalysis,
                'quality': AudioQuality,
                'comparison': FeatureComparison
            }
        except ImportError as e:
            raise ImportError(f"Failed to import audio modules: {e}")
        
        try:
            # Import video modules
            from .video.loader import VideoLoader, VideoWriter, VideoProcessor
            from .video.motion import OpticalFlow, MotionDetection
            from .video.detection import SceneDetection, ShotBoundaryDetection
            from .video.features import TemporalFeatures, VideoStatistics, VideoAugmentation
            
            self._video_modules = {
                'loader': VideoLoader,
                'writer': VideoWriter,
                'processor': VideoProcessor,
                'optical_flow': OpticalFlow,
                'motion': MotionDetection,
                'scene_detection': SceneDetection,
                'shot_detection': ShotBoundaryDetection,
                'temporal': TemporalFeatures,
                'statistics': VideoStatistics,
                'augmentation': VideoAugmentation
            }
        except ImportError as e:
            raise ImportError(f"Failed to import video modules: {e}")
        
        self._modules_loaded = True
    
    def load_data(self, filepath: Union[str, Path], 
                  modality: Optional[str] = None) -> Any:
        """
        Load data from file with auto-detection
        
        Parameters:
        -----------
        filepath : str or Path
            Path to data file
        modality : str, optional
            Data modality ('text', 'image', 'audio', 'video')
            If None, auto-detected from file extension
        
        Returns:
        --------
        Any : Loaded data
        """
        self._load_modules()
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Auto-detect modality from extension
        if modality is None:
            extension = filepath.suffix.lower().strip('.')
            
            if extension in ['txt', 'csv', 'json']:
                modality = 'text'
            elif extension in ['jpg', 'jpeg', 'png', 'bmp', 'ppm', 'pgm']:
                modality = 'image'
            elif extension in ['wav', 'pcm', 'mp3']:
                modality = 'audio'
            elif extension in ['mp4', 'avi', 'mov', 'mkv']:
                modality = 'video'
            else:
                raise ValueError(f"Cannot auto-detect modality for extension: {extension}")
        
        self.modality = modality.lower()
        
        try:
            # Load based on modality
            if self.modality == 'text':
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.data = f.read()
                self.current_shape = len(self.data)
                self.history.append(f"Loaded text file: {filepath} ({len(self.data)} characters)")
            
            elif self.modality == 'image':
                loader = self._image_modules['loader']()
                self.data = loader.load(filepath)
                self.current_shape = self.data.shape
                self.history.append(f"Loaded image: {filepath} (shape: {self.data.shape})")
            
            elif self.modality == 'audio':
                loader = self._audio_modules['loader']()
                audio, sr = loader.load_wav(filepath)
                self.data = audio
                self.metadata['sample_rate'] = sr
                self.current_shape = self.data.shape
                self.history.append(f"Loaded audio: {filepath} ({len(self.data)} samples @ {sr}Hz)")
            
            elif self.modality == 'video':
                loader = self._video_modules['loader']()
                self.data = loader.load_video_frames(filepath)
                self.current_shape = self.data.shape
                self.history.append(f"Loaded video: {filepath} (shape: {self.data.shape})")
            
            else:
                raise ValueError(f"Unknown modality: {self.modality}")
            
            return self.data
        
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")
    
    def preprocess(self, pipeline: List[Dict[str, Any]]) -> Any:
        """
        Execute preprocessing pipeline
        
        Parameters:
        -----------
        pipeline : list of dict
            List of operations to execute
            Each dict should have 'operation' key and optional 'params' key
            
            Example:
            [
                {'operation': 'clean', 'params': {'remove_urls': True}},
                {'operation': 'tokenize', 'params': {'type': 'word'}},
                {'operation': 'stem'}
            ]
        
        Returns:
        --------
        Any : Processed data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self._load_modules()
        
        if not isinstance(pipeline, list):
            raise ValueError("Pipeline must be a list of operations")
        
        result = self.data
        
        for i, step in enumerate(pipeline):
            if not isinstance(step, dict):
                raise ValueError(f"Pipeline step {i} must be a dict")
            
            operation = step.get('operation')
            params = step.get('params', {})
            
            if not operation:
                raise ValueError(f"Pipeline step {i} missing 'operation' key")
            
            try:
                result = self._execute_operation(
                    operation, result, self.modality, params
                )
            except Exception as e:
                raise RuntimeError(f"Error in operation '{operation}': {str(e)}")
            
            self.history.append(f"Applied {operation} with params {params}")
        
        self.processed_data = result
        return result
    
    def _execute_operation(self, operation: str, data: Any, 
                          modality: str, params: Dict) -> Any:
        """Execute a single preprocessing operation"""
        
        if modality == 'text':
            return self._process_text(operation, data, params)
        elif modality == 'image':
            return self._process_image(operation, data, params)
        elif modality == 'audio':
            return self._process_audio(operation, data, params)
        elif modality == 'video':
            return self._process_video(operation, data, params)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def _process_text(self, operation: str, data: str, params: Dict) -> Any:
        """Process text data"""
        
        if operation == 'clean':
            cleaner = self._text_modules['cleaner']()
            return cleaner.clean(
                data,
                remove_urls=params.get('remove_urls', True),
                remove_emails=params.get('remove_emails', True),
                expand_contractions=params.get('expand_contractions', True),
                remove_stopwords=params.get('remove_stopwords', False)
            )
        
        elif operation == 'normalize':
            normalizer = self._text_modules['normalizer']()
            return normalizer.normalize(
                data,
                lowercase=params.get('lowercase', True),
                remove_accents=params.get('remove_accents', True)
            )
        
        elif operation == 'tokenize':
            token_type = params.get('type', 'word')
            if token_type == 'word':
                tokenizer = self._text_modules['tokenizer']()
            elif token_type == 'char':
                tokenizer = self._text_modules['char_tokenizer']()
            elif token_type == 'wordpiece':
                tokenizer = self._text_modules['wordpiece_tokenizer']()
            else:
                tokenizer = self._text_modules['tokenizer']()
            
            return tokenizer.tokenize(data)
        
        elif operation == 'stem':
            stemmer_type = params.get('stemmer', 'porter')
            if stemmer_type == 'simple':
                stemmer = self._text_modules['simple_stemmer']()
            elif stemmer_type == 'lancaster':
                stemmer = self._text_modules['lancaster_stemmer']()
            else:
                stemmer = self._text_modules['stemmer']()
            
            tokens = data if isinstance(data, list) else data.split()
            return [stemmer.stem(t) for t in tokens]
        
        elif operation == 'vectorize':
            vectorizer_type = params.get('type', 'tfidf')
            max_features = params.get('max_features', 5000)
            
            if vectorizer_type == 'count':
                vectorizer = self._text_modules['count_vectorizer'](max_features=max_features)
            elif vectorizer_type == 'binary':
                vectorizer = self._text_modules['binary_vectorizer'](max_features=max_features)
            else:
                vectorizer = self._text_modules['tfidf'](max_features=max_features)
            
            text = ' '.join(data) if isinstance(data, list) else data
            return vectorizer.fit_transform([text])
        
        else:
            raise ValueError(f"Unknown text operation: {operation}")
    
    def _process_image(self, operation: str, data: np.ndarray, params: Dict) -> Any:
        """Process image data"""
        
        if operation == 'resize':
            size = params.get('size', (224, 224))
            method = params.get('method', 'bilinear')
            transforms = self._image_modules['transforms']()
            return transforms.resize(data, size, method=method)
        
        elif operation == 'rotate':
            angle = params.get('angle', 0)
            transforms = self._image_modules['transforms']()
            return transforms.rotate(data, angle)
        
        elif operation == 'flip':
            direction = params.get('direction', 'horizontal')
            transforms = self._image_modules['transforms']()
            if direction == 'horizontal':
                return transforms.flip_horizontal(data)
            else:
                return transforms.flip_vertical(data)
        
        elif operation == 'crop':
            box = params.get('box', (0, 0, data.shape[1], data.shape[0]))
            transforms = self._image_modules['transforms']()
            return transforms.crop(data, box)
        
        elif operation == 'normalize':
            normalize = self._image_modules['normalize']()
            return normalize.normalize(data, method=params.get('method', 'standard'))
        
        elif operation == 'augment':
            augmentation = self._image_modules['augmentation']()
            result = data
            
            if params.get('noise', False):
                result = augmentation.add_gaussian_noise(result, std=params.get('noise_std', 25))
            
            if params.get('brightness', False):
                result = augmentation.adjust_brightness(result, factor=params.get('brightness_factor', 1.2))
            
            if params.get('flip', False):
                result = augmentation.random_flip(result)
            
            return result
        
        elif operation == 'edge_detection':
            method = params.get('method', 'sobel')
            edge_detection = self._image_modules['edge_detection']()
            
            # Convert to grayscale if needed
            if len(data.shape) == 3:
                data = np.mean(data, axis=2).astype(np.uint8)
            
            if method == 'sobel':
                return edge_detection.sobel(data)
            elif method == 'canny':
                return edge_detection.canny(data)
            elif method == 'laplacian':
                return edge_detection.laplacian(data)
            else:
                return edge_detection.sobel(data)
        
        elif operation == 'features':
            if len(data.shape) == 3:
                data = np.mean(data, axis=2).astype(np.uint8)
            
            feature_type = params.get('type', 'hog')
            if feature_type == 'hog':
                hog = self._image_modules['hog']()
                return hog.compute_hog(data)
            elif feature_type == 'histogram':
                hist = self._image_modules['histogram']()
                return hist.compute_histogram(data)
            else:
                hog = self._image_modules['hog']()
                return hog.compute_hog(data)
        
        else:
            raise ValueError(f"Unknown image operation: {operation}")
    
    def _process_audio(self, operation: str, data: np.ndarray, params: Dict) -> Any:
        """Process audio data"""
        
        sr = self.metadata.get('sample_rate', 16000)
        
        if operation == 'mfcc':
            n_mfcc = params.get('n_mfcc', 13)
            extractor = self._audio_modules['mfcc'](
                sample_rate=sr,
                n_mfcc=n_mfcc
            )
            return extractor.extract(data)
        
        elif operation == 'spectrogram':
            n_fft = params.get('n_fft', 2048)
            hop_length = params.get('hop_length', 512)
            spectrogram = self._audio_modules['spectrogram']()
            return spectrogram.spectrogram_db(data, sample_rate=sr, n_fft=n_fft, hop_length=hop_length)
        
        elif operation == 'pitch_shift':
            semitones = params.get('semitones', 2)
            effects = self._audio_modules['effects']()
            return effects.pitch_shift(data, sr, semitones)
        
        elif operation == 'time_stretch':
            rate = params.get('rate', 0.9)
            effects = self._audio_modules['effects']()
            return effects.time_stretch(data, rate)
        
        elif operation == 'effects':
            effect = params.get('effect', 'reverb')
            effects = self._audio_modules['effects']()
            
            if effect == 'reverb':
                return effects.add_reverb(data, sample_rate=sr)
            elif effect == 'echo':
                return effects.add_echo(data, sample_rate=sr)
            elif effect == 'chorus':
                return effects.add_chorus(data, sample_rate=sr)
            else:
                return effects.add_reverb(data, sample_rate=sr)
        
        elif operation == 'augment':
            augmentation = self._audio_modules['augmentation']()
            result = data
            
            if params.get('noise', False):
                result = augmentation.add_background_noise(result, noise_factor=params.get('noise_factor', 0.005))
            
            if params.get('gain', False):
                result = augmentation.random_gain(result, min_gain=params.get('min_gain', 0.8), max_gain=params.get('max_gain', 1.2))
            
            return result
        
        elif operation == 'analysis':
            analysis_type = params.get('type', 'tempo')
            analysis = self._audio_modules['analysis']()
            
            if analysis_type == 'tempo':
                return analysis.estimate_tempo(data, sr)
            elif analysis_type == 'pitch':
                return analysis.estimate_pitch(data, sr)
            else:
                return analysis.estimate_tempo(data, sr)
        
        else:
            raise ValueError(f"Unknown audio operation: {operation}")
    
    def _process_video(self, operation: str, data: np.ndarray, params: Dict) -> Any:
        """Process video data"""
        
        if operation == 'optical_flow':
            method = params.get('method', 'lucas_kanade')
            
            # Convert to grayscale
            gray_frames = np.mean(data[:2], axis=3).astype(np.uint8)
            
            flow = self._video_modules['optical_flow']()
            if method == 'horn_schunck':
                return flow.horn_schunck(gray_frames[0], gray_frames[1])
            else:
                return flow.lucas_kanade(gray_frames[0], gray_frames[1])
        
        elif operation == 'scene_detection':
            threshold = params.get('threshold', 0.5)
            method = params.get('method', 'histogram')
            detection = self._video_modules['scene_detection']()
            return detection.detect_scenes(data, threshold=threshold, method=method)
        
        elif operation == 'temporal_features':
            temporal = self._video_modules['temporal']()
            return temporal.motion_energy(data)
        
        elif operation == 'augment':
            aug_type = params.get('type', 'shift')
            augmentation = self._video_modules['augmentation']()
            
            if aug_type == 'shift':
                shift = params.get('shift', 1)
                return augmentation.temporal_shift(data, shift=shift)
            elif aug_type == 'crop':
                length = params.get('length', len(data) // 2)
                return augmentation.random_temporal_crop(data, length=length)
            else:
                return augmentation.temporal_shift(data)
        
        else:
            raise ValueError(f"Unknown video operation: {operation}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save processed data to file
        
        Parameters:
        -----------
        filepath : str or Path
            Output file path
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save. Call preprocess() first.")
        
        self._load_modules()
        filepath = Path(filepath)
        
        try:
            if self.modality == 'text':
                with open(filepath, 'w', encoding='utf-8') as f:
                    if isinstance(self.processed_data, list):
                        f.write(' '.join(self.processed_data))
                    else:
                        f.write(str(self.processed_data))
            
            elif self.modality == 'image':
                writer = self._image_modules['writer']()
                writer.save_ppm(self.processed_data, filepath, binary=True)
            
            elif self.modality == 'audio':
                sr = self.metadata.get('sample_rate', 16000)
                writer = self._audio_modules['writer']()
                writer.save_wav(self.processed_data, sr, filepath, bit_depth=16)
            
            elif self.modality == 'video':
                fps = self.metadata.get('fps', 30)
                writer = self._video_modules['writer']()
                writer.write_video(self.processed_data, filepath, fps=fps)
            
            self.history.append(f"Saved to {filepath}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to save data: {str(e)}")
    
    def get_history(self) -> List[str]:
        """
        Get preprocessing history
        
        Returns:
        --------
        list : List of operations performed
        """
        return self.history.copy()
    
    def get_metadata(self) -> Dict:
        """
        Get metadata about processed data
        
        Returns:
        --------
        dict : Metadata dictionary
        """
        return {
            'modality': self.modality,
            'current_shape': str(self.current_shape),
            'metadata': self.metadata,
            'history_length': len(self.history)
        }
    
    def summary(self) -> str:
        """
        Get comprehensive preprocessing summary
        
        Returns:
        --------
        str : Summary text
        """
        summary_text = f"""
╔════════════════════════════════════════════════════════════════╗
║           Automated Preprocessor Summary                       ║
╚════════════════════════════════════════════════════════════════╝

📊 MODALITY: {self.modality.upper()}
📏 DATA SHAPE: {self.current_shape}

📋 PROCESSING HISTORY ({len(self.history)} steps):
"""
        for i, step in enumerate(self.history, 1):
            summary_text += f"  {i}. {step}\n"
        
        summary_text += f"\n📌 METADATA:\n"
        for key, value in self.metadata.items():
            summary_text += f"  {key}: {value}\n"
        
        summary_text += "\n" + "="*60 + "\n"
        
        return summary_text
    
    def reset(self) -> None:
        """Reset preprocessor state"""
        self.data = None
        self.processed_data = None
        self.metadata = {}
        self.history = []
        self.current_shape = None
        self.modality = 'auto'
