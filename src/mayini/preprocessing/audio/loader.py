import numpy as np
import struct
import wave
from typing import Tuple, Optional, Union
from pathlib import Path


class AudioLoader:
    """
    Custom audio loader supporting WAV and basic audio formats
    No librosa dependency - pure implementation
    """
    
    def __init__(self):
        """Initialize audio loader"""
        self.supported_formats = ['wav', 'pcm']
        self.audio_data = None
        self.sample_rate = None
        self.n_channels = None
        self.n_samples = None
    
    def load_wav(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load WAV file
        
        Parameters:
        -----------
        filepath : str or Path
            Path to WAV file
        
        Returns:
        --------
        tuple : (audio_data, sample_rate)
            audio_data shape: (n_samples,) for mono or (n_samples, n_channels) for stereo
            sample_rate: integer
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        try:
            with wave.open(str(filepath), 'rb') as wav_file:
                # Get audio parameters
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Read audio data
                audio_bytes = wav_file.readframes(n_frames)
                
                # Convert bytes to numpy array
                audio_array = self._bytes_to_array(
                    audio_bytes,
                    n_channels,
                    sample_width,
                    n_frames
                )
                
                # Store metadata
                self.audio_data = audio_array
                self.sample_rate = sample_rate
                self.n_channels = n_channels
                self.n_samples = n_frames
                
                return audio_array, sample_rate
        
        except Exception as e:
            raise RuntimeError(f"Failed to load WAV file: {e}")
    
    def _bytes_to_array(self, audio_bytes: bytes, n_channels: int,
                       sample_width: int, n_frames: int) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        # Determine dtype based on sample width
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 3:
            # 24-bit audio
            return self._parse_24bit_audio(audio_bytes, n_channels, n_frames)
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert bytes to array
        audio_array = np.frombuffer(audio_bytes, dtype=dtype)
        
        # Reshape if stereo
        if n_channels > 1:
            audio_array = audio_array.reshape((n_frames, n_channels))
        
        # Normalize to [-1, 1] range
        max_val = np.iinfo(dtype).max
        audio_array = audio_array.astype(np.float32) / max_val
        
        return audio_array
    
    def _parse_24bit_audio(self, audio_bytes: bytes, n_channels: int,
                          n_frames: int) -> np.ndarray:
        """Parse 24-bit audio (special case)"""
        audio_array = np.zeros((n_frames * n_channels,), dtype=np.int32)
        
        for i in range(n_frames * n_channels):
            # Read 3 bytes
            byte_vals = audio_bytes[i*3:(i+1)*3]
            if len(byte_vals) == 3:
                # Convert 3 bytes to 32-bit int
                sample = int.from_bytes(byte_vals, byteorder='little', signed=True)
                audio_array[i] = sample
        
        # Reshape and normalize
        if n_channels > 1:
            audio_array = audio_array.reshape((n_frames, n_channels))
        
        audio_array = audio_array.astype(np.float32) / 2**23
        
        return audio_array
    
    def load_raw_pcm(self, filepath: Union[str, Path], sample_rate: int = 16000,
                    n_channels: int = 1, sample_width: int = 2) -> Tuple[np.ndarray, int]:
        """
        Load raw PCM audio file
        
        Parameters:
        -----------
        filepath : str or Path
            Path to PCM file
        sample_rate : int
            Sample rate in Hz
        n_channels : int
            Number of channels
        sample_width : int
            Sample width in bytes (1, 2, 3, or 4)
        
        Returns:
        --------
        tuple : (audio_data, sample_rate)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            audio_bytes = f.read()
        
        # Determine dtype
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert to array
        audio_array = np.frombuffer(audio_bytes, dtype=dtype)
        
        # Reshape
        if n_channels > 1:
            n_frames = len(audio_array) // n_channels
            audio_array = audio_array.reshape((n_frames, n_channels))
        
        # Normalize
        max_val = np.iinfo(dtype).max
        audio_array = audio_array.astype(np.float32) / max_val
        
        self.audio_data = audio_array
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.n_samples = len(audio_array) if n_channels == 1 else len(audio_array)
        
        return audio_array, sample_rate
    
    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo to mono"""
        if len(audio.shape) == 1:
            return audio
        else:
            return np.mean(audio, axis=1)
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to different sample rate
        
        Parameters:
        -----------
        audio : ndarray
            Audio data
        orig_sr : int
            Original sample rate
        target_sr : int
            Target sample rate
        
        Returns:
        --------
        ndarray : Resampled audio
        """
        if orig_sr == target_sr:
            return audio
        
        # Compute resampling ratio
        ratio = target_sr / orig_sr
        
        # Compute new length
        new_length = int(len(audio) * ratio)
        
        # Use simple linear interpolation
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        
        resampled = np.interp(new_indices, old_indices, audio)
        
        return resampled


class AudioWriter:
    """Write audio to WAV files"""
    
    @staticmethod
    def save_wav(audio: np.ndarray, sample_rate: int,
                filepath: Union[str, Path], bit_depth: int = 16):
        """
        Save audio as WAV file
        
        Parameters:
        -----------
        audio : ndarray
            Audio data (shape: (n_samples,) for mono)
        sample_rate : int
            Sample rate in Hz
        filepath : str or Path
            Output file path
        bit_depth : int
            Bit depth (8, 16, 24, or 32)
        """
        filepath = Path(filepath)
        
        # Normalize audio
        if audio.max() <= 1.0 and audio.min() >= -1.0:
            # Already normalized
            pass
        else:
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Convert to appropriate integer type
        if bit_depth == 8:
            audio_int = (audio * 127 + 128).astype(np.uint8)
            sample_width = 1
        elif bit_depth == 16:
            audio_int = (audio * 32767).astype(np.int16)
            sample_width = 2
        elif bit_depth == 32:
            audio_int = (audio * 2147483647).astype(np.int32)
            sample_width = 4
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        # Write WAV
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())


class AudioGenerator:
    """Generate synthetic audio"""
    
    @staticmethod
    def sine_wave(frequency: float, duration: float, sample_rate: int = 16000,
                 amplitude: float = 0.5) -> np.ndarray:
        """
        Generate sine wave
        
        Parameters:
        -----------
        frequency : float
            Frequency in Hz
        duration : float
            Duration in seconds
        sample_rate : int
            Sample rate in Hz
        amplitude : float
            Amplitude (0-1)
        
        Returns:
        --------
        ndarray : Audio data
        """
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        return audio.astype(np.float32)
    
    @staticmethod
    def silence(duration: float, sample_rate: int = 16000) -> np.ndarray:
        """Generate silence"""
        n_samples = int(duration * sample_rate)
        return np.zeros(n_samples, dtype=np.float32)
    
    @staticmethod
    def white_noise(duration: float, sample_rate: int = 16000,
                   amplitude: float = 0.1) -> np.ndarray:
        """Generate white noise"""
        n_samples = int(duration * sample_rate)
        return amplitude * np.random.randn(n_samples).astype(np.float32)
    
    @staticmethod
    def chirp(start_freq: float, end_freq: float, duration: float,
             sample_rate: int = 16000) -> np.ndarray:
        """
        Generate chirp (frequency sweep)
        
        Parameters:
        -----------
        start_freq : float
            Start frequency in Hz
        end_freq : float
            End frequency in Hz
        duration : float
            Duration in seconds
        sample_rate : int
            Sample rate in Hz
        
        Returns:
        --------
        ndarray : Audio data
        """
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        
        # Frequency as function of time
        freq = np.linspace(start_freq, end_freq, n_samples)
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        
        audio = np.sin(phase)
        return audio.astype(np.float32)
    
    @staticmethod
    def concatenate(*audios) -> np.ndarray:
        """Concatenate multiple audio arrays"""
        return np.concatenate(audios)
    
    @staticmethod
    def mix(*audios, weights: Optional[list] = None) -> np.ndarray:
        """
        Mix multiple audio arrays
        
        Parameters:
        -----------
        *audios : ndarray
            Audio arrays to mix
        weights : list, optional
            Mixing weights
        
        Returns:
        --------
        ndarray : Mixed audio
        """
        if weights is None:
            weights = [1.0 / len(audios)] * len(audios)
        
        # Ensure all same length
        max_len = max(len(a) for a in audios)
        
        mixed = np.zeros(max_len, dtype=np.float32)
        
        for audio, weight in zip(audios, weights):
            # Pad if necessary
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)))
            
            mixed += weight * audio[:max_len]
        
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val
        
        return mixed
