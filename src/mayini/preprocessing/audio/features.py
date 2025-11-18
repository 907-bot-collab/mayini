import numpy as np
from scipy import signal, fftpack
from typing import Tuple, Optional
import math


class Spectrogram:
    """Spectrogram computation"""
    
    @staticmethod
    def stft(audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
            window: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        n_fft : int
            FFT size
        hop_length : int
            Hop length between frames
        window : str
            Window type ('hann', 'hamming', 'blackman')
        
        Returns:
        --------
        tuple : (stft_matrix, frequencies, times)
            stft_matrix: Complex STFT (n_fft/2 + 1, n_frames)
            frequencies: Frequency values
            times: Time values
        """
        # Create window
        if window == 'hann':
            win = signal.hann(n_fft)
        elif window == 'hamming':
            win = signal.hamming(n_fft)
        elif window == 'blackman':
            win = signal.blackman(n_fft)
        else:
            win = np.ones(n_fft)
        
        # Compute STFT
        f, t, Sxx = signal.spectrogram(
            audio,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            window=win,
            return_onesided=True
        )
        
        return Sxx, f, t
    
    @staticmethod
    def spectrogram_power(audio: np.ndarray, sample_rate: int = 16000,
                         n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Compute power spectrogram
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        n_fft : int
            FFT size
        hop_length : int
            Hop length
        
        Returns:
        --------
        ndarray : Power spectrogram (n_fft/2 + 1, n_frames)
        """
        f, t, Sxx = Spectrogram.stft(audio, n_fft, hop_length)
        power = np.abs(Sxx) ** 2
        return power
    
    @staticmethod
    def spectrogram_db(audio: np.ndarray, sample_rate: int = 16000,
                      n_fft: int = 2048, hop_length: int = 512,
                      ref: float = 1.0) -> np.ndarray:
        """
        Compute dB-scaled power spectrogram
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        n_fft : int
            FFT size
        hop_length : int
            Hop length
        ref : float
            Reference value
        
        Returns:
        --------
        ndarray : dB-scaled spectrogram
        """
        power = Spectrogram.spectrogram_power(audio, sample_rate, n_fft, hop_length)
        
        # Convert to dB
        S_db = 10 * np.log10(power + 1e-10)
        
        return S_db


class MFCCExtractor:
    """Mel-Frequency Cepstral Coefficients (MFCC) extraction"""
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13,
                n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize MFCC extractor
        
        Parameters:
        -----------
        sample_rate : int
            Sample rate
        n_mfcc : int
            Number of MFCCs to return
        n_fft : int
            FFT size
        hop_length : int
            Hop length
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_fb = None
    
    def _create_mel_filterbank(self, n_fft: int, n_mels: int) -> np.ndarray:
        """
        Create Mel-spaced triangular filterbank
        
        Parameters:
        -----------
        n_fft : int
            FFT size
        n_mels : int
            Number of mel bands
        
        Returns:
        --------
        ndarray : Mel filterbank (n_mels, n_fft/2 + 1)
        """
        # Frequency range
        n_freqs = n_fft // 2 + 1
        f_max = self.sample_rate / 2
        
        # Convert Hz to Mel
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Mel points
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        
        # Convert back to Hz
        hz_points = mel_to_hz(mel_points)
        
        # Frequency bins
        freqs = np.linspace(0, f_max, n_freqs)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, n_freqs))
        
        for m in range(n_mels):
            f_left = hz_points[m]
            f_center = hz_points[m + 1]
            f_right = hz_points[m + 2]
            
            # Left slope
            left_mask = (freqs >= f_left) & (freqs <= f_center)
            if f_center > f_left:
                filterbank[m, left_mask] = (freqs[left_mask] - f_left) / (f_center - f_left)
            
            # Right slope
            right_mask = (freqs >= f_center) & (freqs <= f_right)
            if f_right > f_center:
                filterbank[m, right_mask] = (f_right - freqs[right_mask]) / (f_right - f_center)
        
        return filterbank
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        
        Returns:
        --------
        ndarray : MFCC features (n_mfcc, n_frames)
        """
        # Step 1: Compute STFT
        stft_matrix, freqs, times = Spectrogram.stft(
            audio,
            self.n_fft,
            self.hop_length
        )
        
        # Step 2: Power spectrum
        power = np.abs(stft_matrix) ** 2
        
        # Step 3: Apply Mel filterbank
        if self.mel_fb is None:
            self.mel_fb = self._create_mel_filterbank(self.n_fft, 40)
        
        mel_spectrogram = np.dot(self.mel_fb, power)
        
        # Step 4: Log scale
        mel_log = np.log(mel_spectrogram + 1e-9)
        
        # Step 5: DCT (Discrete Cosine Transform)
        mfcc = fftpack.dct(mel_log, axis=0, type=2, norm='ortho')
        
        # Keep only n_mfcc coefficients
        mfcc = mfcc[:self.n_mfcc]
        
        return mfcc
    
    def extract_delta(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Compute delta (first derivative) of MFCC
        
        Parameters:
        -----------
        mfcc : ndarray
            MFCC features (n_mfcc, n_frames)
        
        Returns:
        --------
        ndarray : Delta MFCC
        """
        delta = np.diff(mfcc, axis=1)
        # Pad to same length
        delta = np.pad(delta, ((0, 0), (0, 1)), mode='edge')
        return delta
    
    def extract_delta_delta(self, mfcc: np.ndarray) -> np.ndarray:
        """Compute delta-delta (second derivative) of MFCC"""
        delta = self.extract_delta(mfcc)
        delta_delta = np.diff(delta, axis=1)
        delta_delta = np.pad(delta_delta, ((0, 0), (0, 1)), mode='edge')
        return delta_delta


class AudioFeatures:
    """Audio feature extraction"""
    
    @staticmethod
    def zero_crossing_rate(audio: np.ndarray, frame_length: int = 2048,
                          hop_length: int = 512) -> np.ndarray:
        """
        Compute zero-crossing rate
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        frame_length : int
            Frame length
        hop_length : int
            Hop length
        
        Returns:
        --------
        ndarray : Zero-crossing rate per frame
        """
        zcr = np.zeros(int((len(audio) - frame_length) / hop_length) + 1)
        
        for i, start in enumerate(range(0, len(audio) - frame_length, hop_length)):
            frame = audio[start:start + frame_length]
            
            # Count zero crossings
            zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        
        return zcr
    
    @staticmethod
    def energy(audio: np.ndarray, frame_length: int = 2048,
              hop_length: int = 512) -> np.ndarray:
        """
        Compute short-time energy
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        frame_length : int
            Frame length
        hop_length : int
            Hop length
        
        Returns:
        --------
        ndarray : Energy per frame
        """
        energy_vals = np.zeros(int((len(audio) - frame_length) / hop_length) + 1)
        
        for i, start in enumerate(range(0, len(audio) - frame_length, hop_length)):
            frame = audio[start:start + frame_length]
            energy_vals[i] = np.sum(frame ** 2)
        
        return energy_vals
    
    @staticmethod
    def spectral_centroid(audio: np.ndarray, sample_rate: int = 16000,
                         n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Compute spectral centroid
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        n_fft : int
            FFT size
        hop_length : int
            Hop length
        
        Returns:
        --------
        ndarray : Spectral centroid per frame
        """
        stft_matrix, freqs, times = Spectrogram.stft(audio, n_fft, hop_length)
        magnitude = np.abs(stft_matrix)
        
        # Compute centroid
        centroid = np.dot(freqs, magnitude) / (np.sum(magnitude, axis=0) + 1e-10)
        
        return centroid
    
    @staticmethod
    def spectral_rolloff(audio: np.ndarray, sample_rate: int = 16000,
                        n_fft: int = 2048, hop_length: int = 512,
                        percentile: float = 0.85) -> np.ndarray:
        """
        Compute spectral roll-off
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        n_fft : int
            FFT size
        hop_length : int
            Hop length
        percentile : float
            Percentile for roll-off (typically 0.85 or 0.95)
        
        Returns:
        --------
        ndarray : Spectral roll-off per frame
        """
        stft_matrix, freqs, times = Spectrogram.stft(audio, n_fft, hop_length)
        magnitude = np.abs(stft_matrix)
        
        # Cumulative sum
        cumsum = np.cumsum(magnitude, axis=0)
        
        # Find frequency where cumsum exceeds percentile
        rolloff = np.zeros(magnitude.shape[1])
        
        for i in range(magnitude.shape[1]):
            total = cumsum[-1, i]
            threshold = percentile * total
            
            idx = np.where(cumsum[:, i] >= threshold)[0]
            if len(idx) > 0:
                rolloff[i] = freqs[idx[0]]
        
        return rolloff
    
    @staticmethod
    def chromagram(audio: np.ndarray, sample_rate: int = 16000,
                   n_fft: int = 4096, hop_length: int = 512) -> np.ndarray:
        """
        Compute chromagram (pitch-based features)
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        n_fft : int
            FFT size
        hop_length : int
            Hop length
        
        Returns:
        --------
        ndarray : Chromagram (12, n_frames)
        """
        stft_matrix, freqs, times = Spectrogram.stft(audio, n_fft, hop_length)
        magnitude = np.abs(stft_matrix) ** 2
        
        # Map to chromatic scale (12 pitches)
        # Reference frequency (A0)
        ref_freq = 27.5
        
        # Compute bin to semitone mapping
        bins_per_semitone = np.log2(freqs / ref_freq) * 12
        
        # Round to nearest semitone
        chromatic_bins = np.round(bins_per_semitone) % 12
        
        # Sum magnitude for each pitch class
        chromagram = np.zeros((12, magnitude.shape[1]))
        
        for i in range(len(freqs)):
            pitch_class = int(chromatic_bins[i]) % 12
            chromagram[pitch_class] += magnitude[i]
        
        return chromagram


class AudioStatistics:
    """Compute audio statistics"""
    
    @staticmethod
    def compute_statistics(audio: np.ndarray) -> dict:
        """
        Compute basic audio statistics
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        
        Returns:
        --------
        dict : Statistics dictionary
        """
        return {
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'min': float(np.min(audio)),
            'max': float(np.max(audio)),
            'median': float(np.median(audio)),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'peak': float(np.max(np.abs(audio))),
            'crest_factor': float(np.max(np.abs(audio)) / np.sqrt(np.mean(audio ** 2)) + 1e-10)
        }
    
    @staticmethod
    def compute_frame_statistics(audio: np.ndarray, frame_length: int = 2048,
                                hop_length: int = 512) -> dict:
        """
        Compute frame-by-frame statistics
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        frame_length : int
            Frame length
        hop_length : int
            Hop length
        
        Returns:
        --------
        dict : Frame statistics
        """
        frame_means = []
        frame_stds = []
        frame_rmss = []
        
        for start in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start:start + frame_length]
            frame_means.append(np.mean(frame))
            frame_stds.append(np.std(frame))
            frame_rmss.append(np.sqrt(np.mean(frame ** 2)))
        
        return {
            'frame_means': np.array(frame_means),
            'frame_stds': np.array(frame_stds),
            'frame_rmss': np.array(frame_rmss),
            'mean_of_means': np.mean(frame_means),
            'mean_of_stds': np.mean(frame_stds),
            'mean_of_rmss': np.mean(frame_rmss)
        }
