import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict


class AudioAnalysis:
    """Audio analysis tools"""
    
    @staticmethod
    def detect_silence(audio: np.ndarray, sample_rate: int = 16000,
                      threshold: float = -40.0) -> np.ndarray:
        """
        Detect silent frames
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        threshold : float
            Silence threshold in dB
        
        Returns:
        --------
        ndarray : Boolean array indicating silent frames
        """
        # Compute frame energy
        frame_length = int(0.02 * sample_rate)  # 20 ms frames
        hop_length = frame_length // 2
        
        n_frames = (len(audio) - frame_length) // hop_length + 1
        is_silent = np.zeros(n_frames, dtype=bool)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            
            # Energy in dB
            energy = np.sum(frame ** 2)
            energy_db = 10 * np.log10(energy + 1e-10)
            
            is_silent[i] = energy_db < threshold
        
        return is_silent
    
    @staticmethod
    def estimate_tempo(audio: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Estimate tempo in BPM
        Simplified beat tracking
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        
        Returns:
        --------
        float : Estimated tempo in BPM
        """
        # Compute onset strength
        frame_length = int(0.02 * sample_rate)
        hop_length = frame_length // 2
        
        n_frames = (len(audio) - frame_length) // hop_length
        onset_strength = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            
            # Spectral flux as onset measure
            if i > 0:
                onset_strength[i] = np.sum(np.abs(np.diff(np.abs(
                    np.fft.rfft(frame)
                ))))
        
        # Find dominant beat period using autocorrelation
        if len(onset_strength) > 1:
            acf = np.correlate(onset_strength, onset_strength, mode='full')
            acf = acf[len(acf) // 2:]
            
            # Find peaks
            peaks, _ = signal.find_peaks(acf, height=0.1 * np.max(acf))
            
            if len(peaks) > 0:
                # First peak after lag 0
                beat_lag = peaks[0]
                beat_period = (beat_lag * hop_length) / sample_rate
                
                if beat_period > 0:
                    tempo = 60.0 / beat_period
                    # Constrain to realistic tempo range
                    tempo = np.clip(tempo, 40, 200)
                    return float(tempo)
        
        return 120.0  # Default
    
    @staticmethod
    def detect_onset(audio: np.ndarray, sample_rate: int = 16000,
                    threshold: float = 0.3) -> np.ndarray:
        """
        Detect onset times (beat times)
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        threshold : float
            Detection threshold (0-1)
        
        Returns:
        --------
        ndarray : Onset times in seconds
        """
        # Compute onset strength signal
        frame_length = int(0.02 * sample_rate)
        hop_length = frame_length // 2
        
        n_frames = (len(audio) - frame_length) // hop_length
        onset_strength = np.zeros(n_frames)
        
        prev_spectrum = np.zeros(frame_length // 2)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            
            # Compute spectrum
            spectrum = np.abs(np.fft.rfft(frame))
            
            # Spectral flux
            if i > 0:
                flux = np.sum(np.maximum(spectrum - prev_spectrum, 0))
                onset_strength[i] = flux
            
            prev_spectrum = spectrum
        
        # Normalize
        onset_strength = onset_strength / (np.max(onset_strength) + 1e-10)
        
        # Detect peaks
        peaks, _ = signal.find_peaks(onset_strength, height=threshold)
        
        # Convert to time
        onset_times = (peaks * hop_length) / sample_rate
        
        return onset_times
    
    @staticmethod
    def estimate_pitch(audio: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Estimate fundamental frequency (pitch)
        Using autocorrelation method
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal (should be a single note or voiced segment)
        sample_rate : int
            Sample rate
        
        Returns:
        --------
        float : Estimated frequency in Hz
        """
        # Autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Find first minimum
        min_period = int(sample_rate / 400)  # Max 400 Hz
        max_period = int(sample_rate / 50)   # Min 50 Hz
        
        # Find peak in valid range
        valid_autocorr = autocorr[min_period:max_period]
        
        if len(valid_autocorr) > 0:
            peak_idx = np.argmax(valid_autocorr) + min_period
            
            if peak_idx > 0:
                frequency = sample_rate / peak_idx
                return float(frequency)
        
        return 0.0
    
    @staticmethod
    def compute_spectral_moments(audio: np.ndarray, sample_rate: int = 16000,
                                n_fft: int = 2048) -> Dict[str, float]:
        """
        Compute spectral moments
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        n_fft : int
            FFT size
        
        Returns:
        --------
        dict : Spectral moments (mean, variance, skewness, kurtosis)
        """
        # Compute FFT
        spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)
        
        # Power spectrum
        power = spectrum ** 2
        
        # Normalize
        power = power / np.sum(power)
        
        # Moments
        mean = np.sum(freqs * power)
        variance = np.sum((freqs - mean) ** 2 * power)
        skewness = np.sum((freqs - mean) ** 3 * power) / (variance ** 1.5 + 1e-10)
        kurtosis = np.sum((freqs - mean) ** 4 * power) / (variance ** 2 + 1e-10)
        
        return {
            'centroid': float(mean),
            'variance': float(variance),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'spread': float(np.sqrt(variance))
        }


class AudioQuality:
    """Audio quality metrics"""
    
    @staticmethod
    def signal_to_noise_ratio(signal_audio: np.ndarray,
                             noise_audio: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio (SNR)
        
        Parameters:
        -----------
        signal_audio : ndarray
            Signal
        noise_audio : ndarray
            Noise
        
        Returns:
        --------
        float : SNR in dB
        """
        signal_power = np.mean(signal_audio ** 2)
        noise_power = np.mean(noise_audio ** 2)
        
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return float(snr)
    
    @staticmethod
    def dynamic_range(audio: np.ndarray) -> float:
        """
        Compute dynamic range
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        
        Returns:
        --------
        float : Dynamic range in dB
        """
        # RMS levels in quiet and loud sections
        frame_length = len(audio) // 100
        
        if frame_length < 1:
            return 0.0
        
        frame_rms = []
        
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            frame_rms.append(rms)
        
        frame_rms = np.array(frame_rms)
        
        # Sort and get quiet and loud sections
        sorted_rms = np.sort(frame_rms)
        
        quiet_level = np.mean(sorted_rms[:len(sorted_rms) // 4])
        loud_level = np.mean(sorted_rms[-len(sorted_rms) // 4:])
        
        dr = 20 * np.log10((loud_level + 1e-10) / (quiet_level + 1e-10))
        
        return float(dr)
    
    @staticmethod
    def peak_amplitude(audio: np.ndarray) -> float:
        """Get peak amplitude"""
        return float(np.max(np.abs(audio)))
    
    @staticmethod
    def rms_level(audio: np.ndarray) -> float:
        """Get RMS level"""
        return float(np.sqrt(np.mean(audio ** 2)))


class FeatureComparison:
    """Compare audio features"""
    
    @staticmethod
    def euclidean_distance(features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two feature vectors
        
        Parameters:
        -----------
        features1, features2 : ndarray
            Feature vectors
        
        Returns:
        --------
        float : Euclidean distance
        """
        return float(np.sqrt(np.sum((features1 - features2) ** 2)))
    
    @staticmethod
    def cosine_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors
        
        Parameters:
        -----------
        features1, features2 : ndarray
            Feature vectors
        
        Returns:
        --------
        float : Cosine similarity (-1 to 1)
        """
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    @staticmethod
    def mfcc_distance(mfcc1: np.ndarray, mfcc2: np.ndarray) -> float:
        """
        Compute distance between two MFCC sequences
        Using Dynamic Time Warping (simplified)
        
        Parameters:
        -----------
        mfcc1, mfcc2 : ndarray
            MFCC sequences
        
        Returns:
        --------
        float : Distance
        """
        # Simple Euclidean distance on averaged MFCCs
        avg_mfcc1 = np.mean(mfcc1, axis=1)
        avg_mfcc2 = np.mean(mfcc2, axis=1)
        
        return FeatureComparison.euclidean_distance(avg_mfcc1, avg_mfcc2)
