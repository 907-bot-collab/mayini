import numpy as np
from scipy import signal, interpolate
from typing import Tuple, Optional


class AudioEffects:
    """Audio effect processing"""
    
    @staticmethod
    def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Time stretching (change speed without changing pitch)
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        rate : float
            Stretching factor (>1 = slower, <1 = faster)
        
        Returns:
        --------
        ndarray : Time-stretched audio
        """
        if rate == 1.0:
            return audio
        
        # Resample using phase vocoder-like approach
        # Simple approach: interpolation
        n_samples = int(len(audio) / rate)
        
        # Create new time indices
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, n_samples)
        
        # Interpolate
        stretched = np.interp(new_indices, old_indices, audio)
        
        return stretched
    
    @staticmethod
    def pitch_shift(audio: np.ndarray, sample_rate: int, n_semitones: float) -> np.ndarray:
        """
        Pitch shifting (change pitch without changing speed)
        Simple implementation using time-stretching + resampling
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        n_semitones : float
            Number of semitones to shift (can be negative)
        
        Returns:
        --------
        ndarray : Pitch-shifted audio
        """
        # Cents per semitone = 100
        # Frequency ratio = 2^(n_semitones/12)
        freq_ratio = 2 ** (n_semitones / 12.0)
        
        # Time-stretch by inverse ratio
        stretched = AudioEffects.time_stretch(audio, 1.0 / freq_ratio)
        
        # Resample
        n_samples = int(len(stretched) * freq_ratio)
        old_indices = np.arange(len(stretched))
        new_indices = np.linspace(0, len(stretched) - 1, n_samples)
        
        resampled = np.interp(new_indices, old_indices, stretched)
        
        return resampled
    
    @staticmethod
    def add_reverb(audio: np.ndarray, sample_rate: int = 16000,
                  room_scale: float = 0.5, dampening: float = 0.5) -> np.ndarray:
        """
        Add reverb effect (simplified Schroeder reverb)
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        room_scale : float
            Room size (0-1)
        dampening : float
            Dampening (0-1)
        
        Returns:
        --------
        ndarray : Reverberated audio
        """
        # Delay line lengths (in samples)
        delays = [
            int(0.0297 * sample_rate * room_scale),
            int(0.0371 * sample_rate * room_scale),
            int(0.0411 * sample_rate * room_scale),
            int(0.0437 * sample_rate * room_scale)
        ]
        
        # Create output
        output = np.zeros_like(audio)
        
        # Comb filters (parallel)
        for delay in delays:
            delayed = np.zeros_like(audio)
            
            for i in range(len(audio)):
                if i >= delay:
                    # Apply dampening
                    delayed[i] = audio[i] + dampening * delayed[i - delay]
                else:
                    delayed[i] = audio[i]
            
            output += delayed
        
        # Average
        output = output / len(delays)
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        return output
    
    @staticmethod
    def add_echo(audio: np.ndarray, sample_rate: int = 16000,
                delay: float = 0.5, decay: float = 0.5) -> np.ndarray:
        """
        Add echo effect
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        delay : float
            Delay in seconds
        decay : float
            Decay factor (0-1)
        
        Returns:
        --------
        ndarray : Audio with echo
        """
        delay_samples = int(delay * sample_rate)
        output = np.zeros(len(audio) + delay_samples)
        
        # Original audio
        output[:len(audio)] = audio
        
        # Echo
        output[delay_samples:delay_samples + len(audio)] += decay * audio
        
        # Trim to original length
        output = output[:len(audio)]
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        return output
    
    @staticmethod
    def add_chorus(audio: np.ndarray, sample_rate: int = 16000,
                  rate: float = 1.5, depth: float = 0.002) -> np.ndarray:
        """
        Add chorus effect (modulated delay)
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        rate : float
            LFO rate in Hz
        depth : float
            Depth of modulation in seconds
        
        Returns:
        --------
        ndarray : Audio with chorus
        """
        n_samples = len(audio)
        t = np.arange(n_samples) / sample_rate
        
        # LFO (Low Frequency Oscillator)
        lfo = depth * np.sin(2 * np.pi * rate * t)
        
        # Delay in samples
        base_delay = int(0.03 * sample_rate)  # 30 ms base delay
        delay_samples = base_delay + (lfo * sample_rate).astype(int)
        
        # Apply variable delay
        output = np.zeros_like(audio)
        
        for i in range(n_samples):
            delay = delay_samples[i]
            if i >= delay:
                output[i] = audio[i] + 0.5 * audio[i - delay]
            else:
                output[i] = audio[i]
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        return output
    
    @staticmethod
    def add_tremolo(audio: np.ndarray, sample_rate: int = 16000,
                   rate: float = 5.0, depth: float = 0.5) -> np.ndarray:
        """
        Add tremolo effect (amplitude modulation)
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        rate : float
            Modulation rate in Hz
        depth : float
            Depth of modulation (0-1)
        
        Returns:
        --------
        ndarray : Audio with tremolo
        """
        n_samples = len(audio)
        t = np.arange(n_samples) / sample_rate
        
        # LFO
        lfo = 1 - depth + depth * np.sin(2 * np.pi * rate * t)
        
        # Apply modulation
        output = audio * lfo
        
        return output
    
    @staticmethod
    def add_vibrato(audio: np.ndarray, sample_rate: int = 16000,
                   rate: float = 5.0, depth: float = 0.002) -> np.ndarray:
        """
        Add vibrato effect (frequency modulation)
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        rate : float
            Vibrato rate in Hz
        depth : float
            Depth of vibrato in seconds
        
        Returns:
        --------
        ndarray : Audio with vibrato
        """
        n_samples = len(audio)
        t = np.arange(n_samples) / sample_rate
        
        # LFO
        lfo = depth * np.sin(2 * np.pi * rate * t)
        
        # Variable delay
        delay_samples = lfo * sample_rate
        
        output = np.zeros_like(audio)
        
        for i in range(n_samples):
            delay = delay_samples[i]
            
            if delay >= 0 and i >= int(delay) + 1:
                # Linear interpolation
                delay_int = int(delay)
                delay_frac = delay - delay_int
                
                s1 = audio[i - delay_int - 1] if i > delay_int else 0
                s2 = audio[i - delay_int] if i > delay_int else audio[i]
                
                output[i] = s1 * delay_frac + s2 * (1 - delay_frac)
            else:
                output[i] = audio[i]
        
        return output


class AudioAugmentation:
    """Audio augmentation techniques"""
    
    @staticmethod
    def add_background_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        Add background Gaussian noise
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        noise_factor : float
            Noise magnitude
        
        Returns:
        --------
        ndarray : Audio with noise
        """
        noise = noise_factor * np.random.randn(len(audio))
        noisy = audio + noise
        
        # Normalize if necessary
        max_val = np.max(np.abs(noisy))
        if max_val > 1.0:
            noisy = noisy / max_val
        
        return noisy
    
    @staticmethod
    def add_background_hum(audio: np.ndarray, sample_rate: int = 16000,
                          freq: float = 50.0, magnitude: float = 0.01) -> np.ndarray:
        """
        Add background hum (e.g., 50/60 Hz power line hum)
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        freq : float
            Hum frequency (50 or 60 Hz)
        magnitude : float
            Hum magnitude
        
        Returns:
        --------
        ndarray : Audio with hum
        """
        n_samples = len(audio)
        t = np.arange(n_samples) / sample_rate
        
        hum = magnitude * np.sin(2 * np.pi * freq * t)
        
        audio_with_hum = audio + hum
        
        # Normalize if necessary
        max_val = np.max(np.abs(audio_with_hum))
        if max_val > 1.0:
            audio_with_hum = audio_with_hum / max_val
        
        return audio_with_hum
    
    @staticmethod
    def random_gain(audio: np.ndarray, min_gain: float = 0.5,
                   max_gain: float = 2.0) -> np.ndarray:
        """
        Apply random gain
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        min_gain : float
            Minimum gain
        max_gain : float
            Maximum gain
        
        Returns:
        --------
        ndarray : Audio with random gain
        """
        gain = np.random.uniform(min_gain, max_gain)
        augmented = audio * gain
        
        # Normalize if necessary
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / max_val
        
        return augmented
    
    @staticmethod
    def time_shift(audio: np.ndarray, max_shift: float = 0.1,
                  sample_rate: int = 16000) -> np.ndarray:
        """
        Shift audio in time (with circular padding or zero padding)
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        max_shift : float
            Maximum shift in seconds
        sample_rate : int
            Sample rate
        
        Returns:
        --------
        ndarray : Time-shifted audio
        """
        shift_samples = np.random.randint(-int(max_shift * sample_rate),
                                         int(max_shift * sample_rate))
        
        return np.roll(audio, shift_samples)
    
    @staticmethod
    def pitch_shift_random(audio: np.ndarray, sample_rate: int = 16000,
                          max_semitones: float = 2.0) -> np.ndarray:
        """
        Apply random pitch shift
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        max_semitones : float
            Maximum pitch shift in semitones
        
        Returns:
        --------
        ndarray : Pitch-shifted audio
        """
        n_semitones = np.random.uniform(-max_semitones, max_semitones)
        
        return AudioEffects.pitch_shift(audio, sample_rate, n_semitones)
    
    @staticmethod
    def time_stretch_random(audio: np.ndarray, min_rate: float = 0.8,
                           max_rate: float = 1.2) -> np.ndarray:
        """
        Apply random time stretch
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        min_rate : float
            Minimum stretch rate
        max_rate : float
            Maximum stretch rate
        
        Returns:
        --------
        ndarray : Time-stretched audio
        """
        rate = np.random.uniform(min_rate, max_rate)
        
        return AudioEffects.time_stretch(audio, rate)
    
    @staticmethod
    def concatenate_with_silence(audio: np.ndarray, sample_rate: int = 16000,
                                silence_duration: float = 0.5) -> np.ndarray:
        """
        Add silence before and/or after audio
        
        Parameters:
        -----------
        audio : ndarray
            Audio signal
        sample_rate : int
            Sample rate
        silence_duration : float
            Duration of silence in seconds
        
        Returns:
        --------
        ndarray : Audio with silence
        """
        silence_samples = int(silence_duration * sample_rate)
        silence = np.zeros(silence_samples)
        
        # Randomly add silence before or after
        if np.random.rand() < 0.5:
            result = np.concatenate([silence, audio])
        else:
            result = np.concatenate([audio, silence])
        
        return result
