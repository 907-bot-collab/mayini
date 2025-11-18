import numpy as np
from typing import Dict, List, Tuple, Optional


class TemporalFeatures:
    """Temporal feature extraction"""
    
    @staticmethod
    def temporal_gradient(frames: np.ndarray) -> np.ndarray:
        """
        Compute temporal gradient (difference between consecutive frames)
        
        Parameters:
        -----------
        frames : ndarray
            Video frames (n_frames, height, width, 3)
        
        Returns:
        --------
        ndarray : Temporal gradients
        """
        # Convert to grayscale
        gray_frames = np.mean(frames, axis=3)
        
        # Compute differences
        temporal_grad = np.diff(gray_frames, axis=0)
        
        # Pad first frame
        temporal_grad = np.concatenate([temporal_grad[0:1], temporal_grad], axis=0)
        
        return temporal_grad
    
    @staticmethod
    def motion_energy(frames: np.ndarray) -> np.ndarray:
        """
        Compute motion energy per frame
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        
        Returns:
        --------
        ndarray : Motion energy per frame
        """
        temporal_grad = TemporalFeatures.temporal_gradient(frames)
        
        # Compute energy
        motion_energy = np.sum(temporal_grad**2, axis=(1, 2))
        
        return motion_energy
    
    @staticmethod
    def temporal_coherence(frames: np.ndarray, window: int = 3) -> np.ndarray:
        """
        Compute temporal coherence (consistency across frames)
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        window : int
            Window size
        
        Returns:
        --------
        ndarray : Temporal coherence per frame
        """
        coherence = np.zeros(len(frames))
        
        for i in range(window, len(frames) - window):
            # Get window
            window_frames = frames[i-window:i+window+1]
            
            # Compute variance
            variance = np.var(window_frames, axis=0)
            
            # Average variance
            coherence[i] = np.mean(variance)
        
        # Pad edges
        coherence[:window] = coherence[window]
        coherence[-window:] = coherence[-window-1]
        
        return coherence
    
    @staticmethod
    def optical_flow_magnitude_temporal(u_sequence: List[np.ndarray],
                                       v_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Compute optical flow magnitude over time
        
        Parameters:
        -----------
        u_sequence : list
            Sequence of u flow components
        v_sequence : list
            Sequence of v flow components
        
        Returns:
        --------
        ndarray : Flow magnitude per frame
        """
        magnitudes = []
        
        for u, v in zip(u_sequence, v_sequence):
            mag = np.sqrt(u**2 + v**2)
            magnitudes.append(np.mean(mag))
        
        return np.array(magnitudes)
    
    @staticmethod
    def frame_difference_sequence(frames: np.ndarray) -> np.ndarray:
        """
        Compute frame-by-frame differences
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        
        Returns:
        --------
        ndarray : Frame differences
        """
        gray_frames = np.mean(frames, axis=3).astype(np.float32)
        
        differences = []
        
        for i in range(len(frames) - 1):
            diff = np.mean(np.abs(gray_frames[i+1] - gray_frames[i]))
            differences.append(diff)
        
        # Pad
        differences.append(differences[-1] if differences else 0)
        
        return np.array(differences)
    
    @staticmethod
    def compute_histogram_sequence(frames: np.ndarray, bins: int = 256) -> np.ndarray:
        """
        Compute histogram for each frame
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        bins : int
            Number of histogram bins
        
        Returns:
        --------
        ndarray : Histograms per frame (n_frames, bins)
        """
        gray_frames = np.mean(frames, axis=3).astype(np.uint8)
        
        histograms = []
        
        for frame in gray_frames:
            hist, _ = np.histogram(frame.flatten(), bins=bins, range=(0, 256))
            hist = hist / (hist.sum() + 1e-10)
            histograms.append(hist)
        
        return np.array(histograms)
    
    @staticmethod
    def compute_statistics(frames: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics over video sequence
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        
        Returns:
        --------
        dict : Video statistics
        """
        # Motion energy
        motion_energy_seq = TemporalFeatures.motion_energy(frames)
        
        # Frame differences
        frame_diffs = TemporalFeatures.frame_difference_sequence(frames)
        
        return {
            'n_frames': len(frames),
            'height': frames.shape[1],
            'width': frames.shape[2],
            'mean_brightness': float(np.mean(frames)),
            'std_brightness': float(np.std(frames)),
            'mean_motion': float(np.mean(motion_energy_seq)),
            'max_motion': float(np.max(motion_energy_seq)),
            'min_motion': float(np.min(motion_energy_seq)),
            'mean_frame_diff': float(np.mean(frame_diffs)),
            'std_frame_diff': float(np.std(frame_diffs)),
            'total_motion': float(np.sum(motion_energy_seq))
        }


class VideoStatistics:
    """Video statistics computation"""
    
    @staticmethod
    def compute_frame_statistics(frames: np.ndarray) -> np.ndarray:
        """
        Compute per-frame statistics
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        
        Returns:
        --------
        ndarray : Statistics per frame (n_frames, 5)
        """
        stats = np.zeros((len(frames), 5))
        
        for i, frame in enumerate(frames):
            gray = np.mean(frame, axis=2)
            stats[i, 0] = np.mean(gray)
            stats[i, 1] = np.std(gray)
            stats[i, 2] = np.min(gray)
            stats[i, 3] = np.max(gray)
            stats[i, 4] = np.median(gray)
        
        return stats
    
    @staticmethod
    def compute_color_histogram(frames: np.ndarray,
                               n_bins: int = 256) -> np.ndarray:
        """
        Compute color histogram for video
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        n_bins : int
            Number of bins
        
        Returns:
        --------
        ndarray : Combined color histogram
        """
        r_hist = np.zeros(n_bins)
        g_hist = np.zeros(n_bins)
        b_hist = np.zeros(n_bins)
        
        for frame in frames:
            r, g, b = np.histogram(frame[:, :, 0].flatten(), n_bins, (0, 256))[:1], \
                     np.histogram(frame[:, :, 1].flatten(), n_bins, (0, 256))[:1], \
                     np.histogram(frame[:, :, 2].flatten(), n_bins, (0, 256))[:1]
            
            r_hist += r[0]
            g_hist += g[0]
            b_hist += b[0]
        
        # Normalize
        r_hist = r_hist / (np.sum(r_hist) + 1e-10)
        g_hist = g_hist / (np.sum(g_hist) + 1e-10)
        b_hist = b_hist / (np.sum(b_hist) + 1e-10)
        
        # Concatenate
        combined = np.concatenate([r_hist, g_hist, b_hist])
        
        return combined
    
    @staticmethod
    def compute_edge_density(frames: np.ndarray) -> np.ndarray:
        """
        Compute edge density per frame
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        
        Returns:
        --------
        ndarray : Edge density per frame
        """
        from scipy import ndimage
        
        edge_density = []
        
        for frame in frames:
            # Convert to grayscale
            gray = np.mean(frame, axis=2).astype(np.uint8)
            
            # Compute edges using Sobel
            edges_x = ndimage.sobel(gray, axis=1)
            edges_y = ndimage.sobel(gray, axis=0)
            edges = np.sqrt(edges_x**2 + edges_y**2)
            
            # Compute density
            density = np.sum(edges > 0.1 * np.max(edges)) / edges.size
            edge_density.append(density)
        
        return np.array(edge_density)


class VideoAugmentation:
    """Video augmentation techniques"""
    
    @staticmethod
    def temporal_shift(frames: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Shift video frames temporally
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        shift : int
            Shift amount
        
        Returns:
        --------
        ndarray : Shifted frames
        """
        return np.roll(frames, shift, axis=0)
    
    @staticmethod
    def temporal_dropout(frames: np.ndarray, drop_rate: float = 0.1) -> np.ndarray:
        """
        Randomly remove frames (temporal dropout)
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        drop_rate : float
            Dropout rate
        
        Returns:
        --------
        ndarray : Frames with some removed
        """
        n_frames = len(frames)
        mask = np.random.rand(n_frames) > drop_rate
        
        return frames[mask]
    
    @staticmethod
    def temporal_interpolation(frames: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        Interpolate between frames (slow motion effect)
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        factor : int
            Interpolation factor
        
        Returns:
        --------
        ndarray : Interpolated frames
        """
        interpolated = []
        
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])
            
            # Linear interpolation between frames
            for t in np.linspace(0, 1, factor)[1:]:
                inter_frame = (1 - t) * frames[i] + t * frames[i+1]
                interpolated.append(inter_frame.astype(frames.dtype))
        
        interpolated.append(frames[-1])
        
        return np.array(interpolated)
    
    @staticmethod
    def random_temporal_crop(frames: np.ndarray, length: int) -> np.ndarray:
        """
        Random temporal crop
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        length : int
            Crop length
        
        Returns:
        --------
        ndarray : Cropped frames
        """
        if length >= len(frames):
            return frames
        
        start = np.random.randint(0, len(frames) - length + 1)
        
        return frames[start:start + length]
    
    @staticmethod
    def reverse_temporal(frames: np.ndarray) -> np.ndarray:
        """Reverse video frames"""
        return np.flip(frames, axis=0)
