import numpy as np
from typing import List, Tuple, Optional


class SceneDetection:
    """Scene and shot detection"""
    
    @staticmethod
    def histogram_difference(frame1: np.ndarray, frame2: np.ndarray,
                            bins: int = 256) -> float:
        """
        Compute histogram difference between two frames
        
        Parameters:
        -----------
        frame1, frame2 : ndarray
            Input frames
        bins : int
            Number of histogram bins
        
        Returns:
        --------
        float : Histogram distance (0-1)
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1 = np.mean(frame1, axis=2).astype(np.uint8)
        if len(frame2.shape) == 3:
            frame2 = np.mean(frame2, axis=2).astype(np.uint8)
        
        # Compute histograms
        hist1, _ = np.histogram(frame1.flatten(), bins=bins, range=(0, 256))
        hist2, _ = np.histogram(frame2.flatten(), bins=bins, range=(0, 256))
        
        # Normalize
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)
        
        # Chi-square distance
        diff = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
        
        return float(diff)
    
    @staticmethod
    def chi_square_distance(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Chi-square distance between frames
        
        Parameters:
        -----------
        frame1, frame2 : ndarray
            Input frames
        
        Returns:
        --------
        float : Chi-square distance
        """
        return SceneDetection.histogram_difference(frame1, frame2)
    
    @staticmethod
    def pixel_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute pixel-wise difference (MAE)
        
        Parameters:
        -----------
        frame1, frame2 : ndarray
            Input frames
        
        Returns:
        --------
        float : Mean Absolute Error
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1 = np.mean(frame1, axis=2)
        if len(frame2.shape) == 3:
            frame2 = np.mean(frame2, axis=2)
        
        mae = np.mean(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))
        
        return float(mae)
    
    @staticmethod
    def structural_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute structural similarity (SSIM)
        Simplified implementation
        
        Parameters:
        -----------
        frame1, frame2 : ndarray
            Input frames
        
        Returns:
        --------
        float : SSIM value (-1 to 1)
        """
        # Convert to float
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)
        
        if len(frame1.shape) == 3:
            frame1 = np.mean(frame1, axis=2)
        if len(frame2.shape) == 3:
            frame2 = np.mean(frame2, axis=2)
        
        # Constants
        C1 = 6.5025
        C2 = 58.5225
        
        # Means
        mu1 = frame1.mean()
        mu2 = frame2.mean()
        
        # Variances
        var1 = np.var(frame1)
        var2 = np.var(frame2)
        
        # Covariance
        cov = np.mean((frame1 - mu1) * (frame2 - mu2))
        
        # SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2) + 1e-10)
        
        return float(ssim)
    
    @staticmethod
    def detect_scenes(frames: np.ndarray, threshold: float = 0.5,
                     method: str = 'histogram') -> List[int]:
        """
        Detect scene changes/cuts in video
        
        Parameters:
        -----------
        frames : ndarray
            Video frames (n_frames, height, width, 3)
        threshold : float
            Detection threshold
        method : str
            Detection method ('histogram', 'pixel', 'ssim')
        
        Returns:
        --------
        list : Scene cut frame indices
        """
        scene_cuts = [0]  # First frame is always a scene
        
        if method == 'histogram':
            distance_fn = SceneDetection.histogram_difference
        elif method == 'pixel':
            distance_fn = SceneDetection.pixel_difference
        elif method == 'ssim':
            distance_fn = lambda f1, f2: 1 - SceneDetection.structural_similarity(f1, f2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute distances
        distances = []
        for i in range(len(frames) - 1):
            dist = distance_fn(frames[i], frames[i+1])
            distances.append(dist)
            
            if dist > threshold:
                scene_cuts.append(i + 1)
        
        return scene_cuts
    
    @staticmethod
    def detect_transitions(frames: np.ndarray, window: int = 5,
                          threshold: float = 0.3) -> List[int]:
        """
        Detect transitions (gradual scene changes)
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        window : int
            Window size for transition detection
        threshold : float
            Transition threshold
        
        Returns:
        --------
        list : Transition frame indices
        """
        transitions = []
        
        # Compute frame differences
        distances = []
        for i in range(len(frames) - 1):
            dist = SceneDetection.histogram_difference(frames[i], frames[i+1])
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Find sustained increases (transitions)
        for i in range(window, len(distances) - window):
            # Average distance in window before and after
            before = np.mean(distances[i-window:i])
            after = np.mean(distances[i:i+window])
            
            if abs(after - before) > threshold and after > threshold:
                transitions.append(i)
        
        return transitions
    
    @staticmethod
    def detect_flashframes(frames: np.ndarray, threshold: float = 0.9) -> List[int]:
        """
        Detect flash frames (very bright/dark frames)
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        threshold : float
            Flash threshold (0-1)
        
        Returns:
        --------
        list : Flash frame indices
        """
        flash_frames = []
        
        for i, frame in enumerate(frames):
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame
            
            # Compute brightness
            brightness = np.mean(gray) / 255.0
            
            # Check if very bright or very dark
            if brightness > threshold or brightness < (1 - threshold):
                flash_frames.append(i)
        
        return flash_frames
    
    @staticmethod
    def detect_blackframes(frames: np.ndarray, threshold: float = 0.1) -> List[int]:
        """
        Detect black/dark frames
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        threshold : float
            Darkness threshold (0-1)
        
        Returns:
        --------
        list : Black frame indices
        """
        black_frames = []
        
        for i, frame in enumerate(frames):
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame
            
            # Compute brightness
            brightness = np.mean(gray) / 255.0
            
            # Check if dark
            if brightness < threshold:
                black_frames.append(i)
        
        return black_frames


class ShotBoundaryDetection:
    """Shot boundary detection using multiple methods"""
    
    @staticmethod
    def adaptive_threshold_detection(frames: np.ndarray,
                                    initial_threshold: float = 0.5,
                                    adaptive: bool = True) -> List[int]:
        """
        Detect shot boundaries with adaptive threshold
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        initial_threshold : float
            Initial detection threshold
        adaptive : bool
            Use adaptive threshold
        
        Returns:
        --------
        list : Shot boundary frame indices
        """
        shot_boundaries = [0]
        
        # Compute distances
        distances = []
        for i in range(len(frames) - 1):
            dist = SceneDetection.histogram_difference(frames[i], frames[i+1])
            distances.append(dist)
        
        distances = np.array(distances)
        
        if adaptive:
            # Compute adaptive threshold
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            threshold = mean_dist + initial_threshold * std_dist
        else:
            threshold = initial_threshold
        
        # Detect boundaries
        for i, dist in enumerate(distances):
            if dist > threshold:
                shot_boundaries.append(i + 1)
        
        return shot_boundaries
    
    @staticmethod
    def two_pass_detection(frames: np.ndarray,
                         threshold1: float = 0.3,
                         threshold2: float = 0.7) -> List[int]:
        """
        Two-pass detection for better accuracy
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        threshold1 : float
            First pass threshold
        threshold2 : float
            Second pass threshold
        
        Returns:
        --------
        list : Shot boundary frame indices
        """
        # First pass - detect candidates
        candidates = SceneDetection.detect_scenes(frames, threshold1, 'histogram')
        
        # Second pass - filter with stricter threshold
        filtered = [candidates[0]]
        
        for i in range(1, len(candidates)):
            frame_idx = candidates[i]
            
            # Check distance to previous confirmed boundary
            prev_idx = filtered[-1]
            
            dist = SceneDetection.histogram_difference(
                frames[prev_idx], frames[frame_idx]
            )
            
            if dist > threshold2:
                filtered.append(frame_idx)
        
        return filtered
