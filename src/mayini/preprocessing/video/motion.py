import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


class OpticalFlow:
    """Optical flow computation"""
    
    @staticmethod
    def lucas_kanade(frame1: np.ndarray, frame2: np.ndarray,
                    window_size: int = 15,
                    num_iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lucas-Kanade optical flow
        
        Parameters:
        -----------
        frame1 : ndarray
            First frame (grayscale)
        frame2 : ndarray
            Second frame (grayscale)
        window_size : int
            Size of neighborhood window
        num_iterations : int
            Number of iterations
        
        Returns:
        --------
        tuple : (u, v) - x and y components of optical flow
        """
        if frame1.dtype != np.float32:
            frame1 = frame1.astype(np.float32)
        if frame2.dtype != np.float32:
            frame2 = frame2.astype(np.float32)
        
        # Compute gradients
        Ix = ndimage.sobel(frame1, axis=1)
        Iy = ndimage.sobel(frame1, axis=0)
        It = frame2 - frame1
        
        # Initialize flow
        u = np.zeros_like(frame1)
        v = np.zeros_like(frame1)
        
        # Window radius
        rad = window_size // 2
        
        # For each pixel
        for i in range(rad, frame1.shape[0] - rad):
            for j in range(rad, frame1.shape[1] - rad):
                # Get window
                Ix_w = Ix[i-rad:i+rad+1, j-rad:j+rad+1].flatten()
                Iy_w = Iy[i-rad:i+rad+1, j-rad:j+rad+1].flatten()
                It_w = It[i-rad:i+rad+1, j-rad:j+rad+1].flatten()
                
                # Build system (AtA)^-1 * At * b
                A = np.column_stack([Ix_w, Iy_w])
                
                try:
                    # Solve least squares
                    AtA = A.T @ A
                    At_b = A.T @ (-It_w)
                    
                    flow = np.linalg.solve(AtA + np.eye(2) * 1e-10, At_b)
                    u[i, j] = flow[0]
                    v[i, j] = flow[1]
                
                except:
                    # Singular matrix
                    pass
        
        return u, v
    
    @staticmethod
    def horn_schunck(frame1: np.ndarray, frame2: np.ndarray,
                    alpha: float = 0.01,
                    num_iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Horn-Schunck optical flow (global motion)
        
        Parameters:
        -----------
        frame1 : ndarray
            First frame (grayscale)
        frame2 : ndarray
            Second frame (grayscale)
        alpha : float
            Smoothness parameter
        num_iterations : int
            Number of iterations
        
        Returns:
        --------
        tuple : (u, v) - x and y components of optical flow
        """
        if frame1.dtype != np.float32:
            frame1 = frame1.astype(np.float32)
        if frame2.dtype != np.float32:
            frame2 = frame2.astype(np.float32)
        
        # Compute gradients
        Ix = ndimage.sobel(frame1, axis=1)
        Iy = ndimage.sobel(frame1, axis=0)
        It = frame2 - frame1
        
        # Initialize flow
        u = np.zeros_like(frame1)
        v = np.zeros_like(frame1)
        
        # Iterative refinement
        for _ in range(num_iterations):
            # Average flow in neighborhood
            u_avg = ndimage.uniform_filter(u, size=3)
            v_avg = ndimage.uniform_filter(v, size=3)
            
            # Compute update
            denom = alpha**2 + Ix**2 + Iy**2
            
            u = u_avg - Ix * ((Ix * u_avg + Iy * v_avg + It) / (denom + 1e-10))
            v = v_avg - Iy * ((Ix * u_avg + Iy * v_avg + It) / (denom + 1e-10))
        
        return u, v
    
    @staticmethod
    def compute_flow_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute optical flow magnitude
        
        Parameters:
        -----------
        u, v : ndarray
            Flow components
        
        Returns:
        --------
        ndarray : Flow magnitude
        """
        magnitude = np.sqrt(u**2 + v**2)
        return magnitude
    
    @staticmethod
    def compute_flow_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute optical flow angle
        
        Parameters:
        -----------
        u, v : ndarray
            Flow components
        
        Returns:
        --------
        ndarray : Flow angle in radians
        """
        angle = np.arctan2(v, u)
        return angle
    
    @staticmethod
    def visualize_flow(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Create flow visualization using HSV color space
        
        Parameters:
        -----------
        u, v : ndarray
            Flow components
        
        Returns:
        --------
        ndarray : Visualization image
        """
        magnitude = OpticalFlow.compute_flow_magnitude(u, v)
        angle = OpticalFlow.compute_flow_angle(u, v)
        
        # Normalize
        magnitude = (magnitude / (np.max(magnitude) + 1e-10) * 255).astype(np.uint8)
        angle = ((angle + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        
        # Create HSV image
        hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
        hsv[:, :, 0] = angle  # Hue
        hsv[:, :, 1] = 255    # Saturation
        hsv[:, :, 2] = magnitude  # Value
        
        return hsv


class MotionDetection:
    """Motion detection methods"""
    
    @staticmethod
    def frame_difference(frame1: np.ndarray, frame2: np.ndarray,
                        threshold: int = 30) -> np.ndarray:
        """
        Simple frame difference motion detection
        
        Parameters:
        -----------
        frame1 : ndarray
            First frame
        frame2 : ndarray
            Second frame
        threshold : int
            Difference threshold
        
        Returns:
        --------
        ndarray : Binary motion mask
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1 = np.mean(frame1, axis=2)
        if len(frame2.shape) == 3:
            frame2 = np.mean(frame2, axis=2)
        
        # Compute difference
        diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
        
        # Threshold
        motion_mask = diff > threshold
        
        return motion_mask
    
    @staticmethod
    def motion_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute motion magnitude from optical flow
        
        Parameters:
        -----------
        u, v : ndarray
            Optical flow components
        
        Returns:
        --------
        ndarray : Motion magnitude
        """
        motion = np.sqrt(u**2 + v**2)
        return motion
    
    @staticmethod
    def motion_histogram(u: np.ndarray, v: np.ndarray,
                        n_bins: int = 8) -> np.ndarray:
        """
        Compute motion direction histogram
        
        Parameters:
        -----------
        u, v : ndarray
            Optical flow components
        n_bins : int
            Number of bins
        
        Returns:
        --------
        ndarray : Motion histogram
        """
        angle = np.arctan2(v, u)
        angle = (angle + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        
        hist, _ = np.histogram(angle.flatten(), bins=n_bins, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        
        return hist
    
    @staticmethod
    def detect_moving_regions(frames: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Detect moving regions in video sequence
        
        Parameters:
        -----------
        frames : ndarray
            Video frames (n_frames, height, width, 3)
        threshold : float
            Motion threshold
        
        Returns:
        --------
        ndarray : Motion masks for each frame
        """
        motion_masks = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale
            gray1 = np.mean(frames[i], axis=2)
            gray2 = np.mean(frames[i+1], axis=2)
            
            # Compute motion
            motion_mask = MotionDetection.frame_difference(gray1, gray2, 
                                                          threshold=int(threshold * 255))
            motion_masks.append(motion_mask)
        
        # Pad last frame
        motion_masks.append(motion_masks[-1])
        
        return np.array(motion_masks, dtype=np.uint8)
    
    @staticmethod
    def compute_motion_statistics(u: np.ndarray, v: np.ndarray) -> dict:
        """
        Compute motion statistics
        
        Parameters:
        -----------
        u, v : ndarray
            Optical flow components
        
        Returns:
        --------
        dict : Motion statistics
        """
        magnitude = OpticalFlow.compute_flow_magnitude(u, v)
        
        return {
            'mean_magnitude': float(np.mean(magnitude)),
            'max_magnitude': float(np.max(magnitude)),
            'min_magnitude': float(np.min(magnitude)),
            'std_magnitude': float(np.std(magnitude)),
            'mean_u': float(np.mean(u)),
            'mean_v': float(np.mean(v)),
            'std_u': float(np.std(u)),
            'std_v': float(np.std(v)),
            'dominant_direction': float(OpticalFlow.compute_flow_angle(
                np.mean(u), np.mean(v)
            ))
        }
