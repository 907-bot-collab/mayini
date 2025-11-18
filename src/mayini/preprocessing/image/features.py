import numpy as np
from typing import Tuple, Optional
import math


class HistogramFeatures:
    """Histogram-based features"""
    
    @staticmethod
    def compute_histogram(image: np.ndarray, bins: int = 256,
                         range_tuple: Tuple = (0, 256)) -> np.ndarray:
        """
        Compute image histogram
        
        Parameters:
        -----------
        image : ndarray
            Input image (grayscale)
        bins : int
            Number of histogram bins
        range_tuple : tuple
            Value range
        
        Returns:
        --------
        ndarray : Histogram
        """
        histogram, _ = np.histogram(image.flatten(), bins=bins, range=range_tuple)
        return histogram / histogram.sum() if histogram.sum() > 0 else histogram
    
    @staticmethod
    def compute_histogram_2d(image: np.ndarray, bins: int = 256) -> np.ndarray:
        """
        2D histogram of image gradients
        
        Parameters:
        -----------
        image : ndarray
            Input image
        bins : int
            Number of bins in each dimension
        
        Returns:
        --------
        ndarray : 2D histogram
        """
        # Compute gradients
        gx = np.gradient(image, axis=1)
        gy = np.gradient(image, axis=0)
        
        # Histogram
        hist_2d, _, _ = np.histogram2d(gx.flatten(), gy.flatten(),
                                       bins=[bins, bins])
        
        return hist_2d / hist_2d.sum() if hist_2d.sum() > 0 else hist_2d
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        Histogram equalization
        
        Parameters:
        -----------
        image : ndarray
            Input image
        
        Returns:
        --------
        ndarray : Equalized image
        """
        # Compute histogram
        hist, bin_edges = np.histogram(image.flatten(), 256, [0, 256])
        
        # Compute CDF
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize
        
        # Map values
        equalized = cdf[image] * 255
        
        return equalized.astype(image.dtype)
    
    @staticmethod
    def histogram_matching(src_image: np.ndarray,
                          ref_image: np.ndarray) -> np.ndarray:
        """
        Match histogram of source image to reference image
        """
        src_hist, _ = np.histogram(src_image.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref_image.flatten(), 256, [0, 256])
        
        src_cdf = src_hist.cumsum()
        src_cdf = src_cdf / src_cdf[-1]
        
        ref_cdf = ref_hist.cumsum()
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # Mapping function
        matched = np.zeros_like(src_image, dtype=np.float32)
        
        for i in range(256):
            idx = np.abs(ref_cdf - src_cdf[i]).argmin()
            matched[src_image == i] = idx
        
        return np.clip(matched, 0, 255).astype(src_image.dtype)


class TextureFeatures:
    """Texture feature extraction"""
    
    @staticmethod
    def local_binary_pattern(image: np.ndarray, radius: int = 1,
                            n_points: int = 8) -> np.ndarray:
        """
        Local Binary Pattern (LBP) feature extraction
        
        Parameters:
        -----------
        image : ndarray
            Input image
        radius : int
            Radius of LBP
        n_points : int
            Number of sample points
        
        Returns:
        --------
        ndarray : LBP feature image
        """
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        
        for y in range(radius, height - radius):
            for x in range(radius, width - radius):
                center = image[y, x]
                
                pattern = 0
                for i, angle in enumerate(angles):
                    # Sample point coordinates
                    nx = int(x + radius * np.cos(angle))
                    ny = int(y + radius * np.sin(angle))
                    
                    # Clamp to image bounds
                    nx = min(max(nx, 0), width - 1)
                    ny = min(max(ny, 0), height - 1)
                    
                    # Compare with center
                    if image[ny, nx] >= center:
                        pattern |= (1 << i)
                
                lbp[y, x] = pattern
        
        return lbp
    
    @staticmethod
    def glcm(image: np.ndarray, distance: int = 1,
            angles: Optional[list] = None) -> dict:
        """
        Gray-Level Co-occurrence Matrix (GLCM)
        
        Parameters:
        -----------
        image : ndarray
            Input image
        distance : int
            Distance between pixels
        angles : list, optional
            Angles for co-occurrence
        
        Returns:
        --------
        dict : GLCM features
        """
        if angles is None:
            angles = [0, 45, 90, 135]
        
        features = {}
        
        for angle in angles:
            # Calculate co-occurrence matrix
            if angle == 0:
                pairs = [(image[y, x], image[y, x + distance])
                        for y in range(image.shape[0])
                        for x in range(image.shape[1] - distance)]
            elif angle == 90:
                pairs = [(image[y, x], image[y + distance, x])
                        for y in range(image.shape[0] - distance)
                        for x in range(image.shape[1])]
            else:
                continue
            
            # Count co-occurrences
            glcm = np.zeros((256, 256), dtype=np.float32)
            for (i, j) in pairs:
                glcm[int(i), int(j)] += 1
                glcm[int(j), int(i)] += 1
            
            # Normalize
            glcm = glcm / glcm.sum()
            
            # Compute features
            features[f'angle_{angle}'] = {
                'contrast': TextureFeatures._glcm_contrast(glcm),
                'dissimilarity': TextureFeatures._glcm_dissimilarity(glcm),
                'homogeneity': TextureFeatures._glcm_homogeneity(glcm),
                'energy': TextureFeatures._glcm_energy(glcm),
                'correlation': TextureFeatures._glcm_correlation(glcm)
            }
        
        return features
    
    @staticmethod
    def _glcm_contrast(glcm: np.ndarray) -> float:
        """Contrast from GLCM"""
        i, j = np.ogrid[:glcm.shape[0], :glcm.shape[1]]
        contrast = np.sum((i - j)**2 * glcm)
        return float(contrast)
    
    @staticmethod
    def _glcm_dissimilarity(glcm: np.ndarray) -> float:
        """Dissimilarity from GLCM"""
        i, j = np.ogrid[:glcm.shape[0], :glcm.shape[1]]
        dissimilarity = np.sum(np.abs(i - j) * glcm)
        return float(dissimilarity)
    
    @staticmethod
    def _glcm_homogeneity(glcm: np.ndarray) -> float:
        """Homogeneity from GLCM"""
        i, j = np.ogrid[:glcm.shape[0], :glcm.shape[1]]
        homogeneity = np.sum(glcm / (1 + (i - j)**2))
        return float(homogeneity)
    
    @staticmethod
    def _glcm_energy(glcm: np.ndarray) -> float:
        """Energy from GLCM"""
        energy = np.sum(glcm ** 2)
        return float(energy)
    
    @staticmethod
    def _glcm_correlation(glcm: np.ndarray) -> float:
        """Correlation from GLCM"""
        i, j = np.ogrid[:glcm.shape[0], :glcm.shape[1]]
        
        mu_x = np.sum(i * glcm)
        mu_y = np.sum(j * glcm)
        
        sigma_x = np.sqrt(np.sum(((i - mu_x)**2) * glcm))
        sigma_y = np.sqrt(np.sum(((j - mu_y)**2) * glcm))
        
        if sigma_x > 0 and sigma_y > 0:
            correlation = np.sum(((i - mu_x) * (j - mu_y) * glcm) / 
                                (sigma_x * sigma_y))
        else:
            correlation = 0
        
        return float(correlation)


class HOGFeatures:
    """Histogram of Oriented Gradients (HOG)"""
    
    @staticmethod
    def compute_hog(image: np.ndarray, cell_size: int = 8,
                   n_bins: int = 9) -> np.ndarray:
        """
        Compute HOG features
        
        Parameters:
        -----------
        image : ndarray
            Input image (grayscale)
        cell_size : int
            Size of cells
        n_bins : int
            Number of orientation bins
        
        Returns:
        --------
        ndarray : HOG feature vector
        """
        # Compute gradients
        gx = np.gradient(image, axis=1)
        gy = np.gradient(image, axis=0)
        
        # Compute magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        # Divide into cells
        height, width = image.shape
        n_cells_h = height // cell_size
        n_cells_w = width // cell_size
        
        hog_features = []
        
        for cy in range(n_cells_h):
            for cx in range(n_cells_w):
                # Extract cell
                y1 = cy * cell_size
                y2 = (cy + 1) * cell_size
                x1 = cx * cell_size
                x2 = (cx + 1) * cell_size
                
                cell_mag = magnitude[y1:y2, x1:x2]
                cell_ori = orientation[y1:y2, x1:x2]
                
                # Compute histogram
                hist, _ = np.histogram(cell_ori.flatten(),
                                      bins=n_bins,
                                      range=(0, 180),
                                      weights=cell_mag.flatten())
                
                hog_features.extend(hist)
        
        return np.array(hog_features)


class ShapeFeatures:
    """Shape-based features"""
    
    @staticmethod
    def compute_moments(image: np.ndarray) -> dict:
        """
        Compute image moments
        
        Parameters:
        -----------
        image : ndarray
            Binary or grayscale image
        
        Returns:
        --------
        dict : Dictionary of moments
        """
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        
        m00 = np.sum(image)
        m10 = np.sum(x * image)
        m01 = np.sum(y * image)
        m20 = np.sum(x**2 * image)
        m02 = np.sum(y**2 * image)
        m11 = np.sum(x * y * image)
        
        # Centroids
        cx = m10 / m00 if m00 != 0 else 0
        cy = m01 / m00 if m00 != 0 else 0
        
        # Central moments
        mu20 = m20 - cx * m10
        mu02 = m02 - cy * m01
        mu11 = m11 - cx * m01
        
        return {
            'area': float(m00),
            'centroid': (float(cx), float(cy)),
            'mu20': float(mu20),
            'mu02': float(mu02),
            'mu11': float(mu11)
        }
    
    @staticmethod
    def compute_eccentricity(image: np.ndarray) -> float:
        """
        Compute eccentricity of shape
        
        Parameters:
        -----------
        image : ndarray
            Binary image
        
        Returns:
        --------
        float : Eccentricity
        """
        moments = ShapeFeatures.compute_moments(image)
        
        mu20 = moments['mu20']
        mu02 = moments['mu02']
        mu11 = moments['mu11']
        
        # Eigenvalues
        lambda1 = ((mu20 + mu02) + np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)) / 2
        lambda2 = ((mu20 + mu02) - np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)) / 2
        
        if lambda1 != 0:
            eccentricity = np.sqrt(1 - lambda2 / lambda1)
        else:
            eccentricity = 0
        
        return float(eccentricity)
    
    @staticmethod
    def compute_hu_moments(image: np.ndarray) -> np.ndarray:
        """
        Compute Hu moments (shape descriptors invariant to scale, rotation, translation)
        
        Parameters:
        -----------
        image : ndarray
            Binary image
        
        Returns:
        --------
        ndarray : 7 Hu moments
        """
        moments = ShapeFeatures.compute_moments(image)
        
        # Simplified Hu moments computation
        # (Full implementation requires normalized central moments)
        
        mu20 = moments['mu20']
        mu02 = moments['mu02']
        mu11 = moments['mu11']
        mu30 = 0  # Would need to compute 3rd order moments
        mu03 = 0
        mu12 = 0
        mu21 = 0
        
        hu = np.array([
            mu20 + mu02,
            (mu20 - mu02)**2 + 4 * mu11**2,
            (mu30 - 3*mu12)**2 + (3*mu21 - mu03)**2,
            (mu30 + mu12)**2 + (mu21 + mu03)**2,
            (mu30 - 3*mu12) * (mu30 + mu12) * ((mu30 + mu12)**2 - 3*(mu21 + mu03)**2) +
            (3*mu21 - mu03) * (mu21 + mu03) * (3*(mu30 + mu12)**2 - (mu21 + mu03)**2),
            (mu20 - mu02) * ((mu30 + mu12)**2 - (mu21 + mu03)**2) + 4*mu11*(mu30 + mu12)*(mu21 + mu03),
            (3*mu21 - mu03) * (mu30 + mu12) * ((mu30 + mu12)**2 - 3*(mu21 + mu03)**2) -
            (mu30 - 3*mu12) * (mu21 + mu03) * (3*(mu30 + mu12)**2 - (mu21 + mu03)**2)
        ])
        
        return hu
