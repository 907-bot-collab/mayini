import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


class ConvolutionFilters:
    """Convolution-based image filters"""
    
    @staticmethod
    def convolve_2d(image: np.ndarray, kernel: np.ndarray,
                   padding: str = 'same', mode: str = 'reflect') -> np.ndarray:
        """
        2D convolution
        
        Parameters:
        -----------
        image : ndarray
            Input image
        kernel : ndarray
            Convolution kernel
        padding : str
            'same' or 'valid'
        mode : str
            'reflect', 'constant', 'nearest'
        
        Returns:
        --------
        ndarray : Convolved image
        """
        return ndimage.convolve(image, kernel, mode=mode)
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: int = 5,
                     sigma: float = 1.0) -> np.ndarray:
        """
        Gaussian blur filter
        
        Parameters:
        -----------
        image : ndarray
            Input image
        kernel_size : int
            Size of Gaussian kernel
        sigma : float
            Standard deviation of Gaussian
        
        Returns:
        --------
        ndarray : Blurred image
        """
        # Create Gaussian kernel
        kernel = ConvolutionFilters._create_gaussian_kernel(kernel_size, sigma)
        
        # Apply convolution
        if len(image.shape) == 3:
            blurred = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                blurred[:, :, c] = ndimage.convolve(
                    image[:, :, c].astype(np.float32),
                    kernel,
                    mode='reflect'
                )
            return np.clip(blurred, 0, 255).astype(image.dtype)
        else:
            blurred = ndimage.convolve(
                image.astype(np.float32),
                kernel,
                mode='reflect'
            )
            return np.clip(blurred, 0, 255).astype(image.dtype)
    
    @staticmethod
    def _create_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
        """Create Gaussian kernel"""
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        y = x[:, np.newaxis]
        
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    @staticmethod
    def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Median filter
        
        Parameters:
        -----------
        image : ndarray
            Input image
        kernel_size : int
            Size of median filter kernel
        
        Returns:
        --------
        ndarray : Filtered image
        """
        return ndimage.median_filter(image, size=kernel_size)
    
    @staticmethod
    def box_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Box/mean filter
        
        Parameters:
        -----------
        image : ndarray
            Input image
        kernel_size : int
            Size of filter kernel
        
        Returns:
        --------
        ndarray : Filtered image
        """
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        if len(image.shape) == 3:
            filtered = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                filtered[:, :, c] = ndimage.convolve(
                    image[:, :, c].astype(np.float32),
                    kernel,
                    mode='reflect'
                )
            return np.clip(filtered, 0, 255).astype(image.dtype)
        else:
            filtered = ndimage.convolve(
                image.astype(np.float32),
                kernel,
                mode='reflect'
            )
            return np.clip(filtered, 0, 255).astype(image.dtype)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9,
                        sigma_color: float = 75,
                        sigma_space: float = 75) -> np.ndarray:
        """
        Bilateral filter (edge-preserving smoothing)
        
        Parameters:
        -----------
        image : ndarray
            Input image
        d : int
            Diameter of pixel neighborhood
        sigma_color : float
            Filter sigma in the color space
        sigma_space : float
            Filter sigma in the coordinate space
        
        Returns:
        --------
        ndarray : Filtered image
        """
        h = image.shape[0]
        w = image.shape[1]
        output = np.zeros_like(image, dtype=np.float32)
        
        r = d // 2
        
        for y in range(r, h - r):
            for x in range(r, w - r):
                # Get neighborhood
                neighborhood = image[y - r:y + r + 1, x - r:x + r + 1]
                center_pixel = image[y, x]
                
                # Spatial weights
                yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
                spatial_weight = np.exp(-(xx**2 + yy**2) / (2 * sigma_space**2))
                
                # Color weights
                color_dist = np.abs(neighborhood.astype(np.float32) - center_pixel)
                color_weight = np.exp(-color_dist / (2 * sigma_color**2))
                
                # Combined weights
                combined_weight = spatial_weight * color_weight
                combined_weight = combined_weight / np.sum(combined_weight)
                
                # Apply filter
                output[y, x] = np.sum(neighborhood.astype(np.float32) * combined_weight)
        
        return np.clip(output, 0, 255).astype(image.dtype)


class EdgeDetection:
    """Edge detection filters"""
    
    @staticmethod
    def sobel(image: np.ndarray, direction: str = 'both') -> np.ndarray:
        """
        Sobel edge detection
        
        Parameters:
        -----------
        image : ndarray
            Input image (grayscale)
        direction : str
            'x', 'y', or 'both'
        
        Returns:
        --------
        ndarray : Edge-detected image
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
        
        gx = ndimage.convolve(image.astype(np.float32), sobel_x, mode='reflect')
        gy = ndimage.convolve(image.astype(np.float32), sobel_y, mode='reflect')
        
        if direction == 'x':
            edges = np.abs(gx)
        elif direction == 'y':
            edges = np.abs(gy)
        else:  # both
            edges = np.sqrt(gx**2 + gy**2)
        
        return np.clip(edges, 0, 255).astype(image.dtype)
    
    @staticmethod
    def scharr(image: np.ndarray) -> np.ndarray:
        """
        Scharr edge detection (improved Sobel)
        """
        scharr_x = np.array([[-3, 0, 3],
                            [-10, 0, 10],
                            [-3, 0, 3]], dtype=np.float32)
        
        scharr_y = np.array([[-3, -10, -3],
                            [0, 0, 0],
                            [3, 10, 3]], dtype=np.float32)
        
        gx = ndimage.convolve(image.astype(np.float32), scharr_x, mode='reflect')
        gy = ndimage.convolve(image.astype(np.float32), scharr_y, mode='reflect')
        
        edges = np.sqrt(gx**2 + gy**2)
        
        return np.clip(edges, 0, 255).astype(image.dtype)
    
    @staticmethod
    def laplacian(image: np.ndarray) -> np.ndarray:
        """
        Laplacian edge detection
        """
        laplacian_kernel = np.array([[0, -1, 0],
                                    [-1, 4, -1],
                                    [0, -1, 0]], dtype=np.float32)
        
        edges = ndimage.convolve(image.astype(np.float32),
                                laplacian_kernel,
                                mode='reflect')
        
        return np.clip(np.abs(edges), 0, 255).astype(image.dtype)
    
    @staticmethod
    def prewitt(image: np.ndarray) -> np.ndarray:
        """
        Prewitt edge detection
        """
        prewitt_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float32)
        
        prewitt_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]], dtype=np.float32)
        
        gx = ndimage.convolve(image.astype(np.float32), prewitt_x, mode='reflect')
        gy = ndimage.convolve(image.astype(np.float32), prewitt_y, mode='reflect')
        
        edges = np.sqrt(gx**2 + gy**2)
        
        return np.clip(edges, 0, 255).astype(image.dtype)
    
    @staticmethod
    def canny(image: np.ndarray, low_threshold: float = 0.1,
             high_threshold: float = 0.3) -> np.ndarray:
        """
        Simplified Canny edge detection
        (Full Canny requires edge thinning and hysteresis)
        
        Parameters:
        -----------
        image : ndarray
            Input grayscale image
        low_threshold : float
            Low threshold (0-1)
        high_threshold : float
            High threshold (0-1)
        
        Returns:
        --------
        ndarray : Binary edge map
        """
        # Step 1: Gaussian blur
        blurred = ConvolutionFilters.gaussian_blur(image, kernel_size=5, sigma=1.4)
        
        # Step 2: Sobel edge detection
        edges = EdgeDetection.sobel(blurred.astype(np.uint8), direction='both')
        
        # Step 3: Normalize
        normalized = edges.astype(np.float32) / np.max(edges + 1e-10)
        
        # Step 4: Threshold
        low = low_threshold
        high = high_threshold
        
        edge_map = np.zeros_like(normalized)
        edge_map[normalized >= high] = 255
        edge_map[(normalized >= low) & (normalized < high)] = 128
        
        return edge_map.astype(np.uint8)


class MorphologicalOperations:
    """Morphological operations (dilation, erosion, etc.)"""
    
    @staticmethod
    def erode(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Erosion operation
        
        Parameters:
        -----------
        image : ndarray
            Binary image
        kernel_size : int
            Size of structuring element
        
        Returns:
        --------
        ndarray : Eroded image
        """
        return ndimage.binary_erosion(image, iterations=1).astype(image.dtype)
    
    @staticmethod
    def dilate(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Dilation operation
        
        Parameters:
        -----------
        image : ndarray
            Binary image
        kernel_size : int
            Size of structuring element
        
        Returns:
        --------
        ndarray : Dilated image
        """
        return ndimage.binary_dilation(image, iterations=1).astype(image.dtype)
    
    @staticmethod
    def opening(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Opening = erosion followed by dilation"""
        eroded = ndimage.binary_erosion(image, iterations=1)
        opened = ndimage.binary_dilation(eroded, iterations=1)
        return opened.astype(image.dtype)
    
    @staticmethod
    def closing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Closing = dilation followed by erosion"""
        dilated = ndimage.binary_dilation(image, iterations=1)
        closed = ndimage.binary_erosion(dilated, iterations=1)
        return closed.astype(image.dtype)


class ColorConversion:
    """Color space conversions"""
    
    @staticmethod
    def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert RGB to grayscale using standard weights
        
        Parameters:
        -----------
        image : ndarray
            RGB image
        
        Returns:
        --------
        ndarray : Grayscale image
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        
        # Standard weights
        weights = np.array([0.299, 0.587, 0.114])
        grayscale = np.dot(image, weights)
        
        return np.clip(grayscale, 0, 255).astype(image.dtype)
    
    @staticmethod
    def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert grayscale to RGB"""
        if len(image.shape) == 2:
            return np.stack([image, image, image], axis=2)
        return image
    
    @staticmethod
    def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV"""
        image_float = image.astype(np.float32) / 255.0
        
        r = image_float[:, :, 0]
        g = image_float[:, :, 1]
        b = image_float[:, :, 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = np.zeros_like(v)
        mask = v != 0
        s[mask] = delta[mask] / v[mask]
        
        # Hue
        h = np.zeros_like(v)
        
        mask_r = (max_val == r) & (delta != 0)
        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        
        mask_g = (max_val == g) & (delta != 0)
        h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
        
        mask_b = (max_val == b) & (delta != 0)
        h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
        
        h = (h + 360) % 360
        
        hsv = np.stack([h, s * 255, v * 255], axis=2)
        
        return np.clip(hsv, 0, 255).astype(np.uint8)
    
    @staticmethod
    def hsv_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB"""
        image_float = image.astype(np.float32)
        
        h = image_float[:, :, 0]
        s = image_float[:, :, 1] / 255.0
        v = image_float[:, :, 2] / 255.0
        
        c = v * s
        hp = h / 60.0
        x = c * (1 - np.abs(hp % 2 - 1))
        
        rgb = np.zeros_like(image_float[:, :, :3])
        
        mask1 = (hp >= 0) & (hp < 1)
        rgb[mask1, 0] = c[mask1]
        rgb[mask1, 1] = x[mask1]
        
        mask2 = (hp >= 1) & (hp < 2)
        rgb[mask2, 0] = x[mask2]
        rgb[mask2, 1] = c[mask2]
        
        # ... continue for other ranges
        
        m = v - c
        rgb = (rgb + m[:, :, np.newaxis]) * 255
        
        return np.clip(rgb, 0, 255).astype(np.uint8)
