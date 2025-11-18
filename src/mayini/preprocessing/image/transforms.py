import numpy as np
from typing import Tuple, Optional
import math


class ImageTransforms:
    """
    Image transformation operations
    Resize, rotate, flip, crop, perspective transforms
    """
    
    @staticmethod
    def resize(image: np.ndarray, size: Tuple[int, int],
              method: str = 'bilinear') -> np.ndarray:
        """
        Resize image to specified size
        
        Parameters:
        -----------
        image : ndarray
            Input image (H, W) or (H, W, C)
        size : tuple
            Target size (height, width)
        method : str
            'nearest' or 'bilinear' interpolation
        
        Returns:
        --------
        ndarray : Resized image
        """
        target_height, target_width = size
        source_height, source_width = image.shape[:2]
        
        if len(image.shape) == 3:
            channels = image.shape[2]
            resized = np.zeros((target_height, target_width, channels), dtype=image.dtype)
            
            for c in range(channels):
                resized[:, :, c] = ImageTransforms._resize_2d(
                    image[:, :, c],
                    (target_height, target_width),
                    method
                )
            
            return resized
        else:
            return ImageTransforms._resize_2d(image, size, method)
    
    @staticmethod
    def _resize_2d(image: np.ndarray, size: Tuple[int, int],
                  method: str = 'bilinear') -> np.ndarray:
        """Resize 2D image (grayscale)"""
        target_height, target_width = size
        source_height, source_width = image.shape
        
        if method == 'nearest':
            return ImageTransforms._resize_nearest(image, size)
        elif method == 'bilinear':
            return ImageTransforms._resize_bilinear(image, size)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def _resize_nearest(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Nearest neighbor interpolation"""
        target_height, target_width = size
        source_height, source_width = image.shape
        
        # Compute scaling factors
        scale_y = source_height / target_height
        scale_x = source_width / target_width
        
        # Create output image
        output = np.zeros((target_height, target_width), dtype=image.dtype)
        
        for y in range(target_height):
            for x in range(target_width):
                src_y = int(y * scale_y)
                src_x = int(x * scale_x)
                
                # Clamp to valid range
                src_y = min(src_y, source_height - 1)
                src_x = min(src_x, source_width - 1)
                
                output[y, x] = image[src_y, src_x]
        
        return output
    
    @staticmethod
    def _resize_bilinear(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Bilinear interpolation
        Higher quality than nearest neighbor
        """
        target_height, target_width = size
        source_height, source_width = image.shape
        
        # Compute scaling factors
        scale_y = (source_height - 1) / (target_height - 1) if target_height > 1 else 1
        scale_x = (source_width - 1) / (target_width - 1) if target_width > 1 else 1
        
        output = np.zeros((target_height, target_width), dtype=np.float32)
        
        for y in range(target_height):
            for x in range(target_width):
                # Map to source coordinates
                src_y = y * scale_y
                src_x = x * scale_x
                
                # Get integer and fractional parts
                y1 = int(src_y)
                x1 = int(src_x)
                y2 = min(y1 + 1, source_height - 1)
                x2 = min(x1 + 1, source_width - 1)
                
                fy = src_y - y1
                fx = src_x - x1
                
                # Bilinear interpolation
                v1 = image[y1, x1] * (1 - fx) * (1 - fy)
                v2 = image[y1, x2] * fx * (1 - fy)
                v3 = image[y2, x1] * (1 - fx) * fy
                v4 = image[y2, x2] * fx * fy
                
                output[y, x] = v1 + v2 + v3 + v4
        
        return np.clip(output, 0, 255).astype(image.dtype)
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float, 
              keep_size: bool = True) -> np.ndarray:
        """
        Rotate image by angle (degrees)
        
        Parameters:
        -----------
        image : ndarray
            Input image
        angle : float
            Rotation angle in degrees (counter-clockwise)
        keep_size : bool
            Keep original image size (crop) or expand to fit
        
        Returns:
        --------
        ndarray : Rotated image
        """
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        height, width = image.shape[:2]
        
        if len(image.shape) == 3:
            channels = image.shape[2]
            rotated = np.zeros_like(image)
            
            for c in range(channels):
                rotated[:, :, c] = ImageTransforms._rotate_2d(
                    image[:, :, c],
                    angle_rad,
                    keep_size
                )
            
            return rotated
        else:
            return ImageTransforms._rotate_2d(image, angle_rad, keep_size)
    
    @staticmethod
    def _rotate_2d(image: np.ndarray, angle_rad: float, 
                  keep_size: bool = True) -> np.ndarray:
        """Rotate 2D image"""
        height, width = image.shape
        
        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Calculate new dimensions
        if not keep_size:
            # Calculate bounding box of rotated image
            corners = np.array([
                [0, 0], [width, 0],
                [0, height], [width, height]
            ]) - np.array([width/2, height/2])
            
            rotated_corners = np.dot(corners,
                np.array([[cos_a, sin_a], [-sin_a, cos_a]]))
            
            new_width = int(np.ceil(np.max(rotated_corners[:, 0]) - 
                                   np.min(rotated_corners[:, 0])))
            new_height = int(np.ceil(np.max(rotated_corners[:, 1]) - 
                                    np.min(rotated_corners[:, 1])))
        else:
            new_height = height
            new_width = width
        
        # Create output image
        output = np.zeros((new_height, new_width), dtype=image.dtype)
        
        # Inverse rotation matrix
        inv_cos_a = cos_a
        inv_sin_a = -sin_a
        
        # Center points
        cx_orig = width / 2
        cy_orig = height / 2
        cx_new = new_width / 2
        cy_new = new_height / 2
        
        # For each output pixel, find corresponding input pixel
        for y in range(new_height):
            for x in range(new_width):
                # Translate to center
                x_c = x - cx_new
                y_c = y - cy_new
                
                # Inverse rotation
                src_x = x_c * inv_cos_a - y_c * inv_sin_a + cx_orig
                src_y = x_c * inv_sin_a + y_c * inv_cos_a + cy_orig
                
                # Bilinear interpolation
                if 0 <= src_x < width - 1 and 0 <= src_y < height - 1:
                    x1 = int(src_x)
                    y1 = int(src_y)
                    x2 = min(x1 + 1, width - 1)
                    y2 = min(y1 + 1, height - 1)
                    
                    fx = src_x - x1
                    fy = src_y - y1
                    
                    v1 = image[y1, x1] * (1 - fx) * (1 - fy)
                    v2 = image[y1, x2] * fx * (1 - fy)
                    v3 = image[y2, x1] * (1 - fx) * fy
                    v4 = image[y2, x2] * fx * fy
                    
                    output[y, x] = np.clip(v1 + v2 + v3 + v4, 0, 255)
        
        return output.astype(image.dtype)
    
    @staticmethod
    def flip(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
        """
        Flip image
        
        Parameters:
        -----------
        image : ndarray
            Input image
        direction : str
            'horizontal' or 'vertical'
        
        Returns:
        --------
        ndarray : Flipped image
        """
        if direction == 'horizontal':
            return np.fliplr(image)
        elif direction == 'vertical':
            return np.flipud(image)
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")
    
    @staticmethod
    def crop(image: np.ndarray, top: int, left: int,
            height: int, width: int) -> np.ndarray:
        """
        Crop image
        
        Parameters:
        -----------
        image : ndarray
            Input image
        top : int
            Top coordinate
        left : int
            Left coordinate
        height : int
            Crop height
        width : int
            Crop width
        
        Returns:
        --------
        ndarray : Cropped image
        """
        img_height, img_width = image.shape[:2]
        
        # Validate bounds
        bottom = min(top + height, img_height)
        right = min(left + width, img_width)
        
        return image[top:bottom, left:right]
    
    @staticmethod
    def pad(image: np.ndarray, top: int = 0, bottom: int = 0,
           left: int = 0, right: int = 0, fill_value: int = 0) -> np.ndarray:
        """
        Pad image with border
        
        Parameters:
        -----------
        image : ndarray
            Input image
        top, bottom, left, right : int
            Padding amounts
        fill_value : int
            Fill value for padding
        
        Returns:
        --------
        ndarray : Padded image
        """
        if len(image.shape) == 3:
            pad_width = ((top, bottom), (left, right), (0, 0))
        else:
            pad_width = ((top, bottom), (left, right))
        
        return np.pad(image, pad_width, mode='constant', 
                     constant_values=fill_value)
    
    @staticmethod
    def zoom(image: np.ndarray, zoom_factor: float,
            center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Zoom image
        
        Parameters:
        -----------
        image : ndarray
            Input image
        zoom_factor : float
            Zoom factor (>1 to zoom in, <1 to zoom out)
        center : tuple, optional
            Zoom center (y, x). If None, use image center
        
        Returns:
        --------
        ndarray : Zoomed image
        """
        height, width = image.shape[:2]
        
        if center is None:
            center = (height // 2, width // 2)
        
        cy, cx = center
        
        # Calculate crop region
        crop_height = int(height / zoom_factor)
        crop_width = int(width / zoom_factor)
        
        top = max(0, cy - crop_height // 2)
        left = max(0, cx - crop_width // 2)
        
        bottom = min(height, top + crop_height)
        right = min(width, left + crop_width)
        
        # Crop
        cropped = image[top:bottom, left:right]
        
        # Resize back to original size
        resized = ImageTransforms.resize(cropped, (height, width), method='bilinear')
        
        return resized
    
    @staticmethod
    def perspective_transform(image: np.ndarray,
                             src_points: np.ndarray,
                             dst_points: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform
        
        Parameters:
        -----------
        image : ndarray
            Input image
        src_points : ndarray
            Source points (4, 2)
        dst_points : ndarray
            Destination points (4, 2)
        
        Returns:
        --------
        ndarray : Transformed image
        """
        # This is a simplified implementation
        # For production, use more robust algorithms
        
        height, width = image.shape[:2]
        output = np.zeros_like(image)
        
        # Apply perspective transform (bilinear approximation)
        for y in range(height):
            for x in range(width):
                # Map destination to source using bilinear mapping
                u = x / width
                v = y / height
                
                # Simple affine mapping
                src_x = u * width
                src_y = v * height
                
                if 0 <= src_x < width - 1 and 0 <= src_y < height - 1:
                    x1, y1 = int(src_x), int(src_y)
                    x2, y2 = x1 + 1, y1 + 1
                    
                    fx, fy = src_x - x1, src_y - y1
                    
                    output[y, x] = (image[y1, x1] * (1-fx) * (1-fy) +
                                   image[y1, x2] * fx * (1-fy) +
                                   image[y2, x1] * (1-fx) * fy +
                                   image[y2, x2] * fx * fy)
        
        return np.clip(output, 0, 255).astype(image.dtype)


class Normalize:
    """Image normalization"""
    
    @staticmethod
    def standardize(image: np.ndarray, mean: Optional[float] = None,
                   std: Optional[float] = None) -> np.ndarray:
        """
        Standardize image (zero mean, unit variance)
        
        Parameters:
        -----------
        image : ndarray
            Input image
        mean : float, optional
            Mean value. If None, compute from image
        std : float, optional
            Standard deviation. If None, compute from image
        
        Returns:
        --------
        ndarray : Standardized image
        """
        image = image.astype(np.float32)
        
        if mean is None:
            mean = np.mean(image)
        if std is None:
            std = np.std(image)
        
        return (image - mean) / (std + 1e-10)
    
    @staticmethod
    def normalize_255(image: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]"""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def normalize_range(image: np.ndarray, min_val: float = 0,
                       max_val: float = 1) -> np.ndarray:
        """Normalize to [min_val, max_val]"""
        img_min = np.min(image)
        img_max = np.max(image)
        
        normalized = (image - img_min) / (img_max - img_min + 1e-10)
        return min_val + normalized * (max_val - min_val)
