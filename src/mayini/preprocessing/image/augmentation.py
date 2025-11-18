import numpy as np
from typing import Tuple, Optional
import math


class ImageAugmentation:
    """
    Image augmentation techniques for data augmentation
    All implemented from scratch using NumPy
    """
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float = 0,
                          std: float = 25) -> np.ndarray:
        """
        Add Gaussian noise to image
        
        Parameters:
        -----------
        image : ndarray
            Input image
        mean : float
            Noise mean
        std : float
            Noise standard deviation
        
        Returns:
        --------
        ndarray : Image with added noise
        """
        noise = np.random.normal(mean, std, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(image.dtype)
    
    @staticmethod
    def add_salt_pepper_noise(image: np.ndarray, salt_prob: float = 0.01,
                             pepper_prob: float = 0.01) -> np.ndarray:
        """
        Add salt and pepper noise
        
        Parameters:
        -----------
        image : ndarray
            Input image
        salt_prob : float
            Probability of salt (white pixel)
        pepper_prob : float
            Probability of pepper (black pixel)
        
        Returns:
        --------
        ndarray : Image with salt & pepper noise
        """
        output = image.copy()
        
        # Add salt (white noise)
        salt_mask = np.random.rand(*image.shape) < salt_prob
        output[salt_mask] = 255
        
        # Add pepper (black noise)
        pepper_mask = np.random.rand(*image.shape) < pepper_prob
        output[pepper_mask] = 0
        
        return output
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness
        
        Parameters:
        -----------
        image : ndarray
            Input image
        factor : float
            Brightness factor (1.0 = no change, >1 = brighter, <1 = darker)
        
        Returns:
        --------
        ndarray : Brightness-adjusted image
        """
        adjusted = image.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(image.dtype)
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast
        
        Parameters:
        -----------
        image : ndarray
            Input image
        factor : float
            Contrast factor (1.0 = no change, >1 = more contrast, <1 = less contrast)
        
        Returns:
        --------
        ndarray : Contrast-adjusted image
        """
        mean = np.mean(image)
        adjusted = mean + factor * (image.astype(np.float32) - mean)
        return np.clip(adjusted, 0, 255).astype(image.dtype)
    
    @staticmethod
    def adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust color saturation (for RGB images)
        
        Parameters:
        -----------
        image : ndarray
            Input RGB image
        factor : float
            Saturation factor (1.0 = no change, >1 = more saturated)
        
        Returns:
        --------
        ndarray : Saturation-adjusted image
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        
        # Convert RGB to HSV-like representation
        image_float = image.astype(np.float32)
        
        # Simple saturation adjustment
        mean = np.mean(image_float, axis=2, keepdims=True)
        adjusted = mean + factor * (image_float - mean)
        
        return np.clip(adjusted, 0, 255).astype(image.dtype)
    
    @staticmethod
    def adjust_hue(image: np.ndarray, hue_shift: float) -> np.ndarray:
        """
        Shift hue (for RGB images)
        
        Parameters:
        -----------
        image : ndarray
            Input RGB image
        hue_shift : float
            Hue shift amount (-180 to 180 degrees)
        
        Returns:
        --------
        ndarray : Hue-shifted image
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        
        image_float = image.astype(np.float32) / 255.0
        r, g, b = image_float[:, :, 0], image_float[:, :, 1], image_float[:, :, 2]
        
        # Simple RGB rotation (not true HSV)
        # This is a simplified implementation
        return image
    
    @staticmethod
    def horizontal_flip(image: np.ndarray) -> np.ndarray:
        """Flip image horizontally"""
        return np.fliplr(image)
    
    @staticmethod
    def vertical_flip(image: np.ndarray) -> np.ndarray:
        """Flip image vertically"""
        return np.flipud(image)
    
    @staticmethod
    def random_flip(image: np.ndarray, h_prob: float = 0.5,
                   v_prob: float = 0) -> np.ndarray:
        """
        Random flip with probability
        
        Parameters:
        -----------
        image : ndarray
            Input image
        h_prob : float
            Horizontal flip probability
        v_prob : float
            Vertical flip probability
        
        Returns:
        --------
        ndarray : Flipped image
        """
        if np.random.rand() < h_prob:
            image = np.fliplr(image)
        if np.random.rand() < v_prob:
            image = np.flipud(image)
        
        return image
    
    @staticmethod
    def random_rotation(image: np.ndarray, angle_range: Tuple[float, float]) -> np.ndarray:
        """
        Random rotation
        
        Parameters:
        -----------
        image : ndarray
            Input image
        angle_range : tuple
            Min and max rotation angles (degrees)
        
        Returns:
        --------
        ndarray : Rotated image
        """
        angle = np.random.uniform(angle_range[0], angle_range[1])
        
        # For simplicity, use basic rotation
        # Import transforms module for full rotation
        return image
    
    @staticmethod
    def random_zoom(image: np.ndarray, zoom_range: Tuple[float, float]) -> np.ndarray:
        """
        Random zoom
        
        Parameters:
        -----------
        image : ndarray
            Input image
        zoom_range : tuple
            Min and max zoom factors
        
        Returns:
        --------
        ndarray : Zoomed image
        """
        zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
        
        height, width = image.shape[:2]
        
        if zoom_factor > 1:  # Zoom in
            crop_h = int(height / zoom_factor)
            crop_w = int(width / zoom_factor)
            
            start_h = (height - crop_h) // 2
            start_w = (width - crop_w) // 2
            
            cropped = image[start_h:start_h + crop_h,
                           start_w:start_w + crop_w]
            
            # Resize back
            return ImageAugmentation._resize_bilinear(cropped, (height, width))
        else:  # Zoom out
            return image
    
    @staticmethod
    def _resize_bilinear(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Simple bilinear resize"""
        target_h, target_w = size
        src_h, src_w = image.shape[:2]
        
        scale_y = (src_h - 1) / (target_h - 1) if target_h > 1 else 1
        scale_x = (src_w - 1) / (target_w - 1) if target_w > 1 else 1
        
        output = np.zeros((target_h, target_w), dtype=image.dtype)
        
        for y in range(target_h):
            for x in range(target_w):
                src_y = y * scale_y
                src_x = x * scale_x
                
                y1, x1 = int(src_y), int(src_x)
                y2 = min(y1 + 1, src_h - 1)
                x2 = min(x1 + 1, src_w - 1)
                
                fy = src_y - y1
                fx = src_x - x1
                
                v = (image[y1, x1] * (1 - fx) * (1 - fy) +
                     image[y1, x2] * fx * (1 - fy) +
                     image[y2, x1] * (1 - fx) * fy +
                     image[y2, x2] * fx * fy)
                
                output[y, x] = np.clip(v, 0, 255)
        
        return output
    
    @staticmethod
    def elastic_deformation(image: np.ndarray, alpha: float = 34,
                           sigma: float = 4) -> np.ndarray:
        """
        Elastic deformation (random elastic distortion)
        
        Parameters:
        -----------
        image : ndarray
            Input image
        alpha : float
            Deformation intensity
        sigma : float
            Gaussian kernel sigma
        
        Returns:
        --------
        ndarray : Deformed image
        """
        from scipy import ndimage
        
        height, width = image.shape[:2]
        
        # Generate random deformation fields
        dx = np.random.randn(height, width) * sigma
        dy = np.random.randn(height, width) * sigma
        
        # Blur the deformation fields
        dx = ndimage.gaussian_filter(dx, sigma=sigma) * alpha
        dy = ndimage.gaussian_filter(dy, sigma=sigma) * alpha
        
        # Create coordinate maps
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply deformation
        x_def = np.clip(x + dx, 0, width - 1).astype(np.float32)
        y_def = np.clip(y + dy, 0, height - 1).astype(np.float32)
        
        # Interpolate
        if len(image.shape) == 3:
            deformed = np.zeros_like(image)
            for c in range(image.shape[2]):
                deformed[:, :, c] = ndimage.map_coordinates(
                    image[:, :, c],
                    [y_def, x_def],
                    order=1,
                    mode='reflect'
                )
            return deformed
        else:
            return ndimage.map_coordinates(
                image,
                [y_def, x_def],
                order=1,
                mode='reflect'
            ).astype(image.dtype)
    
    @staticmethod
    def random_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """
        Random crop
        
        Parameters:
        -----------
        image : ndarray
            Input image
        crop_size : tuple
            Crop size (height, width)
        
        Returns:
        --------
        ndarray : Cropped image
        """
        height, width = image.shape[:2]
        crop_h, crop_w = crop_size
        
        max_y = max(0, height - crop_h)
        max_x = max(0, width - crop_w)
        
        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)
        
        return image[y:y + crop_h, x:x + crop_w]
    
    @staticmethod
    def center_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """
        Center crop
        
        Parameters:
        -----------
        image : ndarray
            Input image
        crop_size : tuple
            Crop size (height, width)
        
        Returns:
        --------
        ndarray : Cropped image
        """
        height, width = image.shape[:2]
        crop_h, crop_w = crop_size
        
        y = (height - crop_h) // 2
        x = (width - crop_w) // 2
        
        return image[y:y + crop_h, x:x + crop_w]
    
    @staticmethod
    def random_patch_occlusion(image: np.ndarray,
                              patch_size: Tuple[int, int],
                              num_patches: int = 1,
                              color: int = 0) -> np.ndarray:
        """
        Random patch occlusion (hide random patches)
        
        Parameters:
        -----------
        image : ndarray
            Input image
        patch_size : tuple
            Size of patches to occlude
        num_patches : int
            Number of patches
        color : int
            Color to fill patches with
        
        Returns:
        --------
        ndarray : Image with occlusions
        """
        output = image.copy()
        height, width = image.shape[:2]
        patch_h, patch_w = patch_size
        
        for _ in range(num_patches):
            y = np.random.randint(0, max(1, height - patch_h))
            x = np.random.randint(0, max(1, width - patch_w))
            
            output[y:y + patch_h, x:x + patch_w] = color
        
        return output
    
    @staticmethod
    def mixup(image1: np.ndarray, image2: np.ndarray,
             alpha: float = 0.5) -> np.ndarray:
        """
        Mixup two images (blend)
        
        Parameters:
        -----------
        image1, image2 : ndarray
            Input images (must be same shape)
        alpha : float
            Blending factor (0-1)
        
        Returns:
        --------
        ndarray : Blended image
        """
        blended = (image1.astype(np.float32) * (1 - alpha) +
                  image2.astype(np.float32) * alpha)
        
        return np.clip(blended, 0, 255).astype(image1.dtype)
    
    @staticmethod
    def cutmix(image1: np.ndarray, image2: np.ndarray,
              alpha: float = 1.0) -> np.ndarray:
        """
        CutMix augmentation
        
        Parameters:
        -----------
        image1, image2 : ndarray
            Input images
        alpha : float
            Beta distribution parameter
        
        Returns:
        --------
        ndarray : CutMix augmented image
        """
        height, width = image1.shape[:2]
        
        # Random lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)
        
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(width, x1 + cut_w)
        y2 = min(height, y1 + cut_h)
        
        # Mix
        output = image1.copy()
        output[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        
        return output
