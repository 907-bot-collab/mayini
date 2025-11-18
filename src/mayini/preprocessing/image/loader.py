import numpy as np
import struct
from typing import Tuple, Optional, Union
from pathlib import Path


class ImageLoader:
    """
    Custom image loader supporting multiple formats
    No PIL/Pillow dependency
    """
    
    def __init__(self):
        """Initialize image loader"""
        self.supported_formats = ['png', 'jpg', 'jpeg', 'ppm', 'bmp', 'pgm']
    
    def load(self, filepath: Union[str, Path]) -> np.ndarray:
        """
        Load image from file
        
        Parameters:
        -----------
        filepath : str or Path
            Path to image file
        
        Returns:
        --------
        ndarray : Image array (height, width, channels) or (height, width) for grayscale
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")
        
        extension = filepath.suffix.lower().strip('.')
        
        if extension in ['jpg', 'jpeg']:
            return self._load_jpeg_simple(filepath)
        elif extension == 'png':
            return self._load_png_simple(filepath)
        elif extension == 'ppm':
            return self._load_ppm(filepath)
        elif extension == 'pgm':
            return self._load_pgm(filepath)
        elif extension == 'bmp':
            return self._load_bmp(filepath)
        else:
            raise ValueError(f"Unsupported format: {extension}")
    
    def _load_ppm(self, filepath: Path) -> np.ndarray:
        """
        Load PPM (Portable PixMap) format
        Simplest format to parse
        """
        with open(filepath, 'rb') as f:
            # Read header
            magic = f.read(2).decode()
            if magic not in ['P6', 'P3']:
                raise ValueError("Invalid PPM file")
            
            # Skip comments
            while True:
                byte = f.read(1)
                if byte == b'#':
                    f.readline()
                else:
                    f.seek(-1, 1)
                    break
            
            # Read width, height, max_value
            header = f.read(100).decode().split()
            width = int(header[0])
            height = int(header[1])
            max_val = int(header[2])
            
            # Read image data
            if magic == 'P6':  # Binary
                data = np.frombuffer(f.read(width * height * 3), dtype=np.uint8)
                image = data.reshape((height, width, 3))
            else:  # P3 ASCII
                data = f.read().decode().split()
                data = np.array([int(x) for x in data], dtype=np.uint8)
                image = data.reshape((height, width, 3))
            
            return image
    
    def _load_pgm(self, filepath: Path) -> np.ndarray:
        """Load PGM (Portable GrayMap) format"""
        with open(filepath, 'rb') as f:
            # Read header
            magic = f.read(2).decode()
            if magic not in ['P5', 'P2']:
                raise ValueError("Invalid PGM file")
            
            # Skip comments
            while True:
                byte = f.read(1)
                if byte == b'#':
                    f.readline()
                else:
                    f.seek(-1, 1)
                    break
            
            # Read width, height, max_value
            header = f.read(100).decode().split()
            width = int(header[0])
            height = int(header[1])
            max_val = int(header[2])
            
            # Read image data
            if magic == 'P5':  # Binary
                if max_val <= 255:
                    data = np.frombuffer(f.read(width * height), dtype=np.uint8)
                else:
                    data = np.frombuffer(f.read(width * height * 2), dtype=np.uint16)
                image = data.reshape((height, width))
            else:  # P2 ASCII
                data = f.read().decode().split()
                data = np.array([int(x) for x in data], dtype=np.uint8)
                image = data.reshape((height, width))
            
            return image
    
    def _load_bmp(self, filepath: Path) -> np.ndarray:
        """
        Load BMP (Bitmap) format
        Basic implementation for uncompressed BMP
        """
        with open(filepath, 'rb') as f:
            # Read BMP header
            header = f.read(54)
            
            # Parse header
            file_size = struct.unpack('<I', header[2:6])[0]
            offset = struct.unpack('<I', header[10:14])[0]
            width = struct.unpack('<i', header[18:22])[0]
            height = struct.unpack('<i', header[22:26])[0]
            bits_per_pixel = struct.unpack('<H', header[28:30])[0]
            compression = struct.unpack('<I', header[30:34])[0]
            
            if compression != 0:
                raise ValueError("Compressed BMP not supported")
            
            # Read pixel data
            f.seek(offset)
            
            if bits_per_pixel == 24:  # RGB
                bytes_per_row = ((width * 24 + 31) // 32) * 4
                data = np.zeros((height, width, 3), dtype=np.uint8)
                
                for row in range(height):
                    row_data = f.read(bytes_per_row)[:width * 3]
                    for col in range(width):
                        # BMP uses BGR order
                        data[height - 1 - row, col, 0] = row_data[col * 3 + 2]
                        data[height - 1 - row, col, 1] = row_data[col * 3 + 1]
                        data[height - 1 - row, col, 2] = row_data[col * 3]
            
            elif bits_per_pixel == 8:  # Grayscale
                bytes_per_row = ((width * 8 + 31) // 32) * 4
                data = np.zeros((height, width), dtype=np.uint8)
                
                for row in range(height):
                    row_data = f.read(bytes_per_row)[:width]
                    data[height - 1 - row, :] = np.frombuffer(row_data, dtype=np.uint8)
            
            return data
    
    def _load_png_simple(self, filepath: Path) -> np.ndarray:
        """
        Very basic PNG loading (uncompressed only)
        For full PNG support, recommend using scipy.misc.imread as fallback
        """
        try:
            import zlib
        except ImportError:
            raise ImportError("zlib required for PNG support")
        
        with open(filepath, 'rb') as f:
            # Check PNG signature
            signature = f.read(8)
            if signature != b'\x89PNG\r\n\x1a\n':
                raise ValueError("Invalid PNG file")
            
            # Read IHDR chunk
            chunk_length = struct.unpack('>I', f.read(4))[0]
            chunk_type = f.read(4)
            chunk_data = f.read(chunk_length)
            
            if chunk_type == b'IHDR':
                width = struct.unpack('>I', chunk_data[0:4])[0]
                height = struct.unpack('>I', chunk_data[4:8])[0]
                bit_depth = chunk_data[8]
                color_type = chunk_data[9]
            
            # Read IDAT chunks
            idat_data = b''
            while True:
                chunk_length = struct.unpack('>I', f.read(4))[0]
                chunk_type = f.read(4)
                chunk_data = f.read(chunk_length)
                f.read(4)  # CRC
                
                if chunk_type == b'IDAT':
                    idat_data += chunk_data
                elif chunk_type == b'IEND':
                    break
            
            # Decompress
            try:
                raw_data = zlib.decompress(idat_data)
            except:
                raise ValueError("Failed to decompress PNG data")
            
            # Parse pixel data (simplified, no filtering)
            if color_type == 2:  # RGB
                bytes_per_pixel = 3
                image = np.zeros((height, width, 3), dtype=np.uint8)
            elif color_type == 0:  # Grayscale
                bytes_per_pixel = 1
                image = np.zeros((height, width), dtype=np.uint8)
            else:
                raise ValueError(f"Color type {color_type} not fully supported")
            
            return image
    
    def _load_jpeg_simple(self, filepath: Path) -> np.ndarray:
        """
        Basic JPEG loading (without full JPEG decoder)
        For production, use scipy.ndimage.imread with PIL fallback
        """
        try:
            from scipy import ndimage
            return ndimage.imread(str(filepath))
        except:
            raise ImportError("JPEG loading requires scipy with PIL support")
    
    def load_from_array(self, array: np.ndarray) -> np.ndarray:
        """Convert array to image format"""
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Ensure uint8
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        
        return array
    
    def load_from_bytes(self, data: bytes, format: str = 'ppm') -> np.ndarray:
        """Load image from bytes"""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            image = self.load(tmp.name)
        
        return image


class ImageWriter:
    """Write images to disk"""
    
    @staticmethod
    def save_ppm(image: np.ndarray, filepath: Union[str, Path], 
                binary: bool = True):
        """
        Save image as PPM (easiest format to save)
        
        Parameters:
        -----------
        image : ndarray
            Image array (H, W, 3) for RGB or (H, W) for grayscale
        filepath : str or Path
            Output file path
        binary : bool
            Use binary (P6) or ASCII (P3) format
        """
        filepath = Path(filepath)
        
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        height, width = image.shape[:2]
        
        with open(filepath, 'wb') as f:
            if len(image.shape) == 3:  # RGB
                if binary:
                    f.write(b'P6\n')
                else:
                    f.write(b'P3\n')
                
                f.write(f'{width} {height}\n'.encode())
                f.write(b'255\n')
                
                if binary:
                    f.write(image.tobytes())
                else:
                    f.write(' '.join(map(str, image.flatten())).encode())
            
            else:  # Grayscale
                if binary:
                    f.write(b'P5\n')
                else:
                    f.write(b'P2\n')
                
                f.write(f'{width} {height}\n'.encode())
                f.write(b'255\n')
                
                if binary:
                    f.write(image.tobytes())
                else:
                    f.write(' '.join(map(str, image.flatten())).encode())
    
    @staticmethod
    def save_pgm(image: np.ndarray, filepath: Union[str, Path], 
                binary: bool = True):
        """Save image as PGM (grayscale)"""
        filepath = Path(filepath)
        
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            # Convert to grayscale
            image = np.mean(image, axis=2).astype(np.uint8)
        
        height, width = image.shape
        
        with open(filepath, 'wb') as f:
            if binary:
                f.write(b'P5\n')
            else:
                f.write(b'P2\n')
            
            f.write(f'{width} {height}\n'.encode())
            f.write(b'255\n')
            
            if binary:
                f.write(image.tobytes())
            else:
                f.write(' '.join(map(str, image.flatten())).encode())


class ImageGenerator:
    """Generate synthetic images"""
    
    @staticmethod
    def create_blank(height: int, width: int, channels: int = 3,
                    color: int = 0) -> np.ndarray:
        """
        Create blank image
        
        Parameters:
        -----------
        height : int
            Image height
        width : int
            Image width
        channels : int
            Number of channels (1 for grayscale, 3 for RGB)
        color : int
            Fill color (0-255)
        
        Returns:
        --------
        ndarray : Image array
        """
        if channels == 1:
            return np.full((height, width), color, dtype=np.uint8)
        else:
            return np.full((height, width, channels), color, dtype=np.uint8)
    
    @staticmethod
    def create_gradient(height: int, width: int, direction: str = 'horizontal') -> np.ndarray:
        """
        Create gradient image
        
        Parameters:
        -----------
        height : int
            Image height
        width : int
            Image width
        direction : str
            'horizontal' or 'vertical'
        
        Returns:
        --------
        ndarray : Gradient image
        """
        if direction == 'horizontal':
            gradient = np.linspace(0, 255, width, dtype=np.uint8)
            image = np.tile(gradient, (height, 1))
        else:  # vertical
            gradient = np.linspace(0, 255, height, dtype=np.uint8)
            image = np.tile(gradient[:, np.newaxis], (1, width))
        
        return image
    
    @staticmethod
    def create_noise(height: int, width: int, channels: int = 3,
                    noise_type: str = 'gaussian') -> np.ndarray:
        """
        Create noise image
        
        Parameters:
        -----------
        height : int
            Image height
        width : int
            Image width
        channels : int
            Number of channels
        noise_type : str
            'gaussian', 'uniform', 'salt_pepper'
        
        Returns:
        --------
        ndarray : Noise image
        """
        if channels == 1:
            shape = (height, width)
        else:
            shape = (height, width, channels)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(128, 30, shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(0, 256, shape)
        elif noise_type == 'salt_pepper':
            noise = np.random.choice([0, 255], shape)
        
        return np.clip(noise, 0, 255).astype(np.uint8)
