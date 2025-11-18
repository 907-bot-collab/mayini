import numpy as np
import struct
import os
from typing import Tuple, Optional, Union, Iterator, List
from pathlib import Path


class VideoLoader:
    """
    Custom video loader for frame extraction
    Supports AVI, MP4 (with proper container parsing)
    Falls back to OpenCV if available
    """
    
    def __init__(self):
        """Initialize video loader"""
        self.supported_formats = ['avi', 'mp4', 'mov', 'mkv']
        self.video_data = None
        self.fps = None
        self.frame_count = None
        self.width = None
        self.height = None
    
    def load_video_frames(self, filepath: Union[str, Path],
                         start_frame: int = 0,
                         end_frame: Optional[int] = None,
                         stride: int = 1) -> np.ndarray:
        """
        Load video frames
        
        Parameters:
        -----------
        filepath : str or Path
            Path to video file
        start_frame : int
            Start frame index
        end_frame : int, optional
            End frame index
        stride : int
            Frame sampling stride
        
        Returns:
        --------
        ndarray : Video frames (n_frames, height, width, channels)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Video file not found: {filepath}")
        
        # Try using OpenCV first (if available)
        try:
            import cv2
            return self._load_with_opencv(filepath, start_frame, end_frame, stride)
        except ImportError:
            pass
        
        # Fallback to manual parsing
        extension = filepath.suffix.lower().strip('.')
        
        if extension == 'avi':
            return self._load_avi(filepath, start_frame, end_frame, stride)
        else:
            raise NotImplementedError(f"Manual parsing for {extension} not implemented. "
                                    "Please install opencv-python.")
    
    def _load_with_opencv(self, filepath: Path, start_frame: int = 0,
                         end_frame: Optional[int] = None,
                         stride: int = 1) -> np.ndarray:
        """Load video using OpenCV"""
        import cv2
        
        cap = cv2.VideoCapture(str(filepath))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {filepath}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.fps = fps
        self.frame_count = frame_count
        self.width = width
        self.height = height
        
        if end_frame is None:
            end_frame = frame_count
        
        # Set start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        frame_idx = start_frame
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            # Skip frames based on stride
            for _ in range(stride - 1):
                cap.read()
            
            frame_idx += stride
        
        cap.release()
        
        return np.array(frames, dtype=np.uint8)
    
    def _load_avi(self, filepath: Path, start_frame: int = 0,
                 end_frame: Optional[int] = None,
                 stride: int = 1) -> np.ndarray:
        """
        Load AVI file manually
        Basic implementation - assumes uncompressed video
        """
        with open(filepath, 'rb') as f:
            # Read RIFF header
            riff_header = f.read(4)
            if riff_header != b'RIFF':
                raise ValueError("Invalid AVI file - RIFF header missing")
            
            file_size = struct.unpack('<I', f.read(4))[0]
            avi_header = f.read(4)
            if avi_header != b'AVI ':
                raise ValueError("Invalid AVI file - AVI header missing")
            
            # Parse chunks
            while True:
                try:
                    chunk_id = f.read(4)
                    chunk_size = struct.unpack('<I', f.read(4))[0]
                except:
                    break
                
                if chunk_id == b'LIST':
                    # Parse LIST chunk
                    list_type = f.read(4)
                    if list_type == b'hdrl':
                        # Main header - skip for now
                        f.seek(f.tell() + chunk_size - 4)
                    elif list_type == b'movi':
                        # Movie data - frames are here
                        movi_end = f.tell() + chunk_size - 4
                        
                        # This is simplified - proper implementation requires
                        # parsing stream headers, format, etc.
                        # For production, recommend using OpenCV
                        f.seek(movi_end)
                else:
                    f.seek(f.tell() + chunk_size)
        
        raise NotImplementedError("Full AVI parsing requires OpenCV. "
                                "Please install: pip install opencv-python")
    
    def frame_iterator(self, filepath: Union[str, Path],
                      stride: int = 1,
                      max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        Iterate through video frames (memory-efficient)
        
        Parameters:
        -----------
        filepath : str or Path
            Path to video file
        stride : int
            Frame sampling stride
        max_frames : int, optional
            Maximum number of frames to load
        
        Yields:
        -------
        ndarray : Single frame
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(filepath))
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {filepath}")
            
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                yield frame
                
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
                
                # Skip frames
                for _ in range(stride - 1):
                    if not cap.read()[0]:
                        break
            
            cap.release()
        
        except ImportError:
            raise ImportError("OpenCV required for video loading. "
                            "Install: pip install opencv-python")
    
    def get_frame(self, filepath: Union[str, Path], frame_idx: int) -> np.ndarray:
        """
        Get specific frame from video
        
        Parameters:
        -----------
        filepath : str or Path
            Path to video file
        frame_idx : int
            Frame index
        
        Returns:
        --------
        ndarray : Single frame (height, width, 3)
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(filepath))
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {filepath}")
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise IndexError(f"Cannot read frame {frame_idx}")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame
        
        except ImportError:
            raise ImportError("OpenCV required for video loading.")


class VideoWriter:
    """Write video frames to file"""
    
    @staticmethod
    def write_video(frames: np.ndarray, filepath: Union[str, Path],
                   fps: int = 30, codec: str = 'mp4v'):
        """
        Write frames to video file
        
        Parameters:
        -----------
        frames : ndarray
            Video frames (n_frames, height, width, 3)
        filepath : str or Path
            Output file path
        fps : int
            Frames per second
        codec : str
            Video codec
        """
        try:
            import cv2
            
            filepath = Path(filepath)
            
            if frames.ndim != 4 or frames.shape[3] != 3:
                raise ValueError("Frames must have shape (n_frames, height, width, 3)")
            
            n_frames, height, width, _ = frames.shape
            
            # Define codec and create writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR
                bgr_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            
            out.release()
        
        except ImportError:
            raise ImportError("OpenCV required for video writing.")


class VideoProcessor:
    """Basic video processing utilities"""
    
    @staticmethod
    def resize_frames(frames: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize video frames
        
        Parameters:
        -----------
        frames : ndarray
            Video frames (n_frames, height, width, channels)
        size : tuple
            Target size (height, width)
        
        Returns:
        --------
        ndarray : Resized frames
        """
        try:
            import cv2
            
            n_frames = frames.shape[0]
            height, width = size
            channels = frames.shape[3]
            
            resized = np.zeros((n_frames, height, width, channels), 
                             dtype=frames.dtype)
            
            for i in range(n_frames):
                resized[i] = cv2.resize(frames[i], (width, height))
            
            return resized
        
        except ImportError:
            raise ImportError("OpenCV required for video resizing.")
    
    @staticmethod
    def to_grayscale(frames: np.ndarray) -> np.ndarray:
        """
        Convert video frames to grayscale
        
        Parameters:
        -----------
        frames : ndarray
            Video frames (n_frames, height, width, 3)
        
        Returns:
        --------
        ndarray : Grayscale frames (n_frames, height, width)
        """
        try:
            import cv2
            
            n_frames = frames.shape[0]
            height, width = frames.shape[1:3]
            
            grayscale = np.zeros((n_frames, height, width), 
                               dtype=np.uint8)
            
            for i in range(n_frames):
                grayscale[i] = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            return grayscale
        
        except ImportError:
            raise ImportError("OpenCV required for grayscale conversion.")
    
    @staticmethod
    def extract_keyframes(frames: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """
        Extract keyframes based on frame difference
        
        Parameters:
        -----------
        frames : ndarray
            Video frames
        threshold : float
            Difference threshold
        
        Returns:
        --------
        ndarray : Keyframe indices
        """
        keyframe_indices = [0]  # First frame is always a keyframe
        
        # Convert to grayscale for comparison
        gray_frames = np.mean(frames, axis=3)  # Simple grayscale
        
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(gray_frames[i] - gray_frames[i-1]))
            
            if diff > threshold:
                keyframe_indices.append(i)
        
        return np.array(keyframe_indices)
