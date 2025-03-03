# utils.py
import cv2
import numpy as np
from typing import Tuple, Optional

def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess frame for model input
    
    Args:
        frame: Input frame
        target_size: Desired output size
        
    Returns:
        Preprocessed frame
    """
    # Resize with aspect ratio preservation
    h, w = frame.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create black canvas of target size
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Center the resized image on canvas
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def optimize_frame_size(frame: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    Optimize frame size for network transmission
    
    Args:
        frame: Input frame
        max_size: Maximum dimension size
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame
        
    scale = max_size / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    
    return cv2.resize(frame, new_size)

class FrameBuffer:
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.frames = []
        
    def add_frame(self, frame: np.ndarray) -> None:
        """Add frame to buffer"""
        self.frames.append(frame)
        if len(self.frames) > self.buffer_size:
            self.frames.pop(0)
            
    def get_smoothed_frame(self) -> Optional[np.ndarray]:
        """Get temporally smoothed frame"""
        if not self.frames:
            return None
            
        return np.mean(self.frames, axis=0).astype(np.uint8)