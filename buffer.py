import numpy as np
import torch
from collections import deque

class SequenceBuffer:
    def __init__(self, target_frames=100):
        self.target_frames = target_frames
        self.buffer = deque(maxlen=target_frames)
        
    def add_frame(self, landmarks):
        """
        landmarks: numpy array of shape (543, 3)
        """
        self.buffer.append(landmarks)
        
    def get_tensor(self):
        """
        Returns a PyTorch tensor of shape (1, target_frames, 543, 3).
        If the buffer hasn't reached target_frames, it pads with zeros.
        """
        if len(self.buffer) == 0:
            return torch.zeros((1, self.target_frames, 543, 3), dtype=torch.float32)
            
        current_frames = len(self.buffer)
        
        # Convert buffer to numpy array (current_frames, 543, 3)
        seq_array = np.array(self.buffer)
        
        # We need padding if current_frames < target_frames
        if current_frames < self.target_frames:
            padding = np.zeros((self.target_frames - current_frames, 543, 3))
            seq_array = np.concatenate([seq_array, padding], axis=0)
            
        # Convert to PyTorch Tensor and add Batch Dimension
        tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0)
        
        return tensor

    def is_ready(self, min_frames=30):
        """
        Returns True if we have at least min_frames to process an inference.
        """
        return len(self.buffer) >= min_frames

    def clear(self):
        self.buffer.clear()
