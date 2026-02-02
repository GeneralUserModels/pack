from dataclasses import dataclass
import numpy as np


@dataclass
class BufferImage:
    timestamp: float
    data: np.ndarray  # RGB numpy array
    monitor_index: int = 0
    monitor_dict: dict = None
    scale_factor: float = 1.0

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'monitor_index': self.monitor_index,
            'shape': self.data.shape if self.data is not None else None,
            'monitor_dict': self.monitor_dict,
            'scale_factor': self.scale_factor
        }
