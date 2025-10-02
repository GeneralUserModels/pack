from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BufferImage:
    timestamp: float
    screenshot: np.ndarray
    ssim_value: Optional[float] = None
    monitor_index: int = 0

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'ssim_value': self.ssim_value,
            'monitor_index': self.monitor_index,
            'shape': self.screenshot.shape if self.screenshot is not None else None
        }
