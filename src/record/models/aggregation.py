from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class AggregationConfig:
    gap_threshold: float  # Max time gap between consecutive events (seconds)
    total_threshold: float  # Max time span from first to last event (seconds)


@dataclass
class AggregationRequest:
    """Represents a request for a screenshot at a specific time."""
    timestamp: float
    end_timestamp: Optional[float]
    reason: str  # e.g., "key_start", "mouse_move_end"
    event_type: str  # e.g., "key", "move"
    request_state: str  # burst start, mid or end
    screenshot: Optional[Any] = None  # Screenshot object from ImageQueue
    screenshot_path: Optional[str] = None  # Path to saved screenshot
    screenshot_timestamp: Optional[float] = None  # Timestamp of the screenshot
    end_screenshot_timestamp: Optional[float] = None  # Timestamp of end screenshot
    monitor: Optional[dict] = None  # Monitor info at the time of screenshot
    burst_id: Optional[int] = None  # ID of the burst this request belongs to
    scale_factor: float = 1.0  # Scaling factor of the screenshot
    monitor_index: Optional[int] = None  # Index of the monitor


@dataclass
class ProcessedAggregation:
    """Represents an aggregation with matched screenshot and events."""
    request: AggregationRequest
    events: List[dict]

    @property
    def screenshot(self) -> Optional[Any]:
        """Get screenshot from request."""
        return self.request.screenshot

    @property
    def screenshot_path(self) -> Optional[str]:
        """Get screenshot path from request."""
        return self.request.screenshot_path

    def to_dict(self):
        request = self.request.__dict__.copy()
        request.pop('screenshot', None)
        return {
            "request": request,
            "events": self.events
        }
