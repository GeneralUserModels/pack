from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AggregationConfig:
    gap_threshold: float  # Max time gap between consecutive events (seconds)
    total_threshold: float  # Max time span from first to last event (seconds)


@dataclass
class AggregationRequest:
    """Represents a request for a screenshot at a specific time."""
    timestamp: float
    end_timestamp: Optional[float]
    reason: str  # e.g., "keyboard_start", "mouse_move_end"
    event_type: str  # e.g., "key", "move"
    is_start: bool  # True for start, False for end


@dataclass
class ProcessedAggregation:
    """Represents an aggregation with matched screenshot and events."""
    request: AggregationRequest
    screenshot: Optional[any]  # Screenshot object from ImageQueue
    screenshot_path: Optional[str]  # Path to saved screenshot
    events: List[dict]  # Events between this and next aggregation
