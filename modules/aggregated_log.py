from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List


@dataclass
class AggregatedLog:
    start_timestamp: str
    end_timestamp: str
    monitor: Dict[str, int]
    start_screenshot_path: Optional[str] = None
    end_screenshot_path: Optional[str] = None
    start_cursor_pos: Optional[Tuple[int, int]] = None
    end_cursor_pos: Optional[Tuple[int, int]] = None
    click_positions: Optional[List[Tuple[int, int]]] = None
    scroll_directions: Optional[List[Tuple[int, int]]] = None
    keys_pressed: Optional[List[List[str]]] = None
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        # return asdict(self)
        return {
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "monitor": self.monitor,
            "start_screenshot_path": self.start_screenshot_path,
            "end_screenshot_path": self.end_screenshot_path,
            "start_cursor_pos": self.start_cursor_pos,
            "end_cursor_pos": self.end_cursor_pos,
            "events": self.events,
        }

    def __repr__(self) -> str:
        return f"Log(start_timestamp={self.start_timestamp}, end_timestamp={self.end_timestamp}, keys_pressed={''.join(self.keys_pressed)})"
