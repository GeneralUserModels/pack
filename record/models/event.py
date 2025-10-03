from dataclasses import dataclass
from typing import Any, Dict
from enum import Enum


class EventType(Enum):
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    MOUSE_MOVE = "mouse_move"
    MOUSE_SCROLL = "mouse_scroll"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"


@dataclass
class InputEvent:
    timestamp: float
    monitor_index: int
    event_type: EventType
    details: Dict[str, Any]

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'monitor_index': self.monitor_index,
            'event_type': self.event_type.value,
            'details': self.details
        }
