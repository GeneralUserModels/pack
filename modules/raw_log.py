import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any
from datetime import datetime


@dataclass
class RawLog:
    timestamp: str = None
    event_type: str = None
    details: Dict[str, Any] = field(default_factory=dict)
    cursor_pos: Optional[Tuple[int, int]] = (None, None)
    screenshot_bytes: Optional[bytes] = field(default=None, repr=False)
    screenshot_size: Optional[Tuple[int, int]] = field(default=None, repr=False)
    screenshot_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("screenshot_bytes", None)
        d.pop("screenshot_size", None)
        return d

    @classmethod
    def from_json(self, data: dict) -> 'RawLog':
        self = RawLog()
        self.timestamp = data.get('timestamp')
        self.event_type = data.get('event_type')
        self.details = data.get('details', {})
        self.cursor_pos = tuple(data.get('cursor_pos', (None, None)))
        self.screenshot_bytes = data.get('screenshot_bytes')
        self.screenshot_size = tuple(data.get('screenshot_size', (0, 0)))
        self.screenshot_path = data.get('screenshot_path')
        return self

    def copy(self) -> 'RawLog':
        return RawLog(
            timestamp=self.timestamp,
            event_type=self.event_type,
            details=self.details.copy(),
            cursor_pos=self.cursor_pos,
            screenshot_bytes=self.screenshot_bytes,
            screenshot_size=self.screenshot_size,
            screenshot_path=self.screenshot_path
        )

    @property
    def strp_timestamp(self) -> datetime:
        return datetime.strptime(self.timestamp, "%Y-%m-%d_%H-%M-%S-%f").timestamp()

    def __gt__(self, other: 'RawLog') -> bool:
        return self.timestamp > other.timestamp

    def __lt__(self, other: 'RawLog') -> bool:
        return self.timestamp < other.timestamp

    def __repr__(self) -> str:
        return f"RawLog(timestamp={self.timestamp}, event_type={self.event_type}, details={self.details})"


@dataclass
class RawLogEvents:
    events: list[RawLog] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"events": [event.to_dict() for event in self.events]}

    def sort(self):
        self.events.sort(key=lambda e: e.timestamp)

    def load(self, path: Path, event_types=None):
        events = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if event_types and data.get('event') not in event_types:
                    continue
                events.append(data)
        self.events = [RawLog().from_json(event) for event in events]
        return self

    def format_to_event_list(self) -> list:
        return [
            {"timestamp": ev.timestamp, "event_type": ev.event_type, "details": ev.details, "cursor_pos": ev.cursor_pos}
            for ev in self.events
        ]

    def append(self, event: RawLog):
        self.events.append(event)

    def extend(self, events: list[RawLog]):
        self.events.extend(events)

    def index(self, event: RawLog) -> int:
        return self.events.index(event)

    def __iter__(self):
        return iter(self.events)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        if isinstance(item, slice):
            sliced_events = self.events[item]
            new_instance = RawLogEvents()
            new_instance.events = sliced_events
            return new_instance
        else:
            return self.events[item]

    def __setitem__(self, key, value):
        self.events[key] = value

    def __repr__(self) -> str:
        return f"RawLogEvents(events_count={len(self.events)})"
