import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any


@dataclass
class RawLog:
    timestamp: str = None
    event_type: str = None
    details: Dict[str, Any] = field(default_factory=dict)
    monitor: Dict[str, int] = field(default_factory=dict)
    screenshot_bytes: Optional[bytes] = field(default=None, repr=False)
    screenshot_size: Optional[Tuple[int, int]] = field(default=None, repr=False)
    screenshot_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("screenshot_bytes", None)
        d.pop("screenshot_size", None)
        return d

    def __repr__(self) -> str:
        return f"RawLog(timestamp={self.timestamp}, event={self.event}, details={self.details})"

    def from_json(self, data: dict) -> 'RawLog':
        self.timestamp = data.get('timestamp')
        self.event_type = data.get('event')
        self.details = data.get('details', {})
        self.monitor = data.get('monitor', {})
        self.screenshot_bytes = data.get('screenshot_bytes')
        self.screenshot_size = tuple(data.get('screenshot_size', (0, 0)))
        self.screenshot_path = data.get('screenshot_path')
        return self


@dataclass
class RawLogEvents:
    events: list[RawLog] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"events": [event.to_dict() for event in self.events]}

    def __repr__(self) -> str:
        return f"RawLogEvents(events_count={len(self.events)})"

    def sort(self):
        self.events.sort(key=lambda e: e.timestamp)

    def load(self, path: Path, event_types=None):
        events = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if event_types and data.get('event') in event_types:
                    events.append(data)
        self.events = [RawLog().from_json(event) for event in events]
        return self
