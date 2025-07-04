from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any


@dataclass
class RawLog:
    timestamp: str
    event: str
    details: Dict[str, Any]
    monitor: Dict[str, int]
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
