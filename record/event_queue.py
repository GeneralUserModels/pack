from queue import Queue, Full
from typing import Optional, Tuple

from modules import RawLog


class EventQueue:

    def __init__(self, maxsize=1024):
        self.queue = Queue(maxsize=maxsize)

    def enqueue(
        self,
        event_type: str,
        details: dict,
        cursor_pos: Optional[Tuple[int, int]] = (None, None),
        screenshot: Optional[Tuple[bytes, Tuple[int, int]]] = None,
        timestamp: Optional[str] = None,
    ):
        raw = RawLog(
            timestamp=timestamp,
            event_type=event_type,
            details=details,
            cursor_pos=cursor_pos,
            screenshot_bytes=screenshot[0] if screenshot else None,
            screenshot_size=screenshot[1] if screenshot else None,
        )
        try:
            self.queue.put_nowait(raw)
        except Full:
            pass
