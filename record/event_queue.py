from queue import Queue, Full
from typing import Optional, Tuple
from datetime import datetime

from modules import RawLog


class EventQueue:

    def __init__(self, maxsize=1024):
        self.queue = Queue(maxsize=maxsize)

    def enqueue(
        self,
        event_type: str,
        details: dict,
        monitor: dict,
        cursor_pos: Optional[Tuple[int, int]] = (None, None),
        screenshot: Optional[Tuple[bytes, Tuple[int, int]]] = None
    ):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        raw = RawLog(
            timestamp=ts,
            event_type=event_type,
            details=details,
            monitor=monitor,
            cursor_pos=cursor_pos,
            screenshot_bytes=screenshot[0] if screenshot else None,
            screenshot_size=screenshot[1] if screenshot else None,
        )
        try:
            self.queue.put_nowait(raw)
        except Full:
            pass
