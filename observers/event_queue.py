from queue import Queue, Full
from typing import Optional, Tuple
from datetime import datetime

from observers.logs import RawLog


class EventQueue:

    def __init__(self, maxsize=1024):
        self.queue = Queue(maxsize=maxsize)

    def enqueue(self, event_type: str, details: dict, monitor: dict,
                screenshot: Optional[Tuple[bytes, Tuple[int, int]]] = None):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        raw = RawLog(
            timestamp=ts,
            event=event_type,
            details=details,
            monitor=monitor,
            screenshot_bytes=screenshot[0] if screenshot else None,
            screenshot_size=screenshot[1] if screenshot else None,
        )
        try:
            self.queue.put_nowait(raw)
        except Full:
            pass
