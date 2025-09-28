import time
from queue import Queue, Full
from collections import deque
from typing import Optional, Tuple, List
from modules import RawLog


class EventQueue:
    """
    A thread-safe enqueueing queue for IO plus an in-memory time-windowed event buffer
    (to keep the last N seconds of events for the heuristic).
    """

    def __init__(self, maxsize=1024, buffer_seconds: float = 6.0):
        self.queue = Queue(maxsize=maxsize)
        self._buffer_seconds = float(buffer_seconds)
        self._in_memory = deque()  # stores (unix_ts_float, RawLog)
        self._id_counter = 0

    def enqueue(
        self,
        event_type: str,
        details: dict,
        cursor_pos: Optional[Tuple[int, int]] = (None, None),
        screenshot: Optional[Tuple[bytes, Tuple[int, int]]] = None,
        timestamp: Optional[str] = None,
    ):
        unix_ts = time.time()
        raw = RawLog(
            timestamp=timestamp,
            event_type=event_type,
            details=details,
            cursor_pos=cursor_pos,
            screenshot_bytes=screenshot[0] if screenshot else None,
            screenshot_size=screenshot[1] if screenshot else None,
        )
        # assign incremental id for dedup/tracking
        self._id_counter += 1
        raw.id = self._id_counter

        try:
            self.queue.put_nowait(raw)
        except Full:
            pass

        # append to in-memory buffer and trim
        self._in_memory.append((unix_ts, raw))
        self._trim_buffer()

    def _trim_buffer(self):
        cutoff = time.time() - self._buffer_seconds
        while self._in_memory and self._in_memory[0][0] < cutoff:
            self._in_memory.popleft()

    def get_recent_events(self, since_ts: float = 0.0, until_ts: float = None) -> List[Tuple[float, RawLog]]:
        """Return events with unix timestamp in [since_ts, until_ts]."""
        if until_ts is None:
            until_ts = time.time()
        res = []
        for ts, raw in reversed(self._in_memory):
            if ts < since_ts:
                break
            if ts <= until_ts:
                res.append((ts, raw))
        return list(reversed(res))

    def get_last_event_before(self, ts: float, max_delta: float):
        """
        Return the most recent event with event_ts <= ts and ts - event_ts <= max_delta,
        or None if none.
        """
        cutoff = ts - max_delta
        # iterate reversed (newest first)
        for ev_ts, raw in reversed(self._in_memory):
            if ev_ts > ts:
                continue
            if ev_ts >= cutoff:
                return ev_ts, raw
            else:
                break
        return None
