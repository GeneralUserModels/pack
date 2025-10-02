import threading
from typing import List, Callable, Optional, Any
from collections import deque
import bisect


class EventQueue:
    """Thread-safe circular queue with timestamp-based retrieval."""

    def __init__(self, max_length: int):
        """
        Initialize the event queue.

        Args:
            max_length: Maximum number of items to store in the circular buffer
        """
        self.max_length = max_length
        self._queue = deque(maxlen=max_length)
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[Any], None]] = []

    def enqueue(self, item: Any) -> None:
        """
        Add an item to the queue and trigger callbacks.

        Args:
            item: Item to add (must have a timestamp attribute)
        """
        with self._lock:
            # Insert in sorted order
            if not self._queue or item.timestamp >= self._queue[-1].timestamp:
                self._queue.append(item)
            else:
                # Convert to list, insert, and recreate deque
                temp_list = list(self._queue)
                idx = bisect.bisect_left([x.timestamp for x in temp_list], item.timestamp)
                temp_list.insert(idx, item)
                self._queue = deque(temp_list[-self.max_length:], maxlen=self.max_length)

            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback(item)
                except Exception as e:
                    print(f"Error in callback: {e}")

    def get_entries_before(self, timestamp: float, milliseconds: int) -> List[Any]:
        """
        Get all entries within X milliseconds before the given timestamp.

        Args:
            timestamp: Reference timestamp
            milliseconds: Time window in milliseconds

        Returns:
            List of items within the time window
        """
        with self._lock:
            start_time = timestamp - (milliseconds / 1000.0)
            return [item for item in self._queue
                    if start_time <= item.timestamp <= timestamp]

    def get_entries_after(self, timestamp: float, milliseconds: int) -> List[Any]:
        """
        Get all entries within X milliseconds after the given timestamp.

        Args:
            timestamp: Reference timestamp
            milliseconds: Time window in milliseconds

        Returns:
            List of items within the time window
        """
        with self._lock:
            end_time = timestamp + (milliseconds / 1000.0)
            return [item for item in self._queue
                    if timestamp <= item.timestamp <= end_time]

    def get_latest(self) -> Optional[Any]:
        """
        Get the most recent item from the queue.

        Returns:
            The latest item or None if queue is empty
        """
        with self._lock:
            return self._queue[-1] if self._queue else None

    def add_callback(self, callback: Callable[[Any], None]) -> None:
        """
        Register a callback to be called when new items are added.

        Args:
            callback: Function to call with the new item
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Any], None]) -> None:
        """
        Remove a registered callback.

        Args:
            callback: Function to remove
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def get_all(self) -> List[Any]:
        """Get all items in the queue."""
        with self._lock:
            return list(self._queue)

    def clear(self) -> None:
        """Clear all items from the queue."""
        with self._lock:
            self._queue.clear()

    def __len__(self) -> int:
        """Get the current size of the queue."""
        with self._lock:
            return len(self._queue)
