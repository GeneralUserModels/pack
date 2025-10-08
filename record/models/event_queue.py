import threading
import time
from typing import List, Callable, Optional
from collections import deque
from dataclasses import dataclass
from record.models.event import InputEvent, EventType


@dataclass
class AggregationConfig:
    gap_threshold: float  # Max time gap between consecutive events (seconds)
    total_threshold: float  # Max time span from first to last event (seconds)


@dataclass
class AggregationRequest:
    """Represents a request for a screenshot at a specific time."""
    timestamp: float
    reason: str  # e.g., "keyboard_start", "mouse_move_end"
    event_type: str  # e.g., "key", "move"
    is_start: bool  # True for start, False for end


class EventQueue:

    def __init__(
        self,
        click_config: Optional[AggregationConfig] = None,
        move_config: Optional[AggregationConfig] = None,
        scroll_config: Optional[AggregationConfig] = None,
        key_config: Optional[AggregationConfig] = None,
        poll_interval: float = 1.0
    ):
        """
        Initialize the input event queue.

        Args:
            click_config: Configuration for click event aggregation
            move_config: Configuration for move event aggregation
            scroll_config: Configuration for scroll event aggregation
            key_config: Configuration for key event aggregation
            poll_interval: Interval in seconds for polling worker
        """
        self.configs = {
            'click': click_config or AggregationConfig(gap_threshold=0.5, total_threshold=2.0),
            'move': move_config or AggregationConfig(gap_threshold=0.1, total_threshold=1.0),
            'scroll': scroll_config or AggregationConfig(gap_threshold=0.3, total_threshold=1.5),
            'key': key_config or AggregationConfig(gap_threshold=0.5, total_threshold=3.0),
        }

        # Aggregation queues for each event type
        self.aggregations = {
            'click': deque(),
            'move': deque(),
            'scroll': deque(),
            'key': deque(),
        }

        # Queue to store ALL input events
        self.all_events = deque()

        # Queue to store aggregation requests (screenshot requests)
        self.aggregation_requests = deque()

        # List of ready aggregation requests to be processed by callback
        self.final_aggregations = []

        # Time threshold - we can safely aggregate events older than this
        self.safe_aggregation_time = 0.0

        # Event type mapping
        self.event_type_mapping = {
            EventType.MOUSE_DOWN: 'click',
            EventType.MOUSE_UP: 'click',
            EventType.MOUSE_MOVE: 'move',
            EventType.MOUSE_SCROLL: 'scroll',
            EventType.KEY_PRESS: 'key',
            EventType.KEY_RELEASE: 'key',
        }

        self._lock = threading.RLock()
        self._callback: Optional[Callable[[str, List[InputEvent]], None]] = None
        self._aggregation_callback: Optional[Callable[[List[AggregationRequest]], None]] = None

        # Polling worker
        self.poll_interval = poll_interval
        self._running = False
        self._poll_thread = None

    def set_callback(self, callback: Callable[[str, List[InputEvent]], None]) -> None:
        """
        Set the callback function for processed aggregated events.

        Args:
            callback: Function that takes (event_type_str, events_list)
        """
        with self._lock:
            self._callback = callback

    def set_aggregation_callback(
        self,
        callback: Callable[[List[AggregationRequest]], None]
    ) -> None:
        """
        Set the callback function for ready aggregation requests.

        Args:
            callback: Function that takes a list of AggregationRequest objects
        """
        with self._lock:
            self._aggregation_callback = callback

    def enqueue(self, event: InputEvent) -> None:
        """
        Add an event to the appropriate aggregation queue and all_events queue.

        Args:
            event: Input event to add
        """
        with self._lock:
            # Add to all_events queue
            self.all_events.append(event)

            # Get the aggregation type
            agg_type = self.event_type_mapping.get(event.event_type)
            if agg_type is None:
                return

            queue = self.aggregations[agg_type]
            config = self.configs[agg_type]

            # If queue is empty, just add the event
            if not queue:
                queue.append(event)
                return

            last_event = queue[-1]
            first_event = queue[0]

            gap_diff = event.timestamp - last_event.timestamp
            total_diff = event.timestamp - first_event.timestamp

            # Check conditions
            gap_ok = gap_diff <= config.gap_threshold
            total_ok = total_diff <= config.total_threshold

            if gap_ok and total_ok:
                # Both conditions met: just append
                queue.append(event)
            elif not gap_ok:
                # Gap too large: process current aggregation and start new one
                self._process_aggregation(agg_type, list(queue))
                queue.clear()
                queue.append(event)
            else:  # not total_ok
                # Total span too large: split and process half
                queue_list = list(queue)
                mid_timestamp = (first_event.timestamp + last_event.timestamp) / 2

                # Split based on timestamp
                first_half = [e for e in queue_list if e.timestamp <= mid_timestamp]
                second_half = [e for e in queue_list if e.timestamp > mid_timestamp]

                # Process first half
                if first_half:
                    self._process_aggregation(agg_type, first_half)

                # Keep second half and add new event
                queue.clear()
                queue.extend(second_half)
                queue.append(event)

    def _process_aggregation(self, agg_type: str, events: List[InputEvent]) -> None:
        """
        Process an aggregated event batch by creating aggregation requests.

        Args:
            agg_type: Type of aggregation ('click', 'move', 'scroll', 'key')
            events: List of events to process
        """
        if not events:
            return

        # Create start and end aggregation requests
        start_request = AggregationRequest(
            timestamp=events[0].timestamp,
            reason=f"{agg_type}_start",
            event_type=agg_type,
            is_start=True
        )

        end_request = AggregationRequest(
            timestamp=events[-1].timestamp,
            reason=f"{agg_type}_end",
            event_type=agg_type,
            is_start=False
        )

        self.aggregation_requests.append(start_request)
        self.aggregation_requests.append(end_request)

        # Call the original callback if set
        if self._callback:
            try:
                self._callback(agg_type, events)
            except Exception as e:
                print(f"Error in aggregation callback for {agg_type}: {e}")

    def _poll_worker(self) -> None:
        """Worker thread that polls for stale aggregations and processes ready requests."""
        while self._running:
            time.sleep(self.poll_interval)

            with self._lock:
                current_time = time.time()

                for agg_type, queue in self.aggregations.items():
                    if not queue:
                        continue

                    config = self.configs[agg_type]
                    last_event = queue[-1]
                    time_since_last = current_time - last_event.timestamp

                    if time_since_last > config.gap_threshold:
                        self._process_aggregation(agg_type, list(queue))
                        queue.clear()

                max_threshold = max(cfg.gap_threshold for cfg in self.configs.values()) + 1
                self.safe_aggregation_time = current_time - max_threshold

                self._prepare_final_aggregations()

    def _prepare_final_aggregations(self) -> None:
        """
        Move aggregation requests that are ready to final_aggregations list
        and trigger callback.
        """
        if not self.aggregation_requests:
            return

        # Find all requests that are safe to process
        ready_requests = []
        remaining_requests = deque()

        for req in self.aggregation_requests:
            if req.timestamp <= self.safe_aggregation_time:
                ready_requests.append(req)
            else:
                remaining_requests.append(req)

        self.aggregation_requests = remaining_requests

        if not ready_requests:
            return

        # Sort requests by timestamp
        ready_requests.sort(key=lambda r: r.timestamp)

        # Add to final_aggregations and trigger callback
        self.final_aggregations.extend(ready_requests)

        if self._aggregation_callback:
            try:
                self._aggregation_callback(list(ready_requests))
            except Exception as e:
                print(f"Error in aggregation callback: {e}")

    def start(self) -> None:
        """Start the polling worker."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._poll_thread = threading.Thread(target=self._poll_worker, daemon=True)
            self._poll_thread.start()

    def stop(self) -> None:
        """Stop the polling worker and process all remaining aggregations."""
        with self._lock:
            self._running = False

            # Process all remaining aggregations
            for agg_type, queue in self.aggregations.items():
                if queue:
                    self._process_aggregation(agg_type, list(queue))
                    queue.clear()

            # Update safe time to current time and process all remaining requests
            self.safe_aggregation_time = time.time()
            self._prepare_final_aggregations()

        # Wait for poll thread to finish
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
