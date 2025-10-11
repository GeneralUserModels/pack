import threading
import time
from pathlib import Path
from typing import Callable, Optional, Dict
from collections import deque
from record.models.event import InputEvent, EventType
from record.models.aggregation import AggregationConfig, AggregationRequest


class EventQueue:

    def __init__(
        self,
        click_config: Optional[AggregationConfig] = None,
        move_config: Optional[AggregationConfig] = None,
        scroll_config: Optional[AggregationConfig] = None,
        key_config: Optional[AggregationConfig] = None,
        poll_interval: float = 1.0,
        session_dir: Path = None,
        safety_margin: float = 0.5
    ):
        """
        Initialize the input event queue.

        Args:
            click_config: Configuration for click event aggregation
            move_config: Configuration for move event aggregation
            scroll_config: Configuration for scroll event aggregation
            key_config: Configuration for key event aggregation
            poll_interval: Interval in seconds for polling worker
            session_dir: Directory to save events JSONL
            safety_margin: Safety margin in seconds to avoid async delivery issues
        """
        self.session_dir = session_dir
        self.safety_margin = safety_margin
        self.configs = {
            'click': click_config,
            'move': move_config,
            'scroll': scroll_config,
            'key': key_config,
        }

        self.aggregations = {
            'click': deque(),
            'move': deque(),
            'scroll': deque(),
            'key': deque(),
        }

        self.all_events = deque()

        self.active_bursts: Dict[int, dict] = {}
        self.next_burst_id = 0

        self.completed_bursts = deque()

        self.event_type_mapping = {
            EventType.MOUSE_DOWN: 'click',
            EventType.MOUSE_UP: 'click',
            EventType.MOUSE_MOVE: 'move',
            EventType.MOUSE_SCROLL: 'scroll',
            EventType.KEY_PRESS: 'key',
            EventType.KEY_RELEASE: 'key',
        }

        self._lock = threading.RLock()
        self._callback: Optional[Callable[[AggregationRequest], None]] = None

        # Polling worker
        self.poll_interval = poll_interval
        self._running = False
        self._poll_thread = None

    def set_callback(self, callback: Callable[[AggregationRequest], None]) -> None:
        """
        Set the callback function for processing a completed burst.

        Args:
            callback: Function that takes a single AggregationRequest object
        """
        with self._lock:
            self._callback = callback

    def enqueue(self, event: InputEvent) -> None:
        """
        Add an event to all_events and check if it triggers burst boundaries.

        Args:
            event: Input event to add
        """
        with self._lock:
            # Add to global all_events queue
            self.all_events.append(event)
            self._save_event_to_jsonl(event)

            # Map event to aggregation type
            agg_type = self.event_type_mapping.get(event.event_type)
            if agg_type is None:
                return

            queue = self.aggregations[agg_type]
            config = self.configs[agg_type]

            # Case 1: Queue is empty, start new burst
            if not queue:
                burst_id = self._start_burst(agg_type, event)
                queue.append(event)
                return

            last_event = queue[-1]
            first_event = queue[0]

            gap_diff = event.timestamp - last_event.timestamp
            total_diff = event.timestamp - first_event.timestamp

            gap_ok = gap_diff <= config.gap_threshold
            total_ok = total_diff <= config.total_threshold

            # Case 2: Gap threshold exceeded, end current burst and start new one
            if not gap_ok:
                burst_id = self._find_burst_by_type(agg_type)
                if burst_id is not None:
                    self._end_burst(burst_id, last_event.timestamp)

                burst_id = self._start_burst(agg_type, event)
                queue.clear()
                queue.append(event)

            # Case 3: Total threshold exceeded (but gap OK), split the burst
            elif not total_ok:
                queue_list = list(queue)
                mid_timestamp = (first_event.timestamp + last_event.timestamp) / 2

                # Find the burst and end it at midpoint
                burst_id = self._find_burst_by_type(agg_type)
                if burst_id is not None:
                    self._end_burst(burst_id, mid_timestamp)

                # Start new burst with second half
                burst_id = self._start_burst(agg_type, event)
                queue.clear()
                queue.append(event)

            # Case 4: Both thresholds OK, just append to current burst
            else:
                queue.append(event)

    def _start_burst(self, event_type: str, event: InputEvent) -> int:
        """
        Start a new burst and return its ID.

        Args:
            event_type: Type of event ('click', 'move', 'scroll', 'key')
            event: The triggering event

        Returns:
            Burst ID
        """
        burst_id = self.next_burst_id
        self.next_burst_id += 1

        self.active_bursts[burst_id] = {
            'event_type': event_type,
            'start_time': event.timestamp,
            'end_time': None,
            'start_request': AggregationRequest(
                timestamp=event.timestamp,
                reason=f"{event_type}_start",
                event_type=event_type,
                is_start=True,
                end_timestamp=None
            ),
            'end_request': None
        }

        return burst_id

    def _end_burst(self, burst_id: int, end_time: float) -> None:
        """
        End a burst and move it to completed queue.

        Args:
            burst_id: ID of the burst to end
            end_time: Timestamp when the burst ended
        """
        if burst_id not in self.active_bursts:
            return

        burst = self.active_bursts[burst_id]
        burst['end_time'] = end_time
        burst['end_request'] = AggregationRequest(
            timestamp=end_time,
            reason=f"{burst['event_type']}_end",
            event_type=burst['event_type'],
            is_start=False,
            end_timestamp=None
        )

        # Move to completed bursts
        self.completed_bursts.append(burst)
        del self.active_bursts[burst_id]

    def _find_burst_by_type(self, event_type: str) -> Optional[int]:
        """Find the most recent active burst of a given type."""
        for burst_id, burst in self.active_bursts.items():
            if burst['event_type'] == event_type:
                return burst_id
        return None

    def _save_event_to_jsonl(self, event: InputEvent) -> None:
        """Save event to JSONL file."""
        if self.session_dir:
            try:
                with open(self.session_dir / "events.jsonl", "a") as f:
                    f.write(str(event.to_dict()) + "\n")
            except Exception as e:
                print(f"Error saving event to JSONL: {e}")

    def _poll_stale_bursts(self) -> None:
        """
        Check if any active burst's last event is stale (no new events within gap threshold).
        If so, end that burst.
        """
        current_time = time.time()

        with self._lock:
            burst_ids_to_end = []

            for burst_id, burst in list(self.active_bursts.items()):
                event_type = burst['event_type']
                config = self.configs[event_type]

                # Check if burst is stale (no activity within gap threshold)
                queue = self.aggregations[event_type]
                if queue:
                    last_event = queue[-1]
                    time_since_last_event = current_time - last_event.timestamp

                    if time_since_last_event > config.gap_threshold:
                        burst_ids_to_end.append((burst_id, event_type, last_event.timestamp))

            # End stale bursts
            for burst_id, event_type, end_time in burst_ids_to_end:
                if burst_id in self.active_bursts:
                    self._end_burst(burst_id, end_time)
                    self.aggregations[event_type].clear()

    def _process_ready_bursts(self) -> None:
        """
        Process bursts that are old enough that no new requests can arrive.
        A burst is ready when the next burst started more than (max_threshold + safety_margin) ago.
        """
        with self._lock:
            if len(self.completed_bursts) < 2:
                return

            max_threshold = max(cfg.total_threshold for cfg in self.configs.values() if cfg)
            ready_threshold = max_threshold + self.safety_margin
            current_time = time.time()

            # Find bursts that are old enough to process
            bursts_to_process = []
            bursts_to_keep = deque()

            for i, burst in enumerate(self.completed_bursts):
                # A burst is ready if there's a newer burst that's old enough
                is_last = (i == len(self.completed_bursts) - 1)

                if not is_last:
                    # Check next burst
                    next_burst = list(self.completed_bursts)[i + 1]
                    time_since_next = current_time - next_burst['start_time']

                    if time_since_next > ready_threshold:
                        bursts_to_process.append(burst)
                    else:
                        bursts_to_keep.append(burst)
                else:
                    # Keep the last burst in queue
                    bursts_to_keep.append(burst)

            # Process ready bursts
            for burst in bursts_to_process:
                self._emit_burst_requests(burst)

            self.completed_bursts = bursts_to_keep

    def _emit_burst_requests(self, burst: dict) -> None:
        """
        Emit both start and end requests for a burst through the callback.

        Args:
            burst: Burst dictionary with start_request and end_request
        """
        start_req = burst['start_request']
        end_req = burst['end_request']

        # Set end_timestamp on start request
        start_req.end_timestamp = end_req.timestamp

        # Callback for start request
        if self._callback:
            try:
                self._callback(start_req)
            except Exception as e:
                print(f"Error in burst start callback: {e}")

        # Callback for end request (end_timestamp points to next burst start or is None)
        if self._callback:
            try:
                self._callback(end_req)
            except Exception as e:
                print(f"Error in burst end callback: {e}")

    def _poll_worker(self) -> None:
        """Worker thread that checks for stale bursts and processes ready ones."""
        while self._running:
            try:
                self._poll_stale_bursts()
                self._process_ready_bursts()
                time.sleep(self.poll_interval)
            except Exception as e:
                import traceback
                print(f"Error in poll worker: {e}")
                print(traceback.format_exc())
                time.sleep(self.poll_interval)

    def process_all_remaining(self) -> None:
        """
        Process all remaining bursts and events.
        Call this on ScreenRecorder.stop().
        """
        with self._lock:
            # End all active bursts
            burst_ids_to_end = list(self.active_bursts.keys())
            current_time = time.time()

            for burst_id in burst_ids_to_end:
                burst = self.active_bursts[burst_id]
                queue = self.aggregations[burst['event_type']]
                if queue:
                    last_event = queue[-1]
                    self._end_burst(burst_id, last_event.timestamp)
                else:
                    self._end_burst(burst_id, current_time)

            # Process all completed bursts
            while self.completed_bursts:
                burst = self.completed_bursts.popleft()
                self._emit_burst_requests(burst)

            # Clear all aggregation queues
            for queue in self.aggregations.values():
                queue.clear()

    def start(self) -> None:
        """Start the polling worker."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._poll_thread = threading.Thread(target=self._poll_worker, daemon=True)
            self._poll_thread.start()

    def stop(self) -> None:
        """Stop the polling worker."""
        with self._lock:
            self._running = False

        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
