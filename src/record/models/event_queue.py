import threading
import time
import heapq
import itertools
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from collections import deque
from record.models.event import InputEvent, EventType
from record.models.aggregation import AggregationConfig, AggregationRequest
from record.constants import constants_manager


class EventQueue:

    def __init__(
        self,
        image_queue: Any,  # ImageQueue instance
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
            image_queue: ImageQueue instance for fetching screenshots
            click_config: Configuration for click event aggregation
            move_config: Configuration for move event aggregation
            scroll_config: Configuration for scroll event aggregation
            key_config: Configuration for key event aggregation
            poll_interval: Interval in seconds for polling worker
            session_dir: Directory to save events JSONL
            safety_margin: Safety margin in seconds to avoid async delivery issues
        """
        self.image_queue = image_queue
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

        # Store individual requests in chronological order
        self._pending_requests_heap = []
        self._pending_counter = itertools.count()

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
        Caches screenshot alongside event in the aggregation queue.

        Args:
            event: Input event to add
        """
        with self._lock:
            self.all_events.append(event)
            self._save_event_to_jsonl(event)

            agg_type = self.event_type_mapping.get(event.event_type)
            if agg_type is None:
                return

            queue = self.aggregations[agg_type]
            config = self.configs[agg_type]

            # Case 1: Queue is empty, start new burst
            if not queue:
                screenshot = self._find_screenshot_before(event.timestamp)
                burst_id = self._start_burst(agg_type, event, screenshot)
                queue.append((event, screenshot))
                return

            last_event, last_screenshot = queue[-1]
            first_event, first_screenshot = queue[0]

            # Check if monitor changed - force new burst if so
            monitor_changed = event.monitor_index != last_event.monitor_index

            gap_diff = event.timestamp - last_event.timestamp
            total_diff = event.timestamp - first_event.timestamp

            gap_ok = gap_diff <= config.gap_threshold
            total_ok = total_diff <= config.total_threshold

            # Case 2: Monitor changed or gap threshold exceeded, end current burst and start new one
            if monitor_changed or not gap_ok:
                burst_id = self._find_burst_by_type(agg_type)
                if burst_id is not None:
                    self._end_burst(burst_id, last_event.timestamp, last_screenshot, last_event)

                screenshot = self._find_screenshot_before(event.timestamp)
                burst_id = self._start_burst(agg_type, event, screenshot)
                queue.clear()
                queue.append((event, screenshot))

            # Case 3: Total threshold exceeded (but gap OK and monitor same), split the burst
            elif not total_ok:
                queue_list = list(queue)
                mid_timestamp = (first_event.timestamp + last_event.timestamp) / 2

                first_half = [(e, s) for e, s in queue_list if e.timestamp <= mid_timestamp]
                second_half = [(e, s) for e, s in queue_list if e.timestamp > mid_timestamp]

                burst_id = self._find_burst_by_type(agg_type)
                if burst_id is not None and first_half:
                    self._end_burst(burst_id, first_half[-1][0].timestamp, first_half[-1][1], first_half[-1][0])

                if second_half:
                    updated_second_half = []
                    for i, (e, s) in enumerate(second_half):
                        if i == 0:
                            new_screenshot = self._find_screenshot_after(mid_timestamp)
                        else:
                            new_screenshot = s
                        updated_second_half.append((e, new_screenshot))

                    burst_id = self._start_burst(agg_type, second_half[0][0], updated_second_half[0][1])
                    queue.clear()
                    queue.extend(updated_second_half)

                    screenshot = self._find_screenshot_before(event.timestamp)
                    queue.append((event, screenshot))
                else:
                    screenshot = self._find_screenshot_before(event.timestamp)
                    burst_id = self._start_burst(agg_type, event, screenshot)
                    queue.clear()
                    queue.append((event, screenshot))

            # Case 4: Within thresholds and same monitor, just add to current burst
            else:
                screenshot = self._find_screenshot_before(event.timestamp)
                queue.append((event, screenshot))

    def _start_burst(self, event_type: str, event: InputEvent, screenshot: Optional[Any]) -> int:
        """
        Start a new burst and return its ID.

        Args:
            event_type: Type of event ('click', 'move', 'scroll', 'key')
            event: The triggering event
            screenshot: Cached screenshot for this event

        Returns:
            Burst ID
        """
        burst_id = self.next_burst_id
        self.next_burst_id += 1

        start_request = AggregationRequest(
            timestamp=event.timestamp,
            end_timestamp=None,
            reason=f"{event_type}_start",
            event_type=event_type,
            is_start=True,
            screenshot=screenshot,
            screenshot_path=None,
            screenshot_timestamp=screenshot.timestamp if screenshot else None,
            end_screenshot_timestamp=None,
            monitor=event.monitor,
            burst_id=burst_id
        )

        self.active_bursts[burst_id] = {
            'event_type': event_type,
            'start_time': event.timestamp,
            'start_screenshot': screenshot,
            'end_time': None,
            'end_screenshot': None,
            'start_request': start_request,
            'end_request': None
        }

        # Add start request to pending heap
        self._add_request_to_heap(start_request)

        return burst_id

    def _end_burst(self, burst_id: int, end_time: float, screenshot: Optional[Any], last_event: Optional[InputEvent]) -> None:
        """
        End a burst and add its end request to the heap.

        Args:
            burst_id: ID of the burst to end
            end_time: Timestamp when the burst ended
            screenshot: Cached screenshot for the end event
            last_event: The last event in the burst (for monitor info). May be None.
        """
        with self._lock:
            if burst_id not in self.active_bursts:
                return

            burst = self.active_bursts[burst_id]
            burst['end_time'] = end_time
            burst['end_screenshot'] = screenshot

            end_request = AggregationRequest(
                timestamp=end_time,
                end_timestamp=None,
                reason=f"{burst['event_type']}_end",
                event_type=burst['event_type'],
                is_start=False,
                screenshot=screenshot,
                screenshot_path=None,
                screenshot_timestamp=screenshot.timestamp if screenshot else None,
                end_screenshot_timestamp=None,
                monitor=(last_event.monitor if last_event is not None else None),
                burst_id=burst_id
            )

            burst['end_request'] = end_request

            # Add end request to pending heap
            self._add_request_to_heap(end_request)

            # remove from active_bursts
            del self.active_bursts[burst_id]

    def _add_request_to_heap(self, request: AggregationRequest) -> None:
        """Add a request to the pending heap, sorted by timestamp."""
        counter = next(self._pending_counter)
        heapq.heappush(self._pending_requests_heap, (request.timestamp, counter, request))

    def _find_burst_by_type(self, event_type: str) -> Optional[int]:
        """Find the most recent active burst of a given type."""
        for burst_id, burst in self.active_bursts.items():
            if burst['event_type'] == event_type:
                return burst_id
        return None

    def _find_screenshot_before(self, timestamp: float) -> Optional[Any]:
        """Find screenshot before timestamp (for start events)."""
        constants = constants_manager.get()
        candidates = self.image_queue.get_entries_before(
            timestamp, milliseconds=constants.PADDING_BEFORE
        )
        return candidates[-1]

    def _find_screenshot_after(self, timestamp: float) -> Optional[Any]:
        """Find screenshot after timestamp (for end events and split points)."""
        constants = constants_manager.get()
        candidates = self.image_queue.get_entries_after(
            timestamp, milliseconds=constants.PADDING_AFTER
        )
        return candidates[0]

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

                queue = self.aggregations[event_type]
                if queue:
                    last_event, last_screenshot = queue[-1]
                    time_since_last_event = current_time - last_event.timestamp

                    if time_since_last_event > config.gap_threshold:
                        burst_ids_to_end.append((burst_id, event_type, last_event.timestamp, last_screenshot, last_event))

            for burst_id, event_type, end_time, screenshot, last_event in burst_ids_to_end:
                if burst_id in self.active_bursts:
                    self._end_burst(burst_id, end_time, screenshot, last_event)
                    self.aggregations[event_type].clear()

    def _process_ready_requests(self) -> None:
        """
        Process requests that are old enough that the next request is certain.
        A request is ready when the next request started more than (max_threshold + safety_margin) ago.
        Always leaves at least one request in the heap (the most recent one).
        """
        with self._lock:
            # need at least two requests to compare current vs next
            if len(self._pending_requests_heap) < 2:
                return

            max_threshold = max(cfg.total_threshold for cfg in self.configs.values() if cfg)
            ready_threshold = max_threshold + self.safety_margin
            current_time = time.time()

            # Sort requests by timestamp
            sorted_items = sorted(self._pending_requests_heap)

            requests_to_emit = []
            requests_to_keep = []

            n = len(sorted_items)
            for i in range(n - 1):  # Always keep the last request
                _, _, current_req = sorted_items[i]
                next_timestamp, _, next_req = sorted_items[i + 1]

                time_since_next = current_time - next_timestamp

                # If enough time has passed since the next request, we can safely emit current
                if time_since_next > ready_threshold:
                    # Set end_timestamp and end_screenshot_timestamp to next request's values
                    current_req.end_timestamp = next_req.timestamp
                    current_req.end_screenshot_timestamp = next_req.screenshot_timestamp
                    requests_to_emit.append(current_req)
                else:
                    requests_to_keep.append((sorted_items[i]))

            # Always keep the last request
            requests_to_keep.append(sorted_items[-1])

            # Emit ready requests
            for req in requests_to_emit:
                if self._callback:
                    try:
                        self._callback(req)
                    except Exception as e:
                        print(f"Error in request callback: {e}")

            # Rebuild heap from requests to keep
            self._pending_requests_heap = requests_to_keep
            heapq.heapify(self._pending_requests_heap)

    def _poll_worker(self) -> None:
        """Worker thread that checks for stale bursts and processes ready requests."""
        while self._running:
            try:
                self._poll_stale_bursts()
                self._process_ready_requests()
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
                    last_event, last_screenshot = queue[-1]
                    self._end_burst(burst_id, last_event.timestamp, last_screenshot, last_event)
                else:
                    self._end_burst(burst_id, current_time, None, None)

            # Sort all remaining requests by timestamp
            sorted_items = sorted(self._pending_requests_heap)

            # Emit all requests with proper end_timestamp set
            for i in range(len(sorted_items)):
                _, _, current_req = sorted_items[i]

                if i < len(sorted_items) - 1:
                    next_timestamp, _, next_req = sorted_items[i + 1]
                    current_req.end_timestamp = next_req.timestamp
                    current_req.end_screenshot_timestamp = next_req.screenshot_timestamp

                if self._callback:
                    try:
                        self._callback(current_req)
                    except Exception as e:
                        print(f"Error in final request callback: {e}")

            # Clear heap
            self._pending_requests_heap.clear()

            # Clear aggregations
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
