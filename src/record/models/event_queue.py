import threading
import time
import heapq
import itertools
from pathlib import Path
from typing import Callable, Optional, Dict, Any, NamedTuple
from enum import StrEnum
from collections import deque
from record.models.event import InputEvent, EventType
from record.models.aggregation import AggregationConfig, AggregationRequest
from record.constants import constants_manager


class EventScreenshots(NamedTuple):
    """Container for the screenshot types captured when event is enqueued."""
    start: Optional[Any]  # Screenshot with padding before
    exact: Optional[Any]  # Screenshot with no padding


class Reason(StrEnum):
    STALE = "stale"
    MONITOR_SWITCH = "monitor_switch"
    MAX_LENGTH_EXCEEDED = "max_length_exceeded"


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
        Captures all three screenshot types alongside event in the aggregation queue.

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

            # Capture all three screenshot types for this event
            screenshots = self._collect_screenshots(event.timestamp)

            last_event, last_screenshots = queue[-1] if queue else (None, None)
            first_event, first_screenshots = queue[0] if queue else (None, None)

            monitor_changed = event.monitor_index != last_event.monitor_index if last_event else False
            gap_ok = (event.timestamp - last_event.timestamp) <= config.gap_threshold if last_event else True
            total_ok = (event.timestamp - first_event.timestamp) <= config.total_threshold if first_event else True

            # Case 1: Queue is empty or gap exceeded --> start new burst (clear queue)
            if not queue or not gap_ok:
                reason = Reason.STALE
                if queue:
                    self._create_burst_request(last_event, last_screenshots, reason, is_burst_end=True)
                    queue.clear()
                self._create_burst_request(event, screenshots, reason, is_burst_end=False)
                queue.append((event, screenshots))
            # Case 2: Monitor changed or total length exceeded --> start new burst
            elif monitor_changed or not total_ok:
                evt = event
                scr_shots = screenshots
                remaining_queue = []
                reason = Reason.MONITOR_SWITCH if monitor_changed else Reason.MAX_LENGTH_EXCEEDED
                if not monitor_changed:
                    timestamp_estimate = (first_event.timestamp + last_event.timestamp) / 2
                    first_half = [(e, s) for e, s in list(queue) if e.timestamp <= timestamp_estimate]
                    remaining_queue = [(e, s) for e, s in list(queue) if e.timestamp > timestamp_estimate]
                    evt, scr_shots = first_half[-1]
                self._create_burst_request(evt, scr_shots, reason, is_burst_end=False)
                queue.clear()
                queue.extend(remaining_queue)
            # Case 3: Continue current burst
            else:
                queue.append((event, screenshots))

    def _collect_screenshots(self, timestamp: float) -> EventScreenshots:
        constants = constants_manager.get()

        start_candidates = self.image_queue.get_entries_before(
            timestamp, milliseconds=constants.PADDING_BEFORE
        )
        start_screenshot = start_candidates[-1] if start_candidates else None

        exact_candidates = self.image_queue.get_entries_before(
            timestamp, milliseconds=0
        )
        exact_screenshot = exact_candidates[-1] if exact_candidates else None

        if not start_screenshot:
            print(f"Warning: No start screenshot found for timestamp {timestamp}")
        if not exact_screenshot:
            print(f"Warning: No exact screenshot found for timestamp {timestamp}")

        return EventScreenshots(start=start_screenshot, exact=exact_screenshot)

    def _create_burst_request(
        self,
        event: InputEvent,
        screenshots: EventScreenshots,
        reason: str,
        is_burst_end: bool
    ) -> AggregationRequest:

        screenshot = self._resolve_screenshot(screenshots, event, reason, is_burst_end)
        event_type = self.event_type_mapping.get(event.event_type)
        request_state = 'end' if is_burst_end else 'start' if reason == Reason.STALE else 'mid'
        reason_str = f"{event_type}_{request_state}_{reason}"

        if not is_burst_end and reason == Reason.STALE:
            self.next_burst_id += 1
            self.active_bursts[self.next_burst_id] = {
                "event_type": event_type
            }

        request = AggregationRequest(
            timestamp=event.timestamp,
            end_timestamp=None,
            reason=reason_str,
            event_type=self.event_type_mapping.get(event.event_type),
            request_state=request_state,
            screenshot=screenshot,
            screenshot_path=None,
            screenshot_timestamp=screenshot.timestamp if screenshot else None,
            end_screenshot_timestamp=None,
            monitor=screenshot.monitor_dict if screenshot else None,
            burst_id=self.next_burst_id,
            scale_factor=screenshot.scale_factor if screenshot else None
        )

        self._add_request_to_heap(request)
        if is_burst_end:
            id = [k for k, v in self.active_bursts.items() if v['event_type'] == event_type][0]
            del self.active_bursts[id]

    def _resolve_screenshot(self, screenshots: EventScreenshots, event: InputEvent, reason: str, is_burst_end: bool) -> Optional[Any]:
        if is_burst_end:
            exact_candidates = self.image_queue.get_entries_after(
                event.timestamp, milliseconds=constants_manager.get().PADDING_AFTER
            )
            print(f"Selecting end screenshot with padding after for reason: {reason}")
            return exact_candidates[-1] if exact_candidates else None
        elif reason == Reason.STALE:
            print(f"Selecting start screenshot with padding before for reason: {reason}")
            return screenshots.start
        else:
            print(f"Selecting exact screenshot for reason: {reason}")
            return screenshots.exact

    def _add_request_to_heap(self, request: AggregationRequest) -> None:
        """Add a request to the pending heap, sorted by timestamp."""
        counter = next(self._pending_counter)
        heapq.heappush(self._pending_requests_heap, (request.timestamp, counter, request))

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
                    last_event, last_screenshots = queue[-1]
                    time_since_last_event = current_time - last_event.timestamp

                    if time_since_last_event > config.gap_threshold:
                        burst_ids_to_end.append((burst_id, event_type, last_event.timestamp, last_screenshots, last_event))

            for burst_id, event_type, end_time, screenshots, last_event in burst_ids_to_end:
                if burst_id in self.active_bursts:
                    self._create_burst_request(last_event, screenshots, reason=Reason.STALE, is_burst_end=True)
                    self.aggregations[event_type].clear()

    def _process_ready_requests(self) -> None:
        """
        Process requests that are old enough that the next request is certain.
        A request is ready when the next request started more than (max_threshold + safety_margin) ago.
        Always leaves at least one request in the heap (the most recent one).
        """
        with self._lock:
            if len(self._pending_requests_heap) < 2:
                return

            sorted_items = sorted(self._pending_requests_heap)

            requests_to_emit = []
            requests_to_keep = []

            n = len(sorted_items)
            for i in range(n - 1):
                current_timestamp, _, current_req = sorted_items[i]
                next_timestamp, _, next_req = sorted_items[i + 1]

                next_ensured = True
                for request in self._pending_requests_heap:
                    if request[2].request_state == "end" or request[2].timestamp >= current_timestamp:
                        continue
                    config = self.configs[request[2].event_type]
                    if current_timestamp < request[2].timestamp + config.total_threshold < next_timestamp:
                        next_ensured = False
                        break

                if next_ensured:
                    current_req.end_timestamp = next_req.timestamp
                    current_req.end_screenshot_timestamp = next_req.screenshot_timestamp
                    requests_to_emit.append(current_req)
                else:
                    requests_to_keep.append((sorted_items[i]))

            if not any([v for v in self.aggregations.values()]):
                requests_to_emit.append(sorted_items[-1][2])
            else:
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
            burst_ids_to_end = list(self.active_bursts.keys())

            for burst_id in burst_ids_to_end:
                burst = self.active_bursts[burst_id]
                queue = self.aggregations[burst['event_type']]
                if queue:
                    last_event, last_screenshots = queue[-1]
                    self._create_burst_request(last_event, last_screenshots, reason=Reason.STALE, is_burst_end=True)
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
