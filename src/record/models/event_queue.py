import threading
import time
import heapq
import itertools
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from enum import StrEnum
from collections import deque
from record.models.event import InputEvent, EventType
from record.models.aggregation import AggregationConfig, AggregationRequest
from record.models.image_queue import ImageQueue
from record.constants import constants_manager


class Reason(StrEnum):
    STALE = "stale"
    MONITOR_SWITCH = "monitor_switch"
    MAX_LENGTH_EXCEEDED = "max_length_exceeded"


class EventQueue:

    def __init__(
        self,
        image_queue: ImageQueue,
        click_config: Optional[AggregationConfig] = None,
        move_config: Optional[AggregationConfig] = None,
        scroll_config: Optional[AggregationConfig] = None,
        key_config: Optional[AggregationConfig] = None,
        poll_interval: float = 1.0,
        session_dir: Path = None,
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
        """
        self.image_queue = image_queue
        self.session_dir = session_dir
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
        with self._lock:
            self.all_events.append(event)
            self._save_event_to_jsonl(event)

            agg_type = self.event_type_mapping.get(event.event_type)
            if agg_type is None:
                return

            queue = self.aggregations[agg_type]
            config = self.configs[agg_type]
            screenshots = self._collect_screenshots(event.timestamp)

            last_event, last_screenshots = queue[-1] if queue else (None, None)
            first_event, first_screenshots = queue[0] if queue else (None, None)

            monitor_changed = event.monitor_index != last_event.monitor_index if last_event else False
            gap_ok = (event.timestamp - last_event.timestamp) <= config.gap_threshold if last_event else True
            total_ok = (event.timestamp - first_event.timestamp) <= config.total_threshold if first_event else True

            # Case 1: Queue is empty or gap exceeded - end previous burst and start new
            if not queue or not gap_ok:
                if queue:
                    # End the previous burst
                    self._create_burst_request(last_event, last_screenshots, Reason.STALE, is_burst_end=True)
                    queue.clear()

                # Start new burst
                self._create_burst_request(event, screenshots, Reason.STALE, is_burst_end=False)
                queue.append((event, screenshots))

            # Case 2: Monitor changed - create mid request with NEW monitor screenshot
            elif monitor_changed:
                self._create_burst_request(
                    event,
                    None,
                    Reason.MONITOR_SWITCH,
                    is_burst_end=False,
                    monitor_index=event.monitor_index
                )

                queue.append((event, screenshots))

            # Case 3: Max length exceeded - split burst with mid request
            elif not total_ok:
                timestamp_estimate = (first_event.timestamp + last_event.timestamp) / 2

                # Find the event closest to the middle timestamp
                mid_idx = 0
                min_diff = float('inf')
                for idx, (e, s) in enumerate(queue):
                    diff = abs(e.timestamp - timestamp_estimate)
                    if diff < min_diff:
                        min_diff = diff
                        mid_idx = idx

                # Split at the middle event
                first_half = list(queue)[:mid_idx + 1]
                remaining_queue = list(queue)[mid_idx + 1:]

                # Use the screenshot from the middle event for the mid request
                mid_event, mid_screenshots = first_half[-1]
                self._create_burst_request(mid_event, mid_screenshots, Reason.MAX_LENGTH_EXCEEDED, is_burst_end=False)

                # Continue with remaining events in same burst
                queue.clear()
                queue.extend(remaining_queue)
                queue.append((event, screenshots))

            # Case 4: Continue current burst normally
            else:
                queue.append((event, screenshots))

    def _collect_screenshots(self, timestamp: float) -> Any:
        constants = constants_manager.get()

        start_candidates = self.image_queue.get_entries_before(
            timestamp, milliseconds=constants.PADDING_BEFORE
        )
        start_screenshot = start_candidates[-1] if start_candidates else None

        if not start_screenshot:
            print(f"Warning: No start screenshot found for timestamp {timestamp}")
        return start_screenshot

    def _create_burst_request(
        self,
        event: InputEvent,
        screenshot: Any,
        reason: Reason,
        is_burst_end: bool,
        monitor_index: Optional[int] = None
    ) -> AggregationRequest:

        screenshot = self._resolve_screenshot(screenshot, event, reason, is_burst_end)
        event_type = self.event_type_mapping.get(event.event_type)
        request_state = 'end' if is_burst_end else 'start' if reason == Reason.STALE else 'mid'
        reason_str = f"{event_type}_{request_state}_{reason}"

        if not is_burst_end and reason == Reason.STALE:
            self.next_burst_id += 1
            current_burst_id = self.next_burst_id
            self.active_bursts[current_burst_id] = {
                "event_type": event_type
            }
        else:
            current_burst_id = next(
                (k for k, v in self.active_bursts.items() if v['event_type'] == event_type),
                self.next_burst_id  # fallback
            )

        effective_monitor_index = monitor_index if monitor_index is not None else event.monitor_index

        request = AggregationRequest(
            timestamp=event.timestamp,
            end_timestamp=None,
            reason=reason_str,
            event_type=event_type,
            request_state=request_state,
            screenshot=screenshot,
            screenshot_path=None,
            screenshot_timestamp=screenshot.timestamp if screenshot else None,
            end_screenshot_timestamp=None,
            monitor=screenshot.monitor_dict if screenshot else None,
            monitor_index=effective_monitor_index,
            burst_id=current_burst_id,
            scale_factor=screenshot.scale_factor if screenshot else None
        )

        self._add_request_to_heap(request)

        if is_burst_end and current_burst_id in self.active_bursts:
            del self.active_bursts[current_burst_id]

        return request

    def _resolve_screenshot(self, screenshot: Any, event: InputEvent, reason: Reason, is_burst_end: bool) -> Optional[Any]:
        if is_burst_end:
            # End of burst: use screenshot with padding after
            exact_candidates = self.image_queue.get_entries_after(
                event.timestamp, milliseconds=constants_manager.get().PADDING_AFTER
            )
            return exact_candidates[-1] if exact_candidates else None
        elif reason in (Reason.STALE, Reason.MONITOR_SWITCH, Reason.MAX_LENGTH_EXCEEDED):
            return screenshot
        return screenshot

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
        with self._lock:
            if len(self._pending_requests_heap) < 2:
                return

            constants = constants_manager.get()
            grace = 0.1

            sorted_items = sorted(self._pending_requests_heap)

            requests_to_emit = []
            requests_to_keep = []

            n = len(sorted_items)
            for i in range(n - 1):
                current_timestamp, _, current_req = sorted_items[i]
                next_timestamp, _, next_req = sorted_items[i + 1]

                if next_req.request_state == "mid" and next_req.screenshot is None:
                    screenshots = self.image_queue.get_entries_after(next_req.timestamp, milliseconds=0)
                    if screenshots:
                        if next_req.monitor_index is not None:
                            chosen = next((s for s in screenshots if s.monitor_index == next_req.monitor_index), screenshots[0])
                        else:
                            chosen = screenshots[0]

                        next_req.screenshot = chosen
                        next_req.screenshot_timestamp = chosen.timestamp
                        next_req.monitor = chosen.monitor_dict
                        next_req.scale_factor = chosen.scale_factor
                    else:
                        if time.time() - next_req.timestamp < (constants.PADDING_AFTER + grace):
                            requests_to_keep.append(sorted_items[i])
                            continue

                next_ensured = True
                for request in self._pending_requests_heap:
                    r_req = request[2]
                    if r_req.request_state == "end" or r_req.timestamp >= current_timestamp:
                        continue
                    config = self.configs[r_req.event_type]
                    if current_timestamp < r_req.timestamp + config.total_threshold < next_timestamp:
                        next_ensured = False
                        break

                if next_ensured:
                    current_req.end_timestamp = next_req.timestamp
                    current_req.end_screenshot_timestamp = getattr(next_req, "screenshot_timestamp", None)
                    requests_to_emit.append(current_req)
                else:
                    requests_to_keep.append(sorted_items[i])

            if not any([v for v in self.aggregations.values()]):
                last_req = sorted_items[-1][2]
                last_req.end_timestamp = time.time()
                requests_to_emit.append(last_req)
            else:
                requests_to_keep.append(sorted_items[-1])

            # Emit ready requests
            for req in requests_to_emit:
                if req.request_state == "mid" and req.screenshot is None:
                    screenshots = self.image_queue.get_entries_after(req.timestamp, milliseconds=0)
                    if screenshots:
                        if req.monitor_index is not None:
                            chosen = next((s for s in screenshots if s.monitor_index == req.monitor_index), screenshots[0])
                        else:
                            chosen = screenshots[0]
                        req.screenshot = chosen
                        req.screenshot_timestamp = chosen.timestamp
                        req.monitor = chosen.monitor_dict
                        req.scale_factor = chosen.scale_factor

                if self._callback:
                    try:
                        self._callback(req)
                    except Exception as e:
                        print(f"Error in request callback: {e}")

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
