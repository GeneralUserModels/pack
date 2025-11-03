import threading
import time
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


class PendingRequest:
    """Wrapper for requests that need screenshots resolved."""

    def __init__(self, request: AggregationRequest, needs_screenshot: bool = False):
        self.request = request
        self.needs_screenshot = needs_screenshot
        self.ready = not needs_screenshot


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

        # Single ordered list of pending requests
        self._pending_requests = []
        self._sequence_counter = 0

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

        # Buffer for ready requests (3 second delay before emission)
        self._ready_buffer = []

    def set_callback(self, callback: Callable[[AggregationRequest], None]) -> None:
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

            # Case 1: Start new burst (gap exceeded or empty queue)
            if not queue or not gap_ok:
                if queue:
                    self._end_burst(agg_type, last_event, last_screenshots)
                    queue.clear()

                self._start_burst(agg_type, event, screenshots)
                queue.append((event, screenshots))

            # Case 2: Monitor changed - create mid request (screenshot pending)
            elif monitor_changed:
                self._create_mid_request_monitor_switch(agg_type, event)
                queue.append((event, screenshots))

            # Case 3: Max length exceeded - split burst
            elif not total_ok:
                mid_idx = self._find_middle_event(queue)
                first_half = list(queue)[:mid_idx + 1]
                remaining_queue = list(queue)[mid_idx + 1:]

                mid_event, mid_screenshots = first_half[-1]
                self._create_mid_request_split(agg_type, mid_event, mid_screenshots)

                queue.clear()
                queue.extend(remaining_queue)
                queue.append((event, screenshots))

            # Case 4: Continue current burst
            else:
                queue.append((event, screenshots))

    def _find_middle_event(self, queue):
        """Find event closest to temporal middle of queue."""
        first_event, _ = queue[0]
        last_event, _ = queue[-1]
        timestamp_estimate = (first_event.timestamp + last_event.timestamp) / 2

        mid_idx = 0
        min_diff = float('inf')
        for idx, (e, s) in enumerate(queue):
            diff = abs(e.timestamp - timestamp_estimate)
            if diff < min_diff:
                min_diff = diff
                mid_idx = idx
        return mid_idx

    def _start_burst(self, agg_type: str, event: InputEvent, screenshot: Any) -> None:
        """Start a new burst."""
        self.next_burst_id += 1
        current_burst_id = self.next_burst_id
        self.active_bursts[current_burst_id] = {"event_type": agg_type}

        request = self._create_request(
            event=event,
            screenshot=screenshot,
            reason=Reason.STALE,
            state='start',
            burst_id=current_burst_id
        )
        self._add_request(request, needs_screenshot=False)

    def _end_burst(self, agg_type: str, event: InputEvent, screenshot: Any) -> None:
        """End a burst."""
        current_burst_id = self._get_burst_id_for_type(agg_type)

        # Get screenshot with padding after
        end_screenshot = self._collect_end_screenshot(event.timestamp)

        request = self._create_request(
            event=event,
            screenshot=end_screenshot,
            reason=Reason.STALE,
            state='end',
            burst_id=current_burst_id
        )
        self._add_request(request, needs_screenshot=False)

        if current_burst_id in self.active_bursts:
            del self.active_bursts[current_burst_id]

    def _create_mid_request_monitor_switch(self, agg_type: str, event: InputEvent) -> None:
        """Create mid request for monitor switch - screenshot will be resolved later."""
        current_burst_id = self._get_burst_id_for_type(agg_type)

        request = self._create_request(
            event=event,
            screenshot=None,  # Will be resolved later
            reason=Reason.MONITOR_SWITCH,
            state='mid',
            burst_id=current_burst_id,
            monitor_index=event.monitor_index
        )
        self._add_request(request, needs_screenshot=True)

    def _create_mid_request_split(self, agg_type: str, event: InputEvent, screenshot: Any) -> None:
        """Create mid request for burst split."""
        current_burst_id = self._get_burst_id_for_type(agg_type)

        request = self._create_request(
            event=event,
            screenshot=screenshot,
            reason=Reason.MAX_LENGTH_EXCEEDED,
            state='mid',
            burst_id=current_burst_id
        )
        self._add_request(request, needs_screenshot=False)

    def _create_request(
        self,
        event: InputEvent,
        screenshot: Any,
        reason: Reason,
        state: str,
        burst_id: int,
        monitor_index: Optional[int] = None
    ) -> AggregationRequest:
        """Create a request object."""
        event_type = self.event_type_mapping.get(event.event_type)
        reason_str = f"{event_type}_{state}_{reason}"

        effective_monitor_index = monitor_index if monitor_index is not None else event.monitor_index

        return AggregationRequest(
            timestamp=event.timestamp,
            end_timestamp=None,
            reason=reason_str,
            event_type=event_type,
            request_state=state,
            screenshot=screenshot,
            screenshot_path=None,
            screenshot_timestamp=screenshot.timestamp if screenshot else None,
            end_screenshot_timestamp=None,
            monitor=screenshot.monitor_dict if screenshot else None,
            monitor_index=effective_monitor_index,
            burst_id=burst_id,
            scale_factor=screenshot.scale_factor if screenshot else None
        )

    def _add_request(self, request: AggregationRequest, needs_screenshot: bool) -> None:
        """Add request to pending list in order."""
        pending = PendingRequest(request, needs_screenshot)
        self._pending_requests.append(pending)

    def _get_burst_id_for_type(self, agg_type: str) -> int:
        """Get the active burst ID for a given event type."""
        return next(
            (k for k, v in self.active_bursts.items() if v['event_type'] == agg_type),
            self.next_burst_id
        )

    def _collect_screenshots(self, timestamp: float) -> Any:
        """Get screenshot before timestamp."""
        constants = constants_manager.get()
        start_candidates = self.image_queue.get_entries_before(
            timestamp, milliseconds=constants.PADDING_BEFORE
        )
        return start_candidates[-1] if start_candidates else None

    def _collect_end_screenshot(self, timestamp: float) -> Any:
        """Get screenshot after timestamp with padding."""
        constants = constants_manager.get()
        exact_candidates = self.image_queue.get_entries_after(
            timestamp, milliseconds=constants.PADDING_AFTER
        )
        return exact_candidates[-1] if exact_candidates else None

    def _save_event_to_jsonl(self, event: InputEvent) -> None:
        if self.session_dir:
            try:
                with open(self.session_dir / "events.jsonl", "a") as f:
                    f.write(str(event.to_dict()) + "\n")
            except Exception as e:
                print(f"Error saving event to JSONL: {e}")

    def _poll_stale_bursts(self) -> None:
        """End bursts that have gone stale."""
        current_time = time.time()

        with self._lock:
            for burst_id, burst in list(self.active_bursts.items()):
                event_type = burst['event_type']
                config = self.configs[event_type]
                queue = self.aggregations[event_type]

                if queue:
                    last_event, last_screenshots = queue[-1]
                    time_since_last_event = current_time - last_event.timestamp

                    if time_since_last_event > config.gap_threshold:
                        self._end_burst(event_type, last_event, last_screenshots)
                        queue.clear()

    def _resolve_pending_screenshots(self) -> None:
        """Try to resolve screenshots for pending requests that need them."""
        with self._lock:
            for pending in self._pending_requests:
                if not pending.needs_screenshot or pending.ready:
                    continue

                request = pending.request

                # Try to get screenshot after the event timestamp
                screenshots = self.image_queue.get_entries_after(request.timestamp, milliseconds=0)

                if screenshots:
                    # Filter by monitor if specified
                    if request.monitor_index is not None:
                        matching = [s for s in screenshots if s.monitor_index == request.monitor_index]
                        chosen = matching[0] if matching else screenshots[0]
                    else:
                        chosen = screenshots[0]

                    request.screenshot = chosen
                    request.screenshot_timestamp = chosen.timestamp
                    request.monitor = chosen.monitor_dict
                    request.scale_factor = chosen.scale_factor
                    pending.ready = True

    def _link_requests_with_timestamps(self) -> None:
        """
        Link consecutive requests by setting end_timestamp based on next request's screenshot_timestamp.
        Move ready requests to buffer after 3 second delay.
        """
        with self._lock:
            current_time = time.time()
            constants = constants_manager.get()

            # First pass: resolve any missing screenshot timestamps
            for pending in self._pending_requests:
                req = pending.request
                if not req.screenshot_timestamp and req.screenshot:
                    req.screenshot_timestamp = req.screenshot.timestamp
                elif not req.screenshot_timestamp and req.screenshot_path:
                    req.screenshot_timestamp = float(str(Path(req.screenshot_path).name).split("_")[0])
                elif not req.screenshot_timestamp:
                    req.screenshot_timestamp = req.timestamp

            # Sort by screenshot timestamp to ensure correct ordering
            self._pending_requests.sort(key=lambda p: p.request.screenshot_timestamp or p.request.timestamp)

            requests_to_emit = []
            requests_to_keep = []

            n = len(self._pending_requests)

            for i in range(n):
                pending = self._pending_requests[i]

                # Skip if not ready
                if not pending.ready:
                    requests_to_keep.append(pending)
                    continue

                request = pending.request

                # Set end_timestamp from next ready request
                if i < n - 1:
                    # Find next ready request
                    next_ready_idx = None
                    for j in range(i + 1, n):
                        if self._pending_requests[j].ready:
                            next_ready_idx = j
                            break

                    if next_ready_idx is not None:
                        next_req = self._pending_requests[next_ready_idx].request
                        request.end_timestamp = next_req.timestamp
                        request.end_screenshot_timestamp = next_req.screenshot_timestamp
                        requests_to_emit.append(request)
                    else:
                        # No next ready request, keep this one
                        requests_to_keep.append(pending)
                else:
                    # Last request - only emit if all queues are empty
                    if not any(self.aggregations.values()):
                        request.end_timestamp = current_time
                        requests_to_emit.append(request)
                    else:
                        requests_to_keep.append(pending)

            # Move to ready buffer with timestamp
            for req in requests_to_emit:
                self._ready_buffer.append((req.screenshot_timestamp, req, current_time))

            # Update pending list
            self._pending_requests = requests_to_keep

            # Emit from ready buffer after 3 second delay
            remaining_buffer = []
            for screenshot_ts, req, add_time in self._ready_buffer:
                if current_time - add_time > 3.0:
                    if self._callback:
                        try:
                            self._callback(req)
                        except Exception as e:
                            print(f"Error in request callback: {e}")
                else:
                    remaining_buffer.append((screenshot_ts, req, add_time))

            self._ready_buffer = remaining_buffer

    def _poll_worker(self) -> None:
        """Worker thread."""
        while self._running:
            try:
                self._poll_stale_bursts()
                self._resolve_pending_screenshots()
                self._link_requests_with_timestamps()
                time.sleep(self.poll_interval)
            except Exception as e:
                import traceback
                print(f"Error in poll worker: {e}")
                print(traceback.format_exc())
                time.sleep(self.poll_interval)

    def process_all_remaining(self) -> None:
        """Process all remaining bursts and events at shutdown."""
        with self._lock:
            # End all active bursts
            for burst_id in list(self.active_bursts.keys()):
                burst = self.active_bursts[burst_id]
                queue = self.aggregations[burst['event_type']]
                if queue:
                    last_event, last_screenshots = queue[-1]
                    self._end_burst(burst['event_type'], last_event, last_screenshots)
                    queue.clear()

            # Resolve any pending screenshots
            self._resolve_pending_screenshots()

            # Sort all pending by screenshot timestamp
            for pending in self._pending_requests:
                req = pending.request
                if not req.screenshot_timestamp and req.screenshot:
                    req.screenshot_timestamp = req.screenshot.timestamp
                elif not req.screenshot_timestamp:
                    req.screenshot_timestamp = req.timestamp

            self._pending_requests.sort(key=lambda p: p.request.screenshot_timestamp or p.request.timestamp)

            # Link all remaining requests
            n = len(self._pending_requests)
            for i in range(n):
                request = self._pending_requests[i].request

                if i < n - 1:
                    next_req = self._pending_requests[i + 1].request
                    request.end_timestamp = next_req.timestamp
                    request.end_screenshot_timestamp = next_req.screenshot_timestamp
                else:
                    request.end_timestamp = time.time()

            # Emit all (bypass 3 second delay)
            for pending in self._pending_requests:
                if self._callback:
                    try:
                        self._callback(pending.request)
                    except Exception as e:
                        print(f"Error in request callback: {e}")

            # Emit anything remaining in buffer
            for _, req, _ in self._ready_buffer:
                if self._callback:
                    try:
                        self._callback(req)
                    except Exception as e:
                        print(f"Error in request callback: {e}")

            self._pending_requests.clear()
            self._ready_buffer.clear()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._poll_thread = threading.Thread(target=self._poll_worker, daemon=True)
            self._poll_thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
