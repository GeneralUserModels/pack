import threading
import json
from typing import Optional
from collections import deque
from record.models.aggregation import AggregationRequest, ProcessedAggregation
from record.models import ImageQueue
from record.workers.save import SaveWorker
from record.constants import Constants


class AggregationWorker:
    """
    Worker that processes aggregation requests by matching them with screenshots
    and collecting all events within the burst window (including other event types).
    Events are always mapped to the latest (most recent) burst they fall within.
    """

    def __init__(self, event_queue, image_queue: ImageQueue, save_worker: SaveWorker):
        """
        Initialize the aggregation worker.

        Args:
            event_queue: EventQueue instance (to access all_events)
            image_queue: ImageQueue instance (to find screenshots)
            save_worker: SaveWorker instance (to save screenshots and aggregations)
        """
        self.event_queue = event_queue
        self.image_queue = image_queue
        self.save_worker = save_worker
        self._lock = threading.RLock()

        self.aggregations_file = save_worker.session_dir / "aggregations.jsonl"

        # Track processed bursts to avoid double-processing
        self.processed_requests = set()

    def process_aggregation(self, request: AggregationRequest) -> ProcessedAggregation:
        """
        Process a single aggregation request (start or end of a burst).

        Args:
            request: AggregationRequest object with timestamp and is_start flag

        Returns:
            ProcessedAggregation object with matched screenshot and events
        """
        with self._lock:
            # Skip if already processed
            request_key = (request.timestamp, request.reason)
            if request_key in self.processed_requests:
                return ProcessedAggregation(
                    request=request,
                    screenshot=None,
                    screenshot_path=None,
                    events=[]
                )

            self.processed_requests.add(request_key)

            # For start requests: screenshot ~50ms BEFORE
            # For end requests: screenshot ~50ms AFTER
            screenshot = self._find_screenshot(request.timestamp, request.is_start)

            screenshot_path = None
            if screenshot is not None:
                screenshot_path = self.save_worker.save_screenshot(
                    screenshot, force_save=True, save_reason=request.reason
                )

            # Get all events that fall within this burst
            # Start request: from this timestamp to end_timestamp (next burst)
            # End request: uses end_timestamp from when it was created
            if request.end_timestamp is None:
                request.end_timestamp = float('inf')

            events = self._get_events_between(request.timestamp, request.end_timestamp)

            processed_agg = ProcessedAggregation(
                request=request,
                screenshot=screenshot,
                screenshot_path=screenshot_path,
                events=events
            )

            self._save_aggregation_to_jsonl(processed_agg)

            return processed_agg

    def _find_screenshot(self, timestamp: float, is_start: bool) -> Optional[any]:
        """
        Find the best matching screenshot for the given timestamp.

        Args:
            timestamp: Target timestamp
            is_start: If True, look for screenshot ~50ms before; if False, ~50ms after

        Returns:
            Screenshot object or None if not found
        """
        if is_start:
            # For start events, get screenshot before (before the burst begins)
            candidates = self.image_queue.get_entries_before(
                timestamp, milliseconds=Constants.PADDING_BEFORE
            )
            return candidates[-1] if candidates else None
        else:
            # For end events, get screenshot after (after the burst ends)
            candidates = self.image_queue.get_entries_after(
                timestamp, milliseconds=Constants.PADDING_AFTER
            )
            return candidates[0] if candidates else None

    def _get_events_between(self, start_timestamp: float, end_timestamp: float) -> list:
        """
        Get all events (of ALL types) between two timestamps.
        Events are assigned to bursts based on which burst they fall into.
        An event belongs to a burst if: start_timestamp <= event.timestamp < end_timestamp

        Args:
            start_timestamp: Start time (inclusive)
            end_timestamp: End time (exclusive). If inf, include all remaining events.

        Returns:
            List of serialized events
        """
        events_to_process = []
        events_to_keep = deque()

        with self.event_queue._lock:
            for e in self.event_queue.all_events:
                if start_timestamp <= e.timestamp < end_timestamp:
                    events_to_process.append(e)
                else:
                    events_to_keep.append(e)

            # Update the queue to only contain events outside the range
            self.event_queue.all_events = events_to_keep

        # Serialize the processed events
        serialized = [self._serialize_event(e) for e in events_to_process]

        return serialized

    def _serialize_event(self, event) -> dict:
        """Serialize an InputEvent to a dictionary."""
        event_dict = {
            'timestamp': event.timestamp,
            'event_type': event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
        }

        # Add event-specific data
        if hasattr(event, 'x') and hasattr(event, 'y'):
            event_dict['x'] = event.x
            event_dict['y'] = event.y
        if hasattr(event, 'button'):
            event_dict['button'] = str(event.button)
        if hasattr(event, 'dx') and hasattr(event, 'dy'):
            event_dict['dx'] = event.dx
            event_dict['dy'] = event.dy
        if hasattr(event, 'key'):
            event_dict['key'] = str(event.key)
        if hasattr(event, 'char'):
            event_dict['char'] = event.char

        return event_dict

    def _save_aggregation_to_jsonl(self, aggregation: ProcessedAggregation):
        """
        Save a processed aggregation to JSONL file.

        Args:
            aggregation: ProcessedAggregation object to save
        """
        try:
            data = {
                'timestamp': aggregation.request.timestamp,
                'reason': aggregation.request.reason,
                'event_type': aggregation.request.event_type,
                'is_start': aggregation.request.is_start,
                'screenshot_path': aggregation.screenshot_path,
                'screenshot_timestamp': aggregation.screenshot.timestamp if aggregation.screenshot else None,
                'num_events': len(aggregation.events),
                'events': aggregation.events
            }

            with open(self.aggregations_file, 'a') as f:
                json.dump(data, f)
                f.write('\n')

        except Exception as e:
            print(f"Error saving aggregation to JSONL: {e}")

    def cleanup_old_events(self, oldest_timestamp: float):
        """
        Clean up events older than the given timestamp from all_events queue.
        Note: This is now less critical since events are removed as they're aggregated.

        Args:
            oldest_timestamp: Remove events older than this timestamp
        """
        with self._lock:
            with self.event_queue._lock:
                self.event_queue.all_events = deque([
                    e for e in self.event_queue.all_events
                    if e.timestamp >= oldest_timestamp
                ])
