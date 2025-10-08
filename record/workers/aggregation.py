import threading
import json
from typing import List, Optional
from collections import deque
from record.models.aggregation import AggregationRequest, ProcessedAggregation
from record.models import ImageQueue
from record.workers.save import SaveWorker


class AggregationWorker:
    """
    Worker that processes aggregation requests by matching them with screenshots
    and filling in the events between consecutive aggregation points.
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

    def process_aggregations(self, requests: List[AggregationRequest]) -> List[ProcessedAggregation]:
        """
        Process a batch of aggregation requests.

        Args:
            requests: List of AggregationRequest objects (already sorted by timestamp)

        Returns:
            List of ProcessedAggregation objects with matched screenshots and events
        """
        with self._lock:
            processed = []

            for i, req in enumerate(requests):
                screenshot = self._find_screenshot(req.timestamp, req.is_start)

                screenshot_path = None
                if screenshot is not None:
                    screenshot_path = self.save_worker.save_screenshot(screenshot, force_save=True, save_reason=req.reason)

                next_timestamp = (
                    requests[i + 1].timestamp
                    if i + 1 < len(requests)
                    else self.event_queue.safe_aggregation_time
                )

                events = self._get_events_between(req.timestamp, next_timestamp)

                processed_agg = ProcessedAggregation(
                    request=req,
                    screenshot=screenshot,
                    screenshot_path=screenshot_path,
                    events=events
                )

                processed.append(processed_agg)

                self._save_aggregation_to_jsonl(processed_agg)

            return processed

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
            candidates = self.image_queue.get_entries_before(timestamp, milliseconds=75)

            if not candidates:
                candidates = self.image_queue.get_entries_before(timestamp, milliseconds=200)

            return candidates[-1] if candidates else None
        else:
            candidates = self.image_queue.get_entries_after(timestamp, milliseconds=75)

            if not candidates:
                candidates = self.image_queue.get_entries_after(timestamp, milliseconds=200)

            return candidates[0] if candidates else None

    def _get_events_between(self, start_timestamp: float, end_timestamp: float) -> List[dict]:
        """
        Get all events between two timestamps from the all_events queue.

        Args:
            start_timestamp: Start time (inclusive)
            end_timestamp: End time (exclusive)

        Returns:
            List of serialized events
        """
        events = [
            self._serialize_event(e)
            for e in self.event_queue.all_events
            if start_timestamp <= e.timestamp < end_timestamp
        ]

        return events

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

        Args:
            oldest_timestamp: Remove events older than this timestamp
        """
        with self._lock:
            self.event_queue.all_events = deque([
                e for e in self.event_queue.all_events
                if e.timestamp >= oldest_timestamp
            ])
