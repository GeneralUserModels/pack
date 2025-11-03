import threading
import json
from collections import deque
from record.models.aggregation import AggregationRequest, ProcessedAggregation


class AggregationWorker:
    """
    Worker that processes aggregation requests and collects events within burst windows.
    Screenshots are now pre-fetched and stored in the AggregationRequest by EventQueue.
    """

    def __init__(self, event_queue, save_worker):
        """
        Initialize the aggregation worker.

        Args:
            event_queue: EventQueue instance (to access all_events)
            save_worker: SaveWorker instance (to save screenshots and aggregations)
        """
        self.event_queue = event_queue
        self.save_worker = save_worker
        self._lock = threading.RLock()

        self.aggregations_file = save_worker.session_dir / "aggregations.jsonl"
        self.processed_requests = set()
        self.processed_event_timestamps = set()

    def process_aggregation(self, request: AggregationRequest) -> ProcessedAggregation:
        """
        Process a single aggregation request (start or end of a burst).

        Args:
            request: AggregationRequest object with pre-fetched screenshot

        Returns:
            ProcessedAggregation object with matched screenshot and events
        """
        with self._lock:
            request_key = (request.timestamp, request.reason)
            if request_key in self.processed_requests:
                return ProcessedAggregation(
                    request=request,
                    events=[]
                )

            self.processed_requests.add(request_key)

            if request.screenshot is not None:
                request.screenshot_path = self.save_worker.save_screenshot(
                    request.screenshot, force_save=True, save_reason=request.reason
                )

            start_ts = request.screenshot_timestamp

            if request.end_screenshot_timestamp is not None:
                end_ts = request.end_screenshot_timestamp
            else:
                end_ts = float('inf')

            events = self._get_events_between(start_ts, end_ts)

            processed_agg = ProcessedAggregation(
                request=request,
                events=events
            )

            self._save_aggregation_to_jsonl(processed_agg)

            return processed_agg

    def _get_events_between(self, start_screenshot_timestamp: float, end_screenshot_timestamp: float) -> list:
        """
        Get all events (of ALL types) between two screenshot timestamps.
        Events are assigned to bursts based on which burst they fall into.
        An event belongs to a burst if: start_screenshot_timestamp <= event.timestamp < end_screenshot_timestamp

        Args:
            start_screenshot_timestamp: Start screenshot timestamp (inclusive)
            end_screenshot_timestamp: End screenshot timestamp (exclusive). If inf, include all remaining events.

        Returns:
            List of serialized events
        """
        events_to_process = []
        events_to_keep = deque()

        with self.event_queue._lock:
            for e in self.event_queue.all_events:
                # Only process if not already processed (avoid duplicates)
                event_key = (e.timestamp, e.event_type, id(e))

                if start_screenshot_timestamp <= e.timestamp < end_screenshot_timestamp:
                    if event_key not in self.processed_event_timestamps:
                        events_to_process.append(e)
                        self.processed_event_timestamps.add(event_key)
                    # Don't keep it in the queue if it's in range
                else:
                    events_to_keep.append(e)

            self.event_queue.all_events = events_to_keep

        serialized = [e.to_dict() for e in events_to_process]

        return serialized

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
                'request_state': aggregation.request.request_state,
                'screenshot_path': aggregation.request.screenshot_path,
                'screenshot_timestamp': aggregation.request.screenshot_timestamp,
                'end_screenshot_timestamp': aggregation.request.end_screenshot_timestamp,
                'num_events': len(aggregation.events),
                'events': aggregation.events,
                'cursor_position': aggregation.events[0].get('cursor_position') if aggregation.events else None,
                'monitor': aggregation.request.monitor,
                'burst_id': aggregation.request.burst_id,
                'scale_factor': aggregation.request.scale_factor
            }

            with open(self.aggregations_file, 'a') as f:
                json.dump(data, f)
                f.write('\n')

        except Exception as e:
            print(f"Error saving aggregation to JSONL: {e}")

    def validate_events_processed(self):
        """
        Check for any unprocessed events that have fallen through the cracks.
        This should be called at shutdown to ensure no events were lost.
        """
        with self._lock:
            with self.event_queue._lock:
                if self.event_queue.all_events:
                    orphaned_count = len(self.event_queue.all_events)
                    print(f"\n⚠️  WARNING: {orphaned_count} orphaned events found in all_events queue!")
                    print("These events were not captured in any aggregation:")

                    for e in list(self.event_queue.all_events)[:10]:  # Show first 10
                        print(f"  - {e.event_type} at {e.timestamp:.3f}")

                    if orphaned_count > 10:
                        print(f"  ... and {orphaned_count - 10} more")

                    return False
                return True
