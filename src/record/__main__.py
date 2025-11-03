import argparse
import time
import signal
import sys
import threading
from pathlib import Path
from datetime import datetime
from pynput import mouse, keyboard

from record.models import ImageQueue, AggregationConfig, EventQueue
from record.workers import SaveWorker, AggregationWorker
from record.handlers import InputEventHandler, ScreenshotHandler
from record.monitor import RealtimeVisualizer, plot_summary_stats
from record.constants import constants_manager


class ScreenRecorder:

    def __init__(
        self,
        fps: int = 30,
        buffer_seconds: int = 12,
        buffer_all: bool = False,
        monitor: bool = False,
        max_res: tuple[int, int] = None,
        accessibility: bool = False,
        compression_quality: int = 70
    ):
        """
        Initialize the screen recorder.

        Args:
            fps: Frames per second to capture
            buffer_seconds: Number of seconds to keep in buffer
            buffer_all: If True, save all screenshots to disk
            monitor: If True, enable real-time monitoring
            accessibility: If True, enable accessibility info capture
        """
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.buffer_all = buffer_all
        self.max_res = max_res
        constants = constants_manager.get()

        self.image_buffer_size = fps * buffer_seconds
        self.event_buffer_size = fps * buffer_seconds * 30

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(__file__).parent.parent.parent / "logs" / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        print(f"Session directory: {self.session_dir}")

        self.image_queue = ImageQueue(max_length=self.image_buffer_size)

        self.input_event_queue = EventQueue(
            image_queue=self.image_queue,
            click_config=AggregationConfig(
                gap_threshold=constants.CLICK_GAP_THRESHOLD,
                total_threshold=constants.CLICK_TOTAL_THRESHOLD
            ),
            move_config=AggregationConfig(
                gap_threshold=constants.MOVE_GAP_THRESHOLD,
                total_threshold=constants.MOVE_TOTAL_THRESHOLD
            ),
            scroll_config=AggregationConfig(
                gap_threshold=constants.SCROLL_GAP_THRESHOLD,
                total_threshold=constants.SCROLL_TOTAL_THRESHOLD
            ),
            key_config=AggregationConfig(
                gap_threshold=constants.KEY_GAP_THRESHOLD,
                total_threshold=constants.KEY_TOTAL_THRESHOLD
            ),
            poll_interval=1.0,
            session_dir=self.session_dir
        )

        self.save_worker = SaveWorker(self.session_dir, buffer_all, compression_quality=compression_quality)

        self.aggregation_worker = AggregationWorker(
            event_queue=self.input_event_queue,
            save_worker=self.save_worker,
        )

        self.input_event_queue.set_callback(self._on_aggregation_request)
        self.image_queue.add_callback(self._on_new_image)

        self.screenshot_manager = ScreenshotHandler(
            image_queue=self.image_queue,
            fps=self.fps,
            max_res=self.max_res
        )

        self.input_handler = InputEventHandler(
            self.input_event_queue,
            accessibility=accessibility
        )
        self.mouse_listener = None
        self.keyboard_listener = None

        self.running = False
        self.processed_aggregations = 0

        self.monitor_thread = None
        if monitor:
            self._setup_monitor()

    def _on_aggregation_request(self, request):
        """Callback when a single aggregation request is ready for processing."""
        if not request:
            return

        processed = self.aggregation_worker.process_aggregation(request)
        if self.processed_aggregations == 0:
            print("-------------------------------------------------------------------")
            print(">>>>                    Aggregation Summary                    <<<<")
            print(f">>>> Session Directory: {str(self.session_dir.name):37s} <<<<")
            print("-------------------------------------------------------------------")
            print("Screenshot | # Events |     Timestamp     | Capture Reason ")
            print("-------------------------------------------------------------------")

        screenshot_status = "✓" if processed.screenshot else "✗"
        print(f"     {screenshot_status}     | {str(len(processed.events)):8s} |"
              f"{str(processed.request.timestamp):<18} | {processed.request.reason}")

        self.processed_aggregations += 1

    def _on_new_image(self, image):
        """Callback for saving all buffer images (only used if buffer_all)."""
        if self.buffer_all:
            self.save_worker.save_buffer_image(image)

    def _setup_monitor(self):
        """Set up real-time monitoring thread."""
        self.monitor_thread = threading.Thread(target=self._run_monitor, daemon=True)
        self.monitor_thread.start()

    def _run_monitor(self):
        """Run the real-time visualizer."""
        import sys
        import threading

        # On macOS, matplotlib GUI must run on main thread
        # Since this is running in a daemon thread, use non-interactive backend
        if sys.platform == 'darwin' and threading.current_thread() != threading.main_thread():
            import matplotlib
            matplotlib.use('Agg')
            print("Warning: Running monitor in thread on macOS. Using non-interactive backend.")
            print("The visualization window will not be displayed. Consider running without --monitor flag.")

        events_path = self.session_dir / "events.jsonl"
        aggr_path = self.session_dir / "aggregations.jsonl"
        rv = RealtimeVisualizer(events_path, aggr_path, refresh_hz=16, window_s=30.0)
        rv.run()

    def start(self):
        """Start recording."""
        if self.running:
            print("Recorder already running")
            return

        self.running = True
        print(f"Starting screen recorder at {self.fps} FPS")
        print(f"Buffer: {self.buffer_seconds} seconds ({self.image_buffer_size} frames)")
        print("Input event aggregation: ENABLED (polling worker)")
        print(f"  - Save all images: {self.buffer_all}")

        self.screenshot_manager.start()
        self.input_event_queue.start()

        # delay ensuring screenshots exist for first event
        constants = constants_manager.get()
        initial_delay = constants.PADDING_BEFORE / 1000.0
        time.sleep(initial_delay)

        self.mouse_listener = mouse.Listener(
            on_move=self.input_handler.on_move,
            on_click=self.input_handler.on_click,
            on_scroll=self.input_handler.on_scroll
        )
        self.mouse_listener.start()

        self.keyboard_listener = keyboard.Listener(
            on_press=self.input_handler.on_press,
            on_release=self.input_handler.on_release
        )
        self.keyboard_listener.start()

        print("-------------------------------------------------------------------")
        print("Recorder started. Press Ctrl+C to stop.")
        print("-------------------------------------------------------------------")

    def stop(self):
        """Stop recording and process remaining events."""
        if not self.running:
            return

        print("-------------------------------------------------------------------")
        print(">>>>                    Stopping Recorder                      <<<<")
        print(">>>>             Cleaning Up Remaining Processes...            <<<<")
        print("-------------------------------------------------------------------")

        self.running = False

        # Stop listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        self.screenshot_manager.stop()
        self.input_event_queue.stop()

        self.input_event_queue.process_all_remaining()

        time.sleep(2)

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=0.1)

        print(f"\nSession saved to: {self.session_dir}")
        print(f"Aggregations saved to: {self.aggregation_worker.aggregations_file}")
        print(f"Total aggregations processed: {self.processed_aggregations}")

        all_valid = self.aggregation_worker.validate_events_processed()
        if all_valid:
            print("✓ All events were successfully captured in aggregations")
        else:
            print("✗ Some events were NOT captured in any aggregation")

        time.sleep(1)
        self._create_summary()

    def _create_summary(self):
        summary_path = self.session_dir / "summary.png"
        print(f"\nSaving summary plot in {summary_path} ...")
        agg_path = self.session_dir / "aggregations.jsonl"
        events_path = self.session_dir / "events.jsonl"

        if agg_path.exists() and events_path.exists():
            plot_summary_stats(self.session_dir, agg_path, events_path, summary_path)

    def run(self):
        """Run the recorder until interrupted."""
        def signal_handler(sig, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.start()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Record screen activity with input events"
    )
    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=30,
        help="Frames per second to capture (default: 16)"
    )
    parser.add_argument(
        "-s", "--buffer-seconds",
        type=int,
        default=12,
        help="Number of seconds to keep in buffer (default: 24)"
    )
    parser.add_argument(
        "-b", "--buffer-all-images",
        action="store_true",
        help="Save all buffer images to disk"
    )
    parser.add_argument(
        "-m", "--monitor",
        action="store_true",
        help="Enable real-time monitoring of the last session"
    )
    parser.add_argument(
        "-r", "--max-res",
        type=int,
        default=None,
        nargs=2,
        help="Maximal resolution for screenshots (width, height)"
    )
    parser.add_argument(
        "-p", "--precision",
        type=str,
        choices=["accurate", "rough"],
        default="accurate",
        help="Precision level for event aggregation (default: accurate)"
    )
    parser.add_argument(
        "-a", "--accessibility",
        action="store_true",
        help="Enable accessibility info capture (macOS only, may impact performance)"
    )
    parser.add_argument(
        "-c", "--compression-quality",
        type=int,
        help="JPEG compression quality for saved screenshots (1-100, default: 70)",
        default=70
    )

    args = parser.parse_args()

    constants_manager.set_preset(args.precision, verbose=False)

    recorder = ScreenRecorder(
        fps=args.fps,
        buffer_seconds=args.buffer_seconds,
        buffer_all=args.buffer_all_images,
        monitor=args.monitor,
        max_res=args.max_res,
        accessibility=args.accessibility,
        compression_quality=args.compression_quality
    )
    recorder.run()


if __name__ == "__main__":
    main()
