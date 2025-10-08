import argparse
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from pynput import mouse, keyboard

from record.models import ImageQueue, AggregationConfig, EventQueue
from record.workers import SaveWorker, AggregationWorker, WandBLogger
from record.handlers import InputEventHandler, ScreenshotHandler


class ScreenRecorder:

    def __init__(
        self, fps: int = 30,
        buffer_seconds: int = 12,
        buffer_all: bool = False,
        use_wandb: bool = True,
        wandb_project: str = "screen-recorder"
    ):
        """
        Initialize the screen recorder.

        Args:
            fps: Frames per second to capture
            buffer_seconds: Number of seconds to keep in buffer
            buffer_all: If True, save all screenshots to disk
            use_wandb: If True, enable WandB logging
            wandb_project: WandB project name
        """
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.buffer_all = buffer_all
        self.use_wandb = use_wandb

        self.image_buffer_size = fps * buffer_seconds
        self.event_buffer_size = fps * buffer_seconds * 30

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(__file__).parent.parent / "logs" / f"session_v5_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        print(f"Session directory: {self.session_dir}")

        # Initialize WandB logger if enabled
        self.wandb_logger = None
        if use_wandb:
            try:
                session_name = f"session_{timestamp}"
                self.wandb_logger = WandBLogger(
                    project_name=wandb_project,
                    session_name=session_name
                )
                print(f"WandB logging enabled: {wandb_project}/{session_name}")
            except Exception as e:
                print(f"Warning: Failed to initialize WandB: {e}")
                self.wandb_logger = None

        self.input_event_queue = EventQueue(
            click_config=AggregationConfig(gap_threshold=0.5, total_threshold=2.0),
            move_config=AggregationConfig(gap_threshold=0.1, total_threshold=1.0),
            scroll_config=AggregationConfig(gap_threshold=0.3, total_threshold=1.5),
            key_config=AggregationConfig(gap_threshold=0.5, total_threshold=3.0),
            poll_interval=1.0,
            wandb_logger=self.wandb_logger
        )

        print(f"Session directory: {self.session_dir}")

        self.image_queue = ImageQueue(max_length=self.image_buffer_size)
        self.ssim_queue = ImageQueue(max_length=self.image_buffer_size)

        self.save_worker = SaveWorker(self.session_dir, buffer_all)

        self.aggregation_worker = AggregationWorker(
            event_queue=self.input_event_queue,
            image_queue=self.image_queue,
            save_worker=self.save_worker,
            wandb_logger=self.wandb_logger
        )

        # Set callbacks
        self.input_event_queue.set_callback(self._on_aggregated_events)
        self.input_event_queue.set_aggregation_callback(self._on_aggregation_requests)

        self.ssim_queue.add_callback(self._on_ssim_computed)
        self.image_queue.add_callback(self._on_new_image)

        self.input_handler = InputEventHandler(self.input_event_queue)

        self.screenshot_manager = ScreenshotHandler(
            image_queue=self.image_queue,
            ssim_queue=self.ssim_queue,
            fps=self.fps
        )

        # Input listeners
        self.mouse_listener = None
        self.keyboard_listener = None

        # Running flag
        self.running = False

        # Statistics
        self.processed_aggregations = 0

    def _on_aggregated_events(self, event_type: str, events: list):
        """
        Callback for aggregated events (intermediate step).

        Args:
            event_type: Type of aggregation ('click', 'move', 'scroll', 'key')
            events: List of aggregated events
        """
        print(f"Aggregated {event_type} burst: {len(events)} events "
              f"({events[0].timestamp:.3f} -> {events[-1].timestamp:.3f})")

    def _on_aggregation_requests(self, requests: list):
        """
        Callback for ready aggregation requests.
        This is called by the EventQueue poll worker when aggregations are ready.

        Args:
            requests: List of AggregationRequest objects
        """
        if not requests:
            return

        print(f"\nProcessing {len(requests)} aggregation requests...")

        # Process aggregations using the aggregation worker
        processed = self.aggregation_worker.process_aggregations(requests)

        # Log results
        for agg in processed:
            screenshot_status = "✓" if agg.screenshot else "✗"
            path_info = f"-> {agg.screenshot_path}" if agg.screenshot_path else ""
            print(f"  {screenshot_status} {agg.request.reason:20s} @ {agg.request.timestamp:.3f} "
                  f"| {len(agg.events)} events {path_info}")

        self.processed_aggregations += len(processed)

        # Clean up old events to prevent memory growth
        if requests:
            oldest_timestamp = requests[0].timestamp
            self.aggregation_worker.cleanup_old_events(oldest_timestamp)

        # Here you can add your custom logic to save the processed aggregations
        # For now, they are just logged
        # Future: self._save_aggregations(processed)

    def _on_ssim_computed(self, image):
        """Callback for saving SSIM values."""
        self.save_worker.save_ssim_value(image)

    def _on_new_image(self, image):
        """Callback for saving all buffer images (only used if buffer_all)."""
        if self.buffer_all:
            self.save_worker.save_buffer_image(image)

    def start(self):
        """Start recording."""
        if self.running:
            print("Recorder already running")
            return

        self.running = True
        print(f"Starting screen recorder at {self.fps} FPS")
        print(f"Buffer: {self.buffer_seconds} seconds ({self.image_buffer_size} frames)")
        print("Input event aggregation: ENABLED")
        print(f"  - Save all images: {self.buffer_all}")

        # Start input event queue polling
        self.input_event_queue.start()

        self.screenshot_manager.start()

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

        print("Recorder started. Press Ctrl+C to stop.")

    def stop(self):
        """Stop recording."""
        if not self.running:
            return

        print("\nStopping recorder...")
        self.running = False

        # Stop input event queue (this will process remaining aggregations)
        self.input_event_queue.stop()

        self.screenshot_manager.stop()

        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        print(f"\nSession saved to: {self.session_dir}")
        print(f"Aggregations saved to: {self.aggregation_worker.aggregations_file}")

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
        description="Record screen activity with input events and SSIM metrics"
    )
    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=16,
        help="Frames per second to capture (default: 16)"
    )
    parser.add_argument(
        "-s", "--buffer-seconds",
        type=int,
        default=12,
        help="Number of seconds to keep in buffer (default: 12)"
    )
    parser.add_argument(
        "-b", "--buffer-all-images",
        action="store_true",
        help="Save all buffer images to disk"
    )

    args = parser.parse_args()

    recorder = ScreenRecorder(
        fps=args.fps,
        buffer_seconds=args.buffer_seconds,
        buffer_all=args.buffer_all_images
    )
    recorder.run()


if __name__ == "__main__":
    main()
