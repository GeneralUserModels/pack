import argparse
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from pynput import mouse, keyboard

from record.models import EventQueue
from record.handlers import InputEventHandler, ScreenshotHandler
from record.workers.save import SaveWorker


class ScreenRecorder:
    """Main application for recording screen activity."""

    def __init__(self, fps: int = 30, buffer_seconds: int = 6, buffer_all: bool = False):
        """
        Initialize the screen recorder.

        Args:
            fps: Frames per second to capture
            buffer_seconds: Number of seconds to keep in buffer
            buffer_all: If True, save all screenshots to disk
        """
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.buffer_all = buffer_all

        self.image_buffer_size = fps * buffer_seconds
        self.event_buffer_size = fps * buffer_seconds * 30

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(__file__).parent.parent / "logs" / f"session_v4_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        print(f"Session directory: {self.session_dir}")

        # Initialize queues
        self.input_event_queue = EventQueue(max_length=self.event_buffer_size)
        self.image_queue = EventQueue(max_length=self.image_buffer_size)
        self.ssim_queue = EventQueue(max_length=self.image_buffer_size)

        # Initialize save worker
        self.save_worker = SaveWorker(self.session_dir)

        # Register callbacks for saving
        self.input_event_queue.add_callback(self._on_input_event)
        self.ssim_queue.add_callback(self._on_ssim_computed)

        if self.buffer_all:
            self.image_queue.add_callback(self._on_new_image)

        # Initialize input event handler
        self.input_handler = InputEventHandler(self.input_event_queue)

        # Initialize screenshot manager
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

    def _on_input_event(self, event):
        """Callback for saving input events."""
        self.save_worker.save_input_event(event)

    def _on_ssim_computed(self, image):
        """Callback for saving SSIM values."""
        self.save_worker.save_ssim_value(image)

    def _on_new_image(self, image):
        """Callback for saving all buffer images."""
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
        print(f"Save all images: {self.buffer_all}")

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

        self.screenshot_manager.stop()

        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        print("\nRecording statistics:")
        print(f"  Input events: {len(self.input_event_queue)}")
        print(f"  Images captured: {len(self.image_queue)}")
        print(f"  SSIM values computed: {len(self.ssim_queue)}")
        print(f"\nSession saved to: {self.session_dir}")

    def run(self):
        """Run the recorder until interrupted."""
        # Setup signal handler for graceful shutdown
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
        default=30,
        help="Frames per second to capture (default: 30)"
    )
    parser.add_argument(
        "-s", "--buffer-seconds",
        type=int,
        default=6,
        help="Number of seconds to keep in buffer (default: 6)"
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
