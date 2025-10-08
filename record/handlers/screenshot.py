import time
import threading
from typing import Optional
from pynput import mouse
import mss
from record.models.event_queue import EventQueue
from record.models.image import BufferImage
from record.workers.screenshot import capture_screenshot


class ScreenshotHandler:
    """Handler for capturing screenshots"""

    def __init__(
        self,
        image_queue: EventQueue,
        fps: int = 30,
        monitor_index: Optional[int] = None
    ):
        """
        Initialize the screenshot manager.

        Args:
            image_queue: Queue to store captured images
            fps: Frames per second to capture
            monitor_index: Specific monitor to capture (None for active monitor)
        """
        self.image_queue = image_queue
        self.fps = fps
        self.monitor_index = monitor_index
        self.interval = 1.0 / fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._previous_image: Optional[BufferImage] = None
        self.mouse_controller = mouse.Controller()

    def _capture_loop(self) -> None:
        """Main loop for capturing screenshots."""
        with mss.mss() as sct:
            while self._running:
                start_time = time.time()

                try:
                    x, y = self.mouse_controller.position
                    timestamp = time.time()
                    screenshot, monitor_index = capture_screenshot(sct, x, y)

                    if screenshot is not None:
                        buffer_image = BufferImage(
                            timestamp=timestamp,
                            screenshot=screenshot,
                            monitor_index=monitor_index
                        )

                        self.image_queue.enqueue(buffer_image)
                except Exception as e:
                    print(f"Error capturing screenshot: {e}")

                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                time.sleep(sleep_time)

    def start(self) -> None:
        """Start capturing screenshots."""
        if self._running:
            print("Screenshot manager already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"Screenshot manager started at {self.fps} FPS")

    def stop(self) -> None:
        """Stop capturing screenshots."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
