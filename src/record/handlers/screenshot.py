import time
import threading
from typing import Optional, Union
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
        monitor_index: Optional[int] = None,
        max_res: Optional[tuple[int, int]] = None,
        scale: Optional[Union[float, dict[int, float]]] = None
    ):
        """
        Initialize the screenshot manager.

        Args:
            image_queue: Queue to store captured images
            fps: Frames per second to capture
            monitor_index: Specific monitor to capture (None for active monitor)
            max_res: Optional max resolution to fit within
            scale: Optional scale factor (0.0-1.0) or dict of per-monitor scales
        """
        self.image_queue = image_queue
        self.fps = fps
        self.monitor_index = monitor_index
        self.max_res = max_res
        self.scale = scale
        self.interval = 1.0 / fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._previous_image: Optional[BufferImage] = None
        self.mouse_controller = mouse.Controller()

    def _capture_loop(self) -> None:
        """Main loop for capturing screenshots."""
        with mss.mss(with_cursor=True) as sct:
            while self._running:
                start_time = time.time()
                try:
                    x, y = self.mouse_controller.position
                    screenshot, monitor_index, timestamp, scale_factor, monitor_dict = capture_screenshot(
                        sct, x, y, max_res=self.max_res, scale=self.scale
                    )
                    if screenshot is not None:
                        buffer_image = BufferImage(
                            timestamp=timestamp,
                            screenshot=screenshot,
                            monitor_index=monitor_index,
                            scale_factor=scale_factor,
                            monitor_dict=monitor_dict
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
