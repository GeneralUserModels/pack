import time
import threading
from typing import Optional
from record.models.event_queue import EventQueue
from record.models.image import BufferImage
from record.workers.screenshot import capture_screenshot
from record.workers.ssim import compute_ssim


class ScreenshotHandler:
    """Handler for capturing screenshots and computing SSIM values."""

    def __init__(
        self,
        image_queue: EventQueue,
        ssim_queue: EventQueue,
        fps: int = 30,
        monitor_index: Optional[int] = None
    ):
        """
        Initialize the screenshot manager.

        Args:
            image_queue: Queue to store captured images
            ssim_queue: Queue to store SSIM values
            fps: Frames per second to capture
            monitor_index: Specific monitor to capture (None for active monitor)
        """
        self.image_queue = image_queue
        self.ssim_queue = ssim_queue
        self.fps = fps
        self.monitor_index = monitor_index
        self.interval = 1.0 / fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._previous_image: Optional[BufferImage] = None

        # Register callback for SSIM computation
        self.image_queue.add_callback(self._on_new_image)

    def _on_new_image(self, image: BufferImage) -> None:
        """
        Callback triggered when a new image is added to the queue.
        Starts SSIM computation process.

        Args:
            image: Newly added image
        """
        if self._previous_image is not None:
            # Compute SSIM in a separate thread to avoid blocking
            thread = threading.Thread(
                target=self._compute_and_store_ssim,
                args=(self._previous_image, image),
                daemon=True
            )
            thread.start()

        self._previous_image = image

    def _compute_and_store_ssim(self, img1: BufferImage, img2: BufferImage) -> None:
        """
        Compute SSIM between two images and store the result.

        Args:
            img1: First image
            img2: Second image
        """
        try:
            ssim_value = compute_ssim(img1.screenshot, img2.screenshot)

            # Update the second image with SSIM value
            img2.ssim_value = ssim_value

            # Create a copy for the SSIM queue
            ssim_entry = BufferImage(
                timestamp=img2.timestamp,
                screenshot=None,  # Don't duplicate the image
                ssim_value=ssim_value,
                monitor_index=img2.monitor_index
            )
            self.ssim_queue.enqueue(ssim_entry)
        except Exception as e:
            print(f"Error computing SSIM: {e}")

    def _capture_loop(self) -> None:
        """Main loop for capturing screenshots."""
        while self._running:
            start_time = time.time()

            try:
                # Capture screenshot
                timestamp = time.time()
                screenshot = capture_screenshot(self.monitor_index)

                if screenshot is not None:
                    # Create BufferImage
                    buffer_image = BufferImage(
                        timestamp=timestamp,
                        screenshot=screenshot,
                        monitor_index=self.monitor_index if self.monitor_index is not None else 0
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

        self.image_queue.remove_callback(self._on_new_image)
        print("Screenshot manager stopped")
