import time
import threading
from collections import deque
from PIL import Image
import io
import mss


class ScreenshotManager:
    def __init__(self, fps=10, buffer_seconds=3.0):
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.max_frames = int(fps * buffer_seconds)

        self.monitor_buffers = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._sct = None

    def start(self):
        """Start the continuous capture thread"""
        if self._capture_thread and self._capture_thread.is_alive():
            return

        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self):
        """Stop the continuous capture"""
        if self._capture_thread:
            self._stop_event.set()
            self._capture_thread.join(timeout=2)
        if self._sct:
            self._sct.close()

    def _capture_loop(self):
        """Continuously captures all monitors at FPS and stores frames in buffers"""
        self._sct = mss.mss(with_cursor=True)
        capture_interval = 1.0 / self.fps

        try:
            while not self._stop_event.is_set():
                start_time = time.time()

                for i, monitor in enumerate(self._sct.monitors[1:], 1):
                    try:
                        screenshot = self._sct.grab(monitor)
                        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                        with self._lock:
                            if i not in self.monitor_buffers:
                                self.monitor_buffers[i] = deque(maxlen=self.max_frames)
                            self.monitor_buffers[i].append((start_time, img))

                    except Exception as e:
                        print(f"Error capturing monitor {i}: {e}")

                elapsed = time.time() - start_time
                sleep_time = capture_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"Capture loop error: {e}")
        finally:
            if self._sct:
                self._sct.close()

    def _is_active_monitor(self, mon: dict, x: int, y: int) -> bool:
        """Check if coordinates are within monitor bounds"""
        return (mon["left"] <= x < mon["left"] + mon["width"] and
                mon["top"] <= y < mon["top"] + mon["height"])

    def get_active_monitor(self, x: int, y: int) -> dict:
        """Get the monitor dict that contains the given coordinates"""
        if not self._sct:
            with mss.mss() as temp_sct:
                for i, mon in enumerate(temp_sct.monitors[1:], 1):
                    if self._is_active_monitor(mon, x, y):
                        mon['monitor_id'] = i
                        return mon
                # Fallback to primary monitor
                primary = temp_sct.monitors[1].copy()
                primary['monitor_id'] = 1
                return primary
        else:
            for i, mon in enumerate(self._sct.monitors[1:], 1):
                if self._is_active_monitor(mon, x, y):
                    mon['monitor_id'] = i
                    return mon
            primary = self._sct.monitors[1].copy()
            primary['monitor_id'] = 1
            return primary

    def take_screenshot_for_monitor(self, mon: dict, quality: int = 95, lookback_ms: int = 50) -> tuple[bytes, tuple[int, int]]:
        monitor_id = mon.get('monitor_id', 1)

        with self._lock:
            if monitor_id not in self.monitor_buffers or not self.monitor_buffers[monitor_id]:
                print(f"No buffer available for monitor {monitor_id}")
                return None, None

            target_time = time.time() - (lookback_ms / 1000.0)
            buffer = self.monitor_buffers[monitor_id]

            selected_frame = None
            for timestamp, img in buffer:
                if timestamp <= target_time:
                    selected_frame = (timestamp, img)
                else:
                    break

            if selected_frame is None and buffer:
                selected_frame = buffer[0]

            if selected_frame is None:
                return None, None

            _, img = selected_frame

            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=quality)
            jpeg_data = jpeg_buffer.getvalue()

            return jpeg_data, img.size

    def take_virtual_screenshot(self, quality: int = 95) -> tuple[bytes, tuple[int, int]]:
        """Take a screenshot of the entire virtual desktop (all monitors combined)"""
        try:
            with mss.mss(with_cursor=True) as sct:
                img = sct.grab(sct.monitors[0])
                pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

                jpeg_buffer = io.BytesIO()
                pil_img.save(jpeg_buffer, format='JPEG', quality=quality)
                jpeg_data = jpeg_buffer.getvalue()

                return jpeg_data, img.size
        except Exception as e:
            print(f"Error taking virtual screenshot: {e}")
            return None, None

    def close(self):
        """Clean up resources"""
        self.stop()
