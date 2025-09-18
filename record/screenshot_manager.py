import time
import threading
from collections import deque
from PIL import Image
import io
import mss
import multiprocessing
from pathlib import Path
import queue


def _save_worker_process(save_queue, base_dir):
    """Worker process that saves screenshots to disk"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            item = save_queue.get(timeout=1.0)
            if item is None:  # Poison pill to stop
                break

            timestamp, monitor_id, img_data = item

            # Create filename with timestamp and monitor ID
            filename = f"buffer_{monitor_id}_{timestamp:.6f}.jpg"
            filepath = base_path / filename

            # Save the image data directly to disk
            with open(filepath, 'wb') as f:
                f.write(img_data)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in save worker: {e}")
            continue


class ScreenshotManager:
    def __init__(self, fps=10, buffer_seconds=3.0, save_all_buffer=False, buffer_save_dir=None):
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.max_frames = int(fps * buffer_seconds)
        self.save_all_buffer = save_all_buffer
        self.buffer_save_dir = buffer_save_dir

        self.monitor_buffers = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._sct = None

        # For saving buffer frames
        self._save_process = None
        self._save_queue = None

        if self.save_all_buffer:
            if not self.buffer_save_dir:
                raise ValueError("buffer_save_dir must be provided when save_all_buffer=True")
            self._setup_save_process()

    def _setup_save_process(self):
        """Set up the separate process for saving buffer frames"""
        self._save_queue = multiprocessing.Queue(maxsize=1000)  # Prevent memory buildup
        self._save_process = multiprocessing.Process(
            target=_save_worker_process,
            args=(self._save_queue, self.buffer_save_dir),
            daemon=True
        )
        self._save_process.start()

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

        # Clean up save process
        if self._save_process and self._save_process.is_alive():
            try:
                # Send poison pill to stop worker
                self._save_queue.put(None, timeout=1.0)
                self._save_process.join(timeout=3.0)
                if self._save_process.is_alive():
                    self._save_process.terminate()
            except:
                pass

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

                        # Save to disk if flag is enabled
                        if self.save_all_buffer and self._save_queue:
                            try:
                                # Convert to JPEG in memory
                                jpeg_buffer = io.BytesIO()
                                img.save(jpeg_buffer, format='JPEG', quality=85)
                                jpeg_data = jpeg_buffer.getvalue()

                                # Send to save process (non-blocking)
                                self._save_queue.put((start_time, i, jpeg_data), block=False)
                            except queue.Full:
                                print(f"Save queue full, dropping buffer frame for monitor {i}")
                            except Exception as e:
                                print(f"Error queuing buffer frame for save: {e}")

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
