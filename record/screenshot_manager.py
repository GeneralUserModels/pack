import time
import threading
from collections import deque
from PIL import Image
import io
import mss
import multiprocessing
from pathlib import Path
import queue
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim


def get_cursor_position():
    """Get current cursor position - simplified version"""
    try:
        from pynput.mouse import Controller
        mouse = Controller()
        pos = mouse.position
        return int(pos[0]), int(pos[1])
    except Exception:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            x = root.winfo_pointerx()
            y = root.winfo_pointery()
            root.destroy()
            return x, y
        except Exception:
            return 960, 540


def _save_worker_process(save_queue, base_dir):
    """Worker process that saves screenshots to disk"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            item = save_queue.get(timeout=1.0)
            if item is None:  # Poison pill to stop
                break

            timestamp, img_data = item

            # Create filename with timestamp
            filename = f"buffer_active_{timestamp:.6f}.jpg"
            filepath = base_path / filename

            # Save the image data directly to disk
            with open(filepath, 'wb') as f:
                f.write(img_data)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in save worker: {e}")
            continue


def _ssim_logger_worker(ssim_queue, log_file):
    """Worker process that logs SSIM similarities to JSONL file"""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w') as f:
        while True:
            try:
                item = ssim_queue.get(timeout=1.0)
                if item is None:  # Poison pill to stop
                    break

                # Write JSONL entry
                json.dump(item, f)
                f.write('\n')
                f.flush()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in SSIM logger: {e}")
                continue


class ScreenshotManager:
    def __init__(self, fps=10, buffer_seconds=3.0, save_all_buffer=False, buffer_save_dir=None,
                 log_ssim=True, ssim_log_file=None):
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.max_frames = int(fps * buffer_seconds)
        self.save_all_buffer = save_all_buffer
        self.buffer_save_dir = buffer_save_dir
        self.log_ssim = log_ssim
        self.ssim_log_file = ssim_log_file

        self.buffer = deque(maxlen=self.max_frames)  # Single buffer for active monitor
        self.prev_image = None  # Store previous grayscale image for SSIM
        self.current_monitor_id = 1  # Track current active monitor
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._sct = None

        # For saving buffer frames
        self._save_process = None
        self._save_queue = None

        # For SSIM logging
        self._ssim_process = None
        self._ssim_queue = None

        if self.save_all_buffer:
            if not self.buffer_save_dir:
                raise ValueError("buffer_save_dir must be provided when save_all_buffer=True")
            self._setup_save_process()

        if self.log_ssim:
            if not self.ssim_log_file:
                raise ValueError("ssim_log_file must be provided when log_ssim=True")
            self._setup_ssim_process()

    def _setup_save_process(self):
        """Set up the separate process for saving buffer frames"""
        self._save_queue = multiprocessing.Queue(maxsize=1000)
        self._save_process = multiprocessing.Process(
            target=_save_worker_process,
            args=(self._save_queue, self.buffer_save_dir),
            daemon=True
        )
        self._save_process.start()

    def _setup_ssim_process(self):
        """Set up the separate process for logging SSIM similarities"""
        self._ssim_queue = multiprocessing.Queue(maxsize=1000)
        self._ssim_process = multiprocessing.Process(
            target=_ssim_logger_worker,
            args=(self._ssim_queue, self.ssim_log_file),
            daemon=True
        )
        self._ssim_process.start()

    def _get_cursor_position(self):
        """Get current cursor position"""
        return get_cursor_position()

    def _get_active_monitor_bounds(self):
        """Get the bounds of the monitor containing the cursor"""
        cursor_x, cursor_y = self._get_cursor_position()

        # Always create a new MSS instance for thread safety
        try:
            with mss.mss() as temp_sct:
                for i, monitor in enumerate(temp_sct.monitors[1:], 1):
                    if (monitor["left"] <= cursor_x < monitor["left"] + monitor["width"] and
                            monitor["top"] <= cursor_y < monitor["top"] + monitor["height"]):
                        monitor['monitor_id'] = i
                        return monitor
                # Fallback to primary monitor
                primary = temp_sct.monitors[1].copy()
                primary['monitor_id'] = 1
                return primary
        except Exception as e:
            print(f"Error getting active monitor bounds: {e}")
            # Ultimate fallback - return a default monitor configuration
            return {
                'left': 0,
                'top': 0,
                'width': 1920,
                'height': 1080,
                'monitor_id': 1
            }

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
                self._save_queue.put(None, timeout=1.0)
                self._save_process.join(timeout=3.0)
                if self._save_process.is_alive():
                    self._save_process.terminate()
            except:
                pass

        # Clean up SSIM process
        if self._ssim_process and self._ssim_process.is_alive():
            try:
                self._ssim_queue.put(None, timeout=1.0)
                self._ssim_process.join(timeout=3.0)
                if self._ssim_process.is_alive():
                    self._ssim_process.terminate()
            except:
                pass

    def _capture_loop(self):
        """Continuously captures the active monitor at FPS and stores frames in buffer"""
        self._sct = mss.mss(with_cursor=True)
        capture_interval = 1.0 / self.fps

        try:
            while not self._stop_event.is_set():
                start_time = time.time()

                try:
                    # Get active monitor bounds
                    monitor = self._get_active_monitor_bounds()
                    self.current_monitor_id = monitor['monitor_id']

                    # Capture the active monitor
                    screenshot = self._sct.grab(monitor)
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                    # Convert to grayscale for SSIM calculation
                    img_gray = img.convert('L')
                    img_gray_array = np.array(img_gray)

                    # Generate filename for this frame
                    filename = f"buffer_active_{start_time:.6f}.jpg"

                    # Calculate SSIM if we have a previous image and SSIM logging is enabled
                    ssim_value = None
                    if self.log_ssim and self.prev_image is not None:
                        try:
                            ssim_value = ssim(self.prev_image, img_gray_array,
                                              data_range=img_gray_array.max() - img_gray_array.min())

                            # Log SSIM data
                            ssim_entry = {
                                'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S-%f', time.localtime(start_time)),
                                'unix_timestamp': start_time,
                                'monitor_id': self.current_monitor_id,
                                'current_image': filename,
                                'ssim_similarity': float(ssim_value),
                                'image_width': img.size[0],
                                'image_height': img.size[1]
                            }

                            if self._ssim_queue:
                                try:
                                    self._ssim_queue.put(ssim_entry, block=False)
                                except queue.Full:
                                    print(f"SSIM queue full, dropping entry")
                        except Exception as e:
                            print(f"Error calculating SSIM: {e}")

                    # Store current grayscale image for next comparison
                    if self.log_ssim:
                        self.prev_image = img_gray_array

                    with self._lock:
                        # Store frame data
                        frame_data = {
                            'timestamp': start_time,
                            'image': img,
                            'filename': filename,
                            'ssim': ssim_value,
                            'monitor_id': self.current_monitor_id
                        }
                        self.buffer.append(frame_data)

                    # Save to disk if flag is enabled
                    if self.save_all_buffer and self._save_queue:
                        try:
                            # Convert to JPEG in memory
                            jpeg_buffer = io.BytesIO()
                            img.save(jpeg_buffer, format='JPEG', quality=85)
                            jpeg_data = jpeg_buffer.getvalue()

                            # Send to save process (non-blocking)
                            self._save_queue.put((start_time, jpeg_data), block=False)
                        except queue.Full:
                            print(f"Save queue full, dropping buffer frame")
                        except Exception as e:
                            print(f"Error queuing buffer frame for save: {e}")

                except Exception as e:
                    print(f"Error capturing active monitor: {e}")

                elapsed = time.time() - start_time
                sleep_time = capture_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"Capture loop error: {e}")
        finally:
            if self._sct:
                self._sct.close()

    def get_active_monitor(self, x: int, y: int) -> dict:
        """Get the active monitor dict - thread safe version"""
        # Always create a fresh MSS instance for thread safety
        try:
            with mss.mss() as temp_sct:
                for i, monitor in enumerate(temp_sct.monitors[1:], 1):
                    if (monitor["left"] <= x < monitor["left"] + monitor["width"] and
                            monitor["top"] <= y < monitor["top"] + monitor["height"]):
                        monitor['monitor_id'] = i
                        return monitor
                # Fallback to primary monitor
                primary = temp_sct.monitors[1].copy()
                primary['monitor_id'] = 1
                return primary
        except Exception as e:
            print(f"Error in get_active_monitor: {e}")
            # Return default monitor
            return {
                'left': 0,
                'top': 0,
                'width': 1920,
                'height': 1080,
                'monitor_id': 1
            }

    def take_screenshot_for_monitor(self, mon: dict, quality: int = 95, lookback_ms: int = 50) -> tuple[bytes, tuple[int, int]]:
        """Take screenshot from buffer (ignores monitor parameter, always uses active monitor buffer)"""
        with self._lock:
            if not self.buffer:
                print("No buffer available")
                return None, None

            target_time = time.time() - (lookback_ms / 1000.0)

            selected_frame = None
            for frame_data in self.buffer:
                if frame_data['timestamp'] <= target_time:
                    selected_frame = frame_data
                else:
                    break

            if selected_frame is None and self.buffer:
                selected_frame = self.buffer[0]

            if selected_frame is None:
                return None, None

            img = selected_frame['image']

            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=quality)
            jpeg_data = jpeg_buffer.getvalue()

            return jpeg_data, img.size

    def take_virtual_screenshot(self, quality: int = 95) -> tuple[bytes, tuple[int, int]]:
        """Take a screenshot of the active monitor"""
        try:
            monitor = self._get_active_monitor_bounds()
            # Use a separate MSS instance for thread safety
            with mss.mss(with_cursor=True) as sct:
                img = sct.grab(monitor)
                pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

                jpeg_buffer = io.BytesIO()
                pil_img.save(jpeg_buffer, format='JPEG', quality=quality)
                jpeg_data = jpeg_buffer.getvalue()

                return jpeg_data, img.size
        except Exception as e:
            print(f"Error taking active monitor screenshot: {e}")
            return None, None

    def get_recent_ssim_values(self, count: int = 10) -> list:
        """Get recent SSIM values"""
        with self._lock:
            if not self.buffer:
                return []

            recent_frames = list(self.buffer)[-count:]
            return [frame.get('ssim') for frame in recent_frames if frame.get('ssim') is not None]

    def close(self):
        """Clean up resources"""
        self.stop()
