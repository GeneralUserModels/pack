import datetime
import threading
from pathlib import Path
import asyncio

# system libs
import time
import io
import json
from collections import deque
from PIL import Image
import mss
import mss.tools
import multiprocessing
import numpy as np
from multiprocessing import shared_memory, Value, Lock

SESSION_DIR = Path(__file__).parent.parent / "logs" / f"session_v2_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
SCREENSHOT_DIR = SESSION_DIR / "screenshots"
BUFFER_SCREENSHOTS_DIR = SESSION_DIR / "buffer_screenshots"
LOG_FILE = SESSION_DIR / "events.jsonl"
SSIM_LOG_FILE = SESSION_DIR / "img_similarities.jsonl"

SESSION_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

BUFFER_FPS = 60                  # target capture FPS
BUFFER_SECONDS = 6.0             # how many seconds of thumbnails to keep
SAVE_ALL_BUFFER = True           # whether to attempt to save full-res frames (may drop)
LOG_SSIM = True                  # whether to compute & log SSIM
THUMB_W = 320                    # thumbnail width (keeps aspect ratio)
JPEG_QUALITY_SAVE = 70           # quality when saving full-res (tune for speed)
SAVE_QUEUE_MAXSIZE = 512         # max queue size for save worker (bounded to avoid OOM)

# ------------------------
# SSIM worker process (separate from save workers for better performance)
# ------------------------


def _ssim_worker_process(ssim_queue, log_file):
    """Dedicated SSIM computation and logging process."""
    from pathlib import Path
    import queue as _q
    from skimage.metrics import structural_similarity as ssim
    import numpy as np

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prev_thumb = None

    with open(log_path, "w") as f:
        while True:
            try:
                item = ssim_queue.get(timeout=1.0)
                if item is None:
                    break

                # Item format: (timestamp, thumb_bytes, thumb_w, thumb_h, monitor_id, slot_index)
                timestamp, thumb_bytes, thumb_w, thumb_h, monitor_id, slot_index = item

                # Convert bytes to numpy array
                try:
                    arr = np.frombuffer(thumb_bytes, dtype=np.uint8).reshape((thumb_h, thumb_w))
                except Exception as e:
                    print(f"[ssim_worker] error reshaping thumb: {e}")
                    continue

                # compute SSIM against prev_thumb if available
                ssim_value = None
                if prev_thumb is not None:
                    try:
                        data_range = float(arr.max() - arr.min()) if arr.max() != arr.min() else 1.0
                        ssim_value = ssim(prev_thumb, arr, data_range=data_range)

                        ssim_entry = {
                            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S-%f", time.localtime(timestamp)),
                            "unix_timestamp": timestamp,
                            "monitor_id": monitor_id,
                            "slot_index": slot_index,
                            "ssim_similarity": float(ssim_value),
                            "thumb_w": thumb_w,
                            "thumb_h": thumb_h
                        }
                        json.dump(ssim_entry, f)
                        f.write("\n")
                        f.flush()
                    except Exception as e:
                        print(f"[ssim_worker] SSIM computation error: {e}")

                # Update previous thumbnail
                prev_thumb = arr.copy()

            except _q.Empty:
                continue
            except Exception as e:
                print(f"[ssim_worker] error: {e}")
                continue


# ------------------------
# Async save worker
# ------------------------

async def _async_save_worker(name: str, queue: asyncio.Queue, out_dir: Path):
    """Async worker that saves images to disk using aiofiles."""
    import aiofiles

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        if item is None:  # shutdown sentinel
            queue.task_done()
            break

        try:
            timestamp, image_bytes, format_type, filename = item
            filepath = out_dir / filename

            async with aiofiles.open(filepath, "wb") as f:
                await f.write(image_bytes)

        except Exception as e:
            print(f"[{name}] Error saving: {e}")
        finally:
            queue.task_done()


# ------------------------
# Shared thumbnail circular buffer (unchanged from original)
# ------------------------

class ThumbSharedBuffer:
    """
    Circular buffer stored in shared memory that holds grayscale thumbnails (uint8).
    Each slot holds exactly slot_size bytes (thumb_w * thumb_h).
    A small header with head index (next write) is maintained using a multiprocessing.Value.
    """

    def __init__(self, slots: int, thumb_w: int, thumb_h: int, shm_name: str = None):
        self.slots = int(slots)
        self.thumb_w = int(thumb_w)
        self.thumb_h = int(thumb_h)
        self.slot_size = self.thumb_w * self.thumb_h  # bytes per slot (uint8)
        total = self.slots * self.slot_size

        # create shared memory block
        if shm_name is None:
            self.shm = shared_memory.SharedMemory(create=True, size=total)
            self.created_here = True
        else:
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.created_here = False

        # head pointer and lock to coordinate writer(s)
        self.head = Value('i', 0)   # index of next write (0..slots-1)
        self.lock = Lock()

    def name(self):
        return self.shm.name

    def write_slot(self, index: int, data: bytes):
        """
        Write `data` bytes (must be exactly slot_size) into slot `index`.
        """
        if len(data) != self.slot_size:
            raise ValueError(f"Data length {len(data)} != slot_size {self.slot_size}")
        start = index * self.slot_size
        end = start + self.slot_size
        # a memoryview copy
        self.shm.buf[start:end] = data

    def read_slot(self, index: int) -> bytes:
        start = index * self.slot_size
        end = start + self.slot_size
        return bytes(self.shm.buf[start:end])

    def next_write_index(self):
        with self.lock:
            idx = int(self.head.value)
            self.head.value = (idx + 1) % self.slots
            return idx

    def close(self):
        try:
            self.shm.close()
            if self.created_here:
                self.shm.unlink()
        except Exception:
            pass


# ------------------------
# Async capture process using the efficient approach from the second script
# ------------------------

async def _async_capture_process(meta_queue, save_queue_async, ssim_queue, shm_name, thumb_w, thumb_h, slots, fps, stop_event, save_all_buffer=True, with_cursor=True):
    """
    Async capture process that efficiently captures screenshots and processes them.
    Uses the approach from the working async script.
    """
    import concurrent.futures

    # reopen shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    slot_size = thumb_w * thumb_h

    def get_cursor_position_proc():
        try:
            from pynput.mouse import Controller
            c = Controller()
            return int(c.position[0]), int(c.position[1])
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

    def get_active_monitor_bounds_proc():
        cx, cy = get_cursor_position_proc()
        try:
            with mss.mss() as temp_sct:
                for i, monitor in enumerate(temp_sct.monitors[1:], 1):
                    if (monitor["left"] <= cx < monitor["left"] + monitor["width"] and
                            monitor["top"] <= cy < monitor["top"] + monitor["height"]):
                        mon = monitor.copy()
                        mon["monitor_id"] = i
                        return mon
                primary = temp_sct.monitors[1].copy()
                primary["monitor_id"] = 1
                return primary
        except Exception:
            return {"left": 0, "top": 0, "width": 1920, "height": 1080, "monitor_id": 1}

    def write_thumb_slot(slot_idx, thumb_bytes):
        start = slot_idx * slot_size
        end = start + slot_size
        shm.buf[start:end] = thumb_bytes

    # Create thread pool for CPU-intensive operations (thumbnail processing, JPEG encoding)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    interval = 1.0 / max(fps, 0.1)
    loop = asyncio.get_running_loop()

    def process_screenshot(shot_data, timestamp, monitor_id):
        """Process screenshot in thread pool - create thumbnail and optionally encode JPEG"""
        shot, monitor = shot_data

        try:
            # Create PIL image from screenshot
            pil = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

            # Create thumbnail
            w_orig, h_orig = pil.size
            thumb_h_local = max(1, int((thumb_w * h_orig) / w_orig))
            thumb = pil.convert("L").resize((thumb_w, thumb_h_local), resample=Image.BILINEAR)
            thumb_bytes = thumb.tobytes()

            # Pad or crop to match slot size
            if thumb_h_local != thumb_h:
                slot = bytearray(slot_size)
                min_h = min(thumb_h_local, thumb_h)
                for r in range(min_h):
                    src_start = r * thumb_w
                    dst_start = r * thumb_w
                    slot[dst_start:dst_start + thumb_w] = thumb_bytes[src_start:src_start + thumb_w]
                write_bytes = bytes(slot)
            else:
                write_bytes = thumb_bytes

            # Encode full-res JPEG if saving
            full_jpeg_bytes = None
            if save_all_buffer:
                jpeg_buf = io.BytesIO()
                pil.save(jpeg_buf, format="JPEG", quality=JPEG_QUALITY_SAVE)
                full_jpeg_bytes = jpeg_buf.getvalue()

            return write_bytes, full_jpeg_bytes, monitor_id

        except Exception as e:
            print(f"[process_screenshot] error: {e}")
            return None, None, monitor_id

    try:
        with mss.mss(with_cursor=with_cursor) as sct:
            seq = 0
            while not stop_event.is_set():
                loop_start = loop.time()
                timestamp = time.time()

                try:
                    # Fast screenshot capture (this is the bottleneck we want to minimize)
                    monitor = get_active_monitor_bounds_proc()
                    shot = sct.grab(monitor)

                    # Process screenshot in thread pool to avoid blocking the capture loop
                    process_future = loop.run_in_executor(
                        executor,
                        process_screenshot,
                        (shot, monitor),
                        timestamp,
                        monitor.get("monitor_id", 1)
                    )

                    # Create task to handle the result without blocking
                    async def handle_processed_shot(future, ts, seq_num):
                        try:
                            thumb_bytes, jpeg_bytes, mon_id = await future
                            if thumb_bytes is None:
                                return

                            # Write thumbnail to shared memory
                            slot_idx = int(round(ts * fps)) % slots
                            write_thumb_slot(slot_idx, thumb_bytes)

                            # Send metadata to main process
                            try:
                                meta_queue.put((ts, slot_idx, mon_id, len(thumb_bytes)), block=False)
                            except:
                                pass  # Drop if queue full

                            # Send thumbnail to SSIM process
                            try:
                                ssim_queue.put((ts, thumb_bytes, thumb_w, thumb_h, mon_id, slot_idx), block=False)
                            except:
                                pass  # Drop if queue full

                            # Save full-res image if enabled
                            if save_all_buffer and jpeg_bytes and save_queue_async:
                                filename = f"buffer_active_{ts:.6f}.jpg"
                                try:
                                    await save_queue_async.put((ts, jpeg_bytes, "jpeg", filename))
                                except:
                                    pass  # Drop if queue full

                        except Exception as e:
                            print(f"[handle_processed_shot] error: {e}")

                    # Schedule the processing without waiting
                    asyncio.create_task(handle_processed_shot(process_future, timestamp, seq))
                    seq += 1

                except Exception as e:
                    print(f"[async_capture] capture error: {e}")

                # Maintain target FPS
                elapsed = loop.time() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

    finally:
        try:
            shm.close()
            executor.shutdown(wait=False)
        except Exception:
            pass
        # Signal termination
        try:
            meta_queue.put(None)
        except Exception:
            pass


def _run_async_capture_process(meta_queue, save_queue_async, ssim_queue, shm_name, thumb_w, thumb_h, slots, fps, stop_event, save_all_buffer, with_cursor):
    """Wrapper to run the async capture process in a new event loop"""
    async def main():
        await _async_capture_process(
            meta_queue, save_queue_async, ssim_queue, shm_name,
            thumb_w, thumb_h, slots, fps, stop_event, save_all_buffer, with_cursor
        )

    # Create new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main())
    except Exception as e:
        print(f"[async_capture_process] error: {e}")
    finally:
        loop.close()


# ------------------------
# ScreenshotManager with integrated async capture
# ------------------------

class ScreenshotManager:
    def __init__(self, fps=60, buffer_seconds=6.0, save_all_buffer=False, buffer_save_dir=None,
                 log_ssim=True, ssim_log_file=None, thumb_w=THUMB_W, async_workers=4):
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.slots = int(max(1, round(self.fps * self.buffer_seconds)))
        self.thumb_w = thumb_w
        self.thumb_h = int(self.thumb_w * 9 // 16)  # default assume 16:9
        self.slot_size = self.thumb_w * self.thumb_h

        self.save_all_buffer = save_all_buffer
        self.buffer_save_dir = buffer_save_dir
        self.log_ssim = log_ssim
        self.ssim_log_file = ssim_log_file
        self.async_workers = async_workers

        # in-memory buffer for quick retrieval
        self.buffer = deque(maxlen=self.slots)
        self._lock = threading.Lock()

        # shared memory buffer for thumbnails
        self.thumb_shm = None

        # processes and queues
        self._capture_process = None
        self._capture_meta_queue = None
        self._capture_stop_event = None

        # async save components
        self._async_save_process = None
        self._async_save_queue = None

        # SSIM process
        self._ssim_process = None
        self._ssim_queue = None

        # reader thread
        self._reader_thread = None

        # Setup workers
        if self.save_all_buffer:
            if not self.buffer_save_dir:
                raise ValueError("buffer_save_dir must be provided when save_all_buffer=True")
            self._setup_async_save_process()

        if self.log_ssim:
            if not self.ssim_log_file:
                raise ValueError("ssim_log_file must be provided when log_ssim=True")
            self._setup_ssim_process()

    def _setup_async_save_process(self):
        """Setup async save process with multiple workers"""
        def run_async_save_workers(queue, out_dir, num_workers):
            import asyncio

            async def main():
                # Convert multiprocessing queue to asyncio queue
                async_queue = asyncio.Queue(maxsize=1000)

                # Queue reader that feeds from multiprocessing queue to asyncio queue
                async def queue_reader():
                    import queue as _q
                    while True:
                        try:
                            item = queue.get(timeout=1.0)
                            if item is None:
                                # Send shutdown to all workers
                                for _ in range(num_workers):
                                    await async_queue.put(None)
                                break
                            await async_queue.put(item)
                        except _q.Empty:
                            continue
                        except Exception as e:
                            print(f"[queue_reader] error: {e}")

                # Start queue reader
                reader_task = asyncio.create_task(queue_reader())

                # Start save workers
                workers = []
                for i in range(num_workers):
                    worker = asyncio.create_task(_async_save_worker(f"save-worker-{i + 1}", async_queue, Path(out_dir)))
                    workers.append(worker)

                # Wait for all to complete
                await asyncio.gather(reader_task, *workers, return_exceptions=True)

            # Run async main
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(main())
            finally:
                loop.close()

        self._async_save_queue = multiprocessing.Queue(maxsize=SAVE_QUEUE_MAXSIZE)
        self._async_save_process = multiprocessing.Process(
            target=run_async_save_workers,
            args=(self._async_save_queue, self.buffer_save_dir, self.async_workers),
            daemon=True
        )
        self._async_save_process.start()

    def _setup_ssim_process(self):
        self._ssim_queue = multiprocessing.Queue(maxsize=1000)
        self._ssim_process = multiprocessing.Process(
            target=_ssim_worker_process,
            args=(self._ssim_queue, self.ssim_log_file),
            daemon=True
        )
        self._ssim_process.start()

    def start(self):
        # create shared thumb buffer
        self.thumb_shm = ThumbSharedBuffer(slots=self.slots, thumb_w=self.thumb_w, thumb_h=self.thumb_h)
        shm_name = self.thumb_shm.name()

        # meta queue for lightweight metadata
        self._capture_meta_queue = multiprocessing.Queue(maxsize=2048)
        self._capture_stop_event = multiprocessing.Event()

        # start async capture process
        self._capture_process = multiprocessing.Process(
            target=_run_async_capture_process,
            args=(
                self._capture_meta_queue,
                self._async_save_queue if self.save_all_buffer else None,
                self._ssim_queue if self.log_ssim else None,
                shm_name,
                self.thumb_w,
                self.thumb_h,
                self.slots,
                self.fps,
                self._capture_stop_event,
                self.save_all_buffer,
                True  # with_cursor
            ),
            daemon=True
        )
        self._capture_process.start()

        # start reader thread (simplified since SSIM is handled in separate process)
        def reader():
            while True:
                try:
                    item = self._capture_meta_queue.get(timeout=1.0)
                except Exception:
                    if self._capture_stop_event.is_set():
                        break
                    continue

                if item is None:
                    break

                try:
                    timestamp, slot_index, monitor_id, thumb_size = item
                except Exception:
                    continue

                # read thumbnail from shared memory for local buffer
                try:
                    slot_bytes = self.thumb_shm.read_slot(slot_index)
                    arr = np.frombuffer(slot_bytes[:thumb_size], dtype=np.uint8).reshape((self.thumb_h, self.thumb_w))
                    pil_thumb = Image.fromarray(arr, mode="L").convert("RGB")

                    frame_data = {
                        "timestamp": timestamp,
                        "thumb": pil_thumb,
                        "slot_index": slot_index,
                        "monitor_id": monitor_id,
                        "ssim": None  # SSIM computed in separate process
                    }

                    with self._lock:
                        self.buffer.append(frame_data)

                except Exception as e:
                    print(f"[reader] error processing frame: {e}")

        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._reader_thread.start()

    def stop(self):
        # Stop capture process
        try:
            if self._capture_stop_event:
                self._capture_stop_event.set()
            if self._capture_process and self._capture_process.is_alive():
                self._capture_process.join(timeout=3.0)
                if self._capture_process.is_alive():
                    self._capture_process.terminate()
        except Exception as e:
            print(f"[ScreenshotManager.stop] error stopping capture process: {e}")

        # Join reader thread
        try:
            if self._reader_thread and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1.0)
        except Exception:
            pass

        # Cleanup shared memory
        try:
            if self.thumb_shm:
                self.thumb_shm.close()
        except Exception:
            pass

        # Stop async save process
        if self._async_save_process and self._async_save_process.is_alive():
            try:
                self._async_save_queue.put(None, timeout=1.0)
                self._async_save_process.join(timeout=3.0)
                if self._async_save_process.is_alive():
                    self._async_save_process.terminate()
            except Exception as e:
                print(f"[ScreenshotManager.stop] error stopping async save process: {e}")

        # Stop SSIM process
        if self._ssim_process and self._ssim_process.is_alive():
            try:
                self._ssim_queue.put(None, timeout=1.0)
                self._ssim_process.join(timeout=3.0)
                if self._ssim_process.is_alive():
                    self._ssim_process.terminate()
            except Exception as e:
                print(f"[ScreenshotManager.stop] error stopping SSIM process: {e}")

    def take_screenshot_thumbnail(self):
        """Get the most recent thumbnail"""
        with self._lock:
            if not self.buffer:
                return None, None
            f = self.buffer[-1]
            jpeg_buf = io.BytesIO()
            f["thumb"].save(jpeg_buf, format="JPEG", quality=85)
            return jpeg_buf.getvalue(), f["thumb"].size

    def take_virtual_screenshot(self, quality: int = 95):
        """Take an immediate full-resolution screenshot"""
        try:
            cursor_x, cursor_y = self._get_cursor_position()
            with mss.mss(with_cursor=True) as sct:
                monitor = None
                for i, mon in enumerate(sct.monitors[1:], 1):
                    if (mon["left"] <= cursor_x < mon["left"] + mon["width"] and
                            mon["top"] <= cursor_y < mon["top"] + mon["height"]):
                        monitor = mon
                        break
                if monitor is None:
                    monitor = sct.monitors[1]

                shot = sct.grab(monitor)
                pil_img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                jpeg_buffer = io.BytesIO()
                pil_img.save(jpeg_buffer, format="JPEG", quality=quality)
                return jpeg_buffer.getvalue(), pil_img.size
        except Exception as e:
            print(f"[take_virtual_screenshot] error: {e}")
            return None, None

    def _get_cursor_position(self):
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
