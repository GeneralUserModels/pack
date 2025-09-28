import threading
from pathlib import Path
import asyncio
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

BUFFER_FPS = 60                  # target capture FPS
BUFFER_SECONDS = 6.0             # how many seconds of thumbnails to keep
LOG_SSIM = True                  # whether to compute & log SSIM
THUMB_W = 320                    # thumbnail width (keeps aspect ratio)
JPEG_QUALITY_SAVE = 70           # quality when saving full-res (tune for speed)
SAVE_QUEUE_MAXSIZE = 512         # max queue size for save worker (bounded to avoid OOM)

# ------------------------
# SSIM worker process (separate from save workers for better performance)
# ------------------------


def _ssim_worker_process(ssim_queue, log_file, out_queue=None):
    """Dedicated SSIM computation and logging process."""
    from pathlib import Path
    import queue as _q
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    import time as _time

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prev_thumb = None

    with open(log_path, "w") as f:
        while True:
            try:
                item = ssim_queue.get(timeout=1.0)
                if item is None:
                    break

                timestamp, thumb_bytes, thumb_w, thumb_h, monitor_id, slot_index = item

                try:
                    arr = np.frombuffer(thumb_bytes, dtype=np.uint8).reshape((thumb_h, thumb_w))
                except Exception as e:
                    print(f"[ssim_worker] error reshaping thumb: {e}")
                    continue

                ssim_value = None
                if prev_thumb is not None:
                    try:
                        data_range = float(arr.max() - arr.min()) if arr.max() != arr.min() else 1.0
                        ssim_value = float(ssim(prev_thumb, arr, data_range=data_range))

                        ssim_entry = {
                            "timestamp": _time.time() if isinstance(timestamp, float) else timestamp,
                            "formatted_timestamp": _time.strftime("%Y-%m-%d_%H-%M-%S-%f", _time.localtime(timestamp)),
                            "unix_timestamp": float(timestamp),
                            "monitor_id": monitor_id,
                            "slot_index": slot_index,
                            "ssim_similarity": float(ssim_value),
                            "thumb_w": thumb_w,
                            "thumb_h": thumb_h
                        }
                        json.dump(ssim_entry, f)
                        f.write("\n")
                        f.flush()

                        # forward to main process if requested
                        if out_queue is not None:
                            try:
                                out_queue.put_nowait(ssim_entry)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"[ssim_worker] SSIM computation error: {e}")

                prev_thumb = arr.copy()

            except _q.Empty:
                continue
            except Exception as e:
                print(f"[ssim_worker] error: {e}")
                continue
# ------------------------
# Async save worker
# ------------------------


# inside screenshot_manager.py - updated async save worker
async def _async_save_worker(name: str, queue: asyncio.Queue, out_dir: Path):
    import aiofiles
    import os

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")

            # Write to tmp file first (atomic rename later)
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(image_bytes)
                # ensure file is flushed/closed by exiting context

            # Atomically move tmp -> final
            os.replace(str(tmp_path), str(filepath))

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

async def _async_capture_process(
    meta_queue,
    save_queue_async,
    ssim_queue,
    shm_name,
    thumb_w,
    thumb_h,
    slots,
    fps,
    stop_event,
    save_all_buffer=True,
    with_cursor=True,
    export_request_queue=None,
    export_done_queue=None
):
    """
    Async capture process (updated):
      - supports on-demand export requests via export_request_queue (items: (req_id, target_ts, out_path))
      - signals completions on export_done_queue with tuples (req_id, target_ts, out_path, success_bool)
      - encodes full-res JPEG only when save_all_buffer=True or when there are pending exports
      - writes exports atomically (tmp -> final) using aiofiles + os.replace
    """
    import concurrent.futures
    import queue as _q
    import aiofiles
    import os
    import time

    # reopen shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    slot_size = thumb_w * thumb_h

    # pending export requests (serviced when a matching full-res is available)
    pending_exports: List[Dict[str, Any]] = []  # elements: {"req_id":..., "target_ts":..., "out_path":...}
    export_tolerance = 0.5  # seconds tolerance when matching export request to a captured frame

    def drain_export_requests():
        """Move items from export_request_queue into pending_exports (non-blocking)."""
        if export_request_queue is None:
            return
        try:
            while True:
                req = export_request_queue.get_nowait()
                if req is None:
                    break
                try:
                    req_id, target_ts, out_path = req
                    pending_exports.append({"req_id": req_id, "target_ts": float(target_ts), "out_path": str(out_path)})
                except Exception:
                    continue
        except Exception:
            # likely queue empty
            pass

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

    # Create thread pool for CPU-intensive operations (thumbnail processing, optional JPEG encoding)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    interval = 1.0 / max(fps, 0.1)
    loop = asyncio.get_running_loop()

    def _put_into_save_queue(item):
        """
        Put item into save_queue_async whether it's an asyncio.Queue or a multiprocessing.Queue.
        item: (timestamp, image_bytes, format_type, filename)
        """
        if save_queue_async is None:
            return
        # If it's an asyncio.Queue, it will have a coroutine .put
        try:
            # try awaiting in executor if not coroutine-available here (we are not async in this helper)
            # We'll return an awaitable by using loop.run_in_executor to call blocking put
            if hasattr(save_queue_async, "put") and asyncio.iscoroutinefunction(save_queue_async.put):
                # This branch is unlikely because we are in a thread context; the caller will await.
                # But keep it for completeness.
                return loop.create_task(save_queue_async.put(item))
        except Exception:
            pass

        # If it has put_nowait (likely multiprocessing.Queue or similar) use that in executor to avoid blocking event loop
        try:
            if hasattr(save_queue_async, "put_nowait"):
                save_queue_async.put_nowait(item)
                return None
        except Exception:
            # fallback to blocking put in executor
            try:
                loop.run_in_executor(None, save_queue_async.put, item)
                return None
            except Exception:
                return None

    def process_screenshot(shot_data, timestamp, monitor_id, encode_full: bool):
        """Process screenshot in thread pool - create thumbnail and optionally encode JPEG"""
        shot, monitor = shot_data

        try:
            # Create PIL image from screenshot
            pil = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

            # Create thumbnail (grayscale)
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

            # Encode full-res JPEG only if requested
            full_jpeg_bytes = None
            if encode_full:
                try:
                    jpeg_buf = io.BytesIO()
                    # Use reasonable quality (configurable elsewhere)
                    pil.save(jpeg_buf, format="JPEG", quality=JPEG_QUALITY_SAVE)
                    full_jpeg_bytes = jpeg_buf.getvalue()
                except Exception as e:
                    print(f"[process_screenshot] jpeg encode error: {e}")
                    full_jpeg_bytes = None

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
                    # Drain export requests before capture so we know whether to encode full-res for this frame
                    drain_export_requests()
                    encode_full_for_this_frame = save_all_buffer or (len(pending_exports) > 0)

                    monitor = get_active_monitor_bounds_proc()
                    shot = sct.grab(monitor)

                    # Process screenshot in thread pool to avoid blocking the capture loop
                    process_future = loop.run_in_executor(
                        executor,
                        process_screenshot,
                        (shot, monitor),
                        timestamp,
                        monitor.get("monitor_id", 1),
                        encode_full_for_this_frame
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

                            # Send metadata to main process (non-blocking)
                            try:
                                meta_queue.put((ts, slot_idx, mon_id, len(thumb_bytes)), block=False)
                            except Exception:
                                pass  # Drop if queue full

                            # Send thumbnail to SSIM process (non-blocking)
                            try:
                                if ssim_queue is not None:
                                    ssim_queue.put((ts, thumb_bytes, thumb_w, thumb_h, mon_id, slot_idx), block=False)
                            except Exception:
                                pass  # Drop if queue full

                            # Save full-res image if continuous saving is enabled
                            if save_all_buffer and jpeg_bytes is not None and save_queue_async:
                                # Put into save queue (supports asyncio.Queue or multiprocessing.Queue)
                                try:
                                    # If save_queue_async is asyncio.Queue, awaitable returned from _put_into_save_queue
                                    task = _put_into_save_queue((ts, jpeg_bytes, "jpeg", f"buffer_active_{ts:.6f}.jpg"))
                                    if task is not None:
                                        # if _put_into_save_queue returned a coroutine/task, await it
                                        await task
                                except Exception:
                                    pass  # Drop if queue full

                            # Service any pending export requests whose target_ts is near this ts
                            if jpeg_bytes is not None and pending_exports:
                                satisfied = []
                                for req in list(pending_exports):
                                    try:
                                        if abs(req["target_ts"] - ts) <= export_tolerance:
                                            out_path = Path(req["out_path"])
                                            # ensure parent exists
                                            out_path.parent.mkdir(parents=True, exist_ok=True)
                                            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                                            try:
                                                # write atomically with aiofiles
                                                async with aiofiles.open(tmp_path, "wb") as af:
                                                    await af.write(jpeg_bytes)
                                                # atomic replace
                                                os.replace(str(tmp_path), str(out_path))
                                                success = True
                                            except Exception as e:
                                                # fallback attempt: blocking write via run_in_executor
                                                try:
                                                    def blocking_write(p, data):
                                                        p = Path(p)
                                                        tmp = p.with_suffix(p.suffix + ".tmp")
                                                        with open(tmp, "wb") as f:
                                                            f.write(data)
                                                        os.replace(str(tmp), str(p))
                                                    await loop.run_in_executor(None, blocking_write, str(out_path), jpeg_bytes)
                                                    success = True
                                                except Exception as e2:
                                                    success = False
                                                    # print error for debugging
                                                    print(f"[async_capture export] failed to write export {out_path}: {e2}")

                                            # inform requester via export_done_queue if available
                                            if export_done_queue is not None:
                                                try:
                                                    export_done_queue.put_nowait((req["req_id"], req["target_ts"], str(out_path), success))
                                                except Exception:
                                                    # if put_nowait not available, try blocking put in executor
                                                    try:
                                                        loop.run_in_executor(None, export_done_queue.put, (req["req_id"], req["target_ts"], str(out_path), success))
                                                    except Exception:
                                                        pass

                                            satisfied.append(req)
                                    except Exception as e:
                                        # ignore per-request errors
                                        print(f"[async_capture export] per-request error: {e}")
                                # remove satisfied requests
                                for req in satisfied:
                                    try:
                                        pending_exports.remove(req)
                                    except Exception:
                                        pass

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
        try:
            meta_queue.put(None)
        except Exception:
            pass


def _run_async_capture_process(meta_queue, save_queue_async, ssim_queue, shm_name, thumb_w, thumb_h, slots, fps, stop_event, save_all_buffer, with_cursor, export_request_queue=None, export_done_queue=None):
    async def main():
        await _async_capture_process(
            meta_queue, save_queue_async, ssim_queue, shm_name,
            thumb_w, thumb_h, slots, fps, stop_event, save_all_buffer, with_cursor,
            export_request_queue, export_done_queue
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
        self.ssim_buffer = deque(maxlen=self.slots)
        self._ssim_out_queue = None
        self._export_request_queue = None
        self._export_done_queue = None

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
        self._ssim_out_queue = multiprocessing.Queue(maxsize=1000)
        self._ssim_process = multiprocessing.Process(
            target=_ssim_worker_process,
            args=(self._ssim_queue, self.ssim_log_file, self._ssim_out_queue),
            daemon=True
        )
        self._ssim_process.start()

    def request_fullres_save(self, target_ts: float, out_path: str, timeout: float = 2.0):
        """
        Request the capture process to export the nearest full-res JPEG to out_path.
        Returns True on success (export_done received), False on timeout/failure.
        """
        if self._export_request_queue is None or self._export_done_queue is None:
            return False

        # generate a request id (can be timestamp + random)
        req_id = f"req_{time.time()}_{int(threading.get_ident())}"
        try:
            self._export_request_queue.put_nowait((req_id, float(target_ts), str(out_path)))
        except Exception:
            # queue full or not available
            return False

        # wait for done (simple blocking poll with timeout)
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                item = self._export_done_queue.get_nowait()
            except Exception:
                # nothing yet
                time.sleep(0.02)
                continue
            try:
                got_id, got_ts, path, success = item
                if got_id == req_id:
                    return bool(success)
            except Exception:
                continue
        return False

    def start(self):
        # create shared thumb buffer
        self.thumb_shm = ThumbSharedBuffer(slots=self.slots, thumb_w=self.thumb_w, thumb_h=self.thumb_h)
        shm_name = self.thumb_shm.name()

        # meta queue for lightweight metadata
        self._capture_meta_queue = multiprocessing.Queue(maxsize=2048)
        self._capture_stop_event = multiprocessing.Event()

        # start async capture process
        self._export_request_queue = multiprocessing.Queue(maxsize=256)
        self._export_done_queue = multiprocessing.Queue(maxsize=256)

        # start async capture process
        self._capture_process = multiprocessing.Process(
            target=_run_async_capture_process,
            args=(
                self._capture_meta_queue,
                self._async_save_queue if self.save_all_buffer else self._async_save_queue,
                self._ssim_queue if self.log_ssim else None,
                shm_name,
                self.thumb_w,
                self.thumb_h,
                self.slots,
                self.fps,
                self._capture_stop_event,
                self.save_all_buffer,
                True,  # with_cursor
                self._export_request_queue,
                self._export_done_queue,
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

        def ssim_reader():
            if not self._ssim_out_queue:
                return
            import queue as _q
            while True:
                try:
                    entry = self._ssim_out_queue.get(timeout=1.0)
                except _q.Empty:
                    if self._capture_stop_event.is_set():
                        break
                    continue
                if entry is None:
                    break
                try:
                    # append to in-memory ssim buffer (thread-safe enough since only this thread writes)
                    self.ssim_buffer.append(entry)
                except Exception as e:
                    print(f"[ssim_reader] error: {e}")

        self._ssim_reader_thread = threading.Thread(target=ssim_reader, daemon=True)
        if self._ssim_out_queue is not None:
            self._ssim_reader_thread.start()

    def find_buffer_fullres(self, ts: float):
        """
        Try to find an existing full-res buffer file saved by the async save worker.
        Filenames are 'buffer_active_{ts:.6f}.jpg'. We attempt exact match, then nearest.
        """
        if not self.save_all_buffer or not self.buffer_save_dir:
            return None
        # exact prefix (string formatting may differ slightly)
        prefix = f"buffer_active_{ts:.6f}"
        candidates = list(Path(self.buffer_save_dir).glob(f"{prefix}*.jpg"))
        if candidates:
            return str(sorted(candidates)[-1])
        # fallback: find nearest by timestamp encoded in filenames
        # list all buffer files and choose nearest numeric timestamp
        try:
            files = list(Path(self.buffer_save_dir).glob("buffer_active_*.jpg"))
            nearest = None
            best_diff = None
            for p in files:
                name = p.name.replace("buffer_active_", "").replace(".jpg", "")
                try:
                    file_ts = float(name)
                    diff = abs(file_ts - ts)
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        nearest = p
                except Exception:
                    continue
            if nearest:
                return str(nearest)
        except Exception:
            pass
        return None

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
