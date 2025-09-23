import json
import threading
from pathlib import Path
from queue import Empty
from record import EventQueue
from modules import RawLog

stop_event = threading.Event()


def io_worker(event_queue: EventQueue, screenshot_dir: Path, log_file: Path):
    while not stop_event.is_set() or not event_queue.empty():
        try:
            raw: 'RawLog' = event_queue.queue.get(timeout=0.1)
        except Empty:
            continue

        if raw.screenshot_bytes is not None:
            shot_path = screenshot_dir / f"{raw.timestamp}_{raw.event_type}.jpg"
            with open(shot_path, "wb") as imgf:
                imgf.write(raw.screenshot_bytes)
            raw.screenshot_path = str(shot_path)

        with open(log_file, "a") as jf:
            json.dump(raw.to_dict(), jf)
            jf.write("\n")

        event_queue.queue.task_done()
