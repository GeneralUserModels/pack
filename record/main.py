import time
import datetime
import json
import threading
from pathlib import Path
from queue import Empty

from pynput import keyboard, mouse

from record import ScreenshotManager, InputEventHandler, EventQueue
from modules import RawLog

# TODO: factor out
SESSION_DIR = Path(__file__).parent.parent / "logs" / f"session_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}"
SCREENSHOT_DIR = SESSION_DIR / "screenshots"
LOG_FILE = SESSION_DIR / "events.jsonl"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


stop_event = threading.Event()


def io_worker(event_queue: EventQueue):
    while not stop_event.is_set() or not event_queue.empty():
        try:
            raw: RawLog = event_queue.queue.get(timeout=0.1)
        except Empty:
            continue

        if raw.screenshot_bytes is not None:
            shot_path = SCREENSHOT_DIR / f"{raw.timestamp}_{raw.event_type}.jpg"
            with open(shot_path, "wb") as imgf:
                imgf.write(raw.screenshot_bytes)
            raw.screenshot_path = str(shot_path)

        with open(LOG_FILE, "a") as jf:
            json.dump(raw.to_dict(), jf)
            jf.write("\n")

        event_queue.queue.task_done()


def poll_worker(screenshot_manager: ScreenshotManager, event_queue: EventQueue, interval: float = 1.0):
    try:
        while not stop_event.is_set():
            x, y = mouse.Controller().position

            active_mon = screenshot_manager.get_active_monitor(x, y)

            jpg_bytes, (w, h) = screenshot_manager.take_virtual_screenshot()

            event_queue.enqueue(
                event_type="poll",
                details={},
                cursor_pos=[x, y],
                monitor=active_mon,
                screenshot=(jpg_bytes, (w, h))
            )

            time.sleep(interval)
    finally:
        screenshot_manager.close()


def main():
    print(f"Session started: {SESSION_DIR}")
    print(f"Screenshots:      {SCREENSHOT_DIR}")

    event_queue = EventQueue(maxsize=1024)
    screenshot_manager = ScreenshotManager()

    threading.Thread(target=io_worker, args=(event_queue,), daemon=True).start()
    # threading.Thread(target=poll_worker, args=(screenshot_manager, event_queue, 1.0), daemon=True).start()

    h = InputEventHandler(event_queue, screenshot_manager)

    with keyboard.Listener(on_press=h.on_press, on_release=h.on_release) as kl, \
            mouse.Listener(on_click=h.on_click, on_move=h.on_move, on_scroll=h.on_scroll) as ml:
        kl.join()
        ml.join()

    event_queue.queue.join()


if __name__ == "__main__":
    main()
