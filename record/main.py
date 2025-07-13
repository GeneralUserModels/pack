import time
import datetime
import json
import threading
from pathlib import Path
from queue import Empty
from typing import Dict

from pynput import keyboard, mouse

from record import ScreenshotManager, InputEventHandler, EventQueue
from modules import RawLog

# TODO: factor out
SESSION_DIR = Path(__file__).parent.parent / "logs" / f"session_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}"
SCREENSHOT_DIR = SESSION_DIR / "screenshots"
LOG_FILE = SESSION_DIR / "events.jsonl"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

DEBOUNCING_THRESHOLDS = {
    "mouse_move": 2.0,
    "mouse_scroll": 3.0,
    "keyboard_press": 3.0,
    "keyboard_release": 3.0,
}

stop_event = threading.Event()


def io_worker(event_queue: EventQueue):
    last_saved_times: Dict[str, float] = {}
    pending_events: Dict[str, tuple] = {}

    def should_save_screenshot(event_type: str, current_time: float, debouncing_thresholds: Dict[str, float]) -> tuple[bool, bool]:
        if debouncing_thresholds is None or event_type not in debouncing_thresholds:
            return True, False

        threshold = debouncing_thresholds[event_type]
        last_saved = last_saved_times.get(event_type, 0)

        if last_saved == 0:
            return True, True

        time_since_last = current_time - last_saved
        should_save = time_since_last >= threshold

        return should_save, False

    def save_screenshot_and_log(raw: 'RawLog', current_time: float):
        if raw.screenshot_bytes is not None:
            shot_path = SCREENSHOT_DIR / f"{raw.timestamp}_{raw.event_type}.jpg"
            with open(shot_path, "wb") as imgf:
                imgf.write(raw.screenshot_bytes)
            raw.screenshot_path = str(shot_path)

        with open(LOG_FILE, "a") as jf:
            json.dump(raw.to_dict(), jf)
            jf.write("\n")

        if raw.screenshot_bytes is not None:
            last_saved_times[raw.event_type] = current_time

    def process_pending_events():
        current_time = time.time()
        events_to_process = []

        for event_type, (raw, event_time) in pending_events.items():
            if current_time - event_time >= event_queue.debouncing_thresholds.get(event_type, 0):
                events_to_process.append((event_type, raw, event_time))

        for event_type, raw, event_time in events_to_process:
            save_screenshot_and_log(raw, event_time)
            del pending_events[event_type]

    while not stop_event.is_set() or not event_queue.empty():
        try:
            raw: 'RawLog' = event_queue.queue.get(timeout=0.1)
        except Empty:
            process_pending_events()
            continue

        current_time = time.time()
        should_save, is_first = should_save_screenshot(raw.event_type, current_time, event_queue.debouncing_thresholds)

        if should_save:
            print("should save now")
            save_screenshot_and_log(raw, current_time)

            if raw.event_type in pending_events:
                with open(LOG_FILE, "a") as jf:
                    json.dump(pending_events[raw.event_type][0].to_dict(), jf)
                    jf.write("\n")
                del pending_events[raw.event_type]
        else:
            print("should save later")
            pending_events[raw.event_type] = (raw, current_time)

        event_queue.queue.task_done()

    for event_type, (raw, event_time) in pending_events.items():
        save_screenshot_and_log(raw, event_time)


def poll_worker(screenshot_manager: ScreenshotManager, event_queue: EventQueue, interval: float = 60.0):
    try:
        while not stop_event.is_set():
            x, y = mouse.Controller().position

            active_mon = screenshot_manager.get_active_monitor(x, y)

            png, size = screenshot_manager.take_screenshot_for_monitor(active_mon)

            event_queue.enqueue(
                event_type="poll",
                details={},
                cursor_pos=[x, y],
                monitor=active_mon,
                screenshot=(png, size)
            )

            time.sleep(interval)
    finally:
        screenshot_manager.close()


def main():
    print(f"Session started: {SESSION_DIR}")
    print(f"Screenshots: {SCREENSHOT_DIR}")

    event_queue = EventQueue(maxsize=1024, debouncing_thresholds=DEBOUNCING_THRESHOLDS)
    screenshot_manager = ScreenshotManager()

    threading.Thread(target=io_worker, args=(event_queue,), daemon=True).start()
    threading.Thread(target=poll_worker, args=(screenshot_manager, event_queue, 60.0), daemon=True).start()

    h = InputEventHandler(event_queue, screenshot_manager)

    with keyboard.Listener(on_press=h.on_press, on_release=h.on_release) as kl, \
            mouse.Listener(on_click=h.on_click, on_move=h.on_move, on_scroll=h.on_scroll) as ml:
        kl.join()
        ml.join()

    event_queue.queue.join()


if __name__ == "__main__":
    main()
