import datetime
import threading
from pathlib import Path

from pynput import keyboard, mouse
from record import ScreenshotManager, InputEventHandler, EventQueue, io_worker, poll_worker


stop_event = threading.Event()


SESSION_DIR = Path(__file__).parent.parent / "logs" / f"session_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
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


def main():
    print(f"Session started: {SESSION_DIR}")
    print(f"Screenshots: {SCREENSHOT_DIR}")

    event_queue = EventQueue(maxsize=1024, debouncing_thresholds=DEBOUNCING_THRESHOLDS)
    screenshot_manager = ScreenshotManager()

    threading.Thread(target=io_worker, args=(event_queue, SCREENSHOT_DIR, LOG_FILE), daemon=True).start()
    threading.Thread(target=poll_worker, args=(screenshot_manager, event_queue, 60.0), daemon=True).start()

    h = InputEventHandler(event_queue, screenshot_manager)

    with keyboard.Listener(on_press=h.on_press, on_release=h.on_release) as kl, \
            mouse.Listener(on_click=h.on_click, on_move=h.on_move, on_scroll=h.on_scroll) as ml:
        kl.join()
        ml.join()

    event_queue.queue.join()


if __name__ == "__main__":
    main()
