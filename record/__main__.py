import datetime
import threading
from pathlib import Path

from pynput import keyboard, mouse
from record import EventQueue, io_worker, poll_worker, ScreenshotManager, InputEventHandler


stop_event = threading.Event()

SESSION_DIR = Path(__file__).parent.parent / "logs" / f"session_v2_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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

BUFFER_FPS = 60
BUFFER_SECONDS = 3.0
LOOKBACK_MS = 100
FORWARD_DELAY_MS = 100


def main():
    print(f"Session started: {SESSION_DIR}")
    print(f"Screenshots: {SCREENSHOT_DIR}")
    print(f"Using buffered screenshots at {BUFFER_FPS} FPS, keeping {BUFFER_SECONDS}s in memory")

    event_queue = EventQueue(maxsize=1024, debouncing_thresholds=DEBOUNCING_THRESHOLDS)
    screenshot_manager = ScreenshotManager(fps=BUFFER_FPS, buffer_seconds=BUFFER_SECONDS)

    screenshot_manager.start()
    print("Buffered screenshot capture started")

    threading.Thread(target=io_worker, args=(event_queue, SCREENSHOT_DIR, LOG_FILE), daemon=True).start()
    threading.Thread(target=poll_worker, args=(screenshot_manager, event_queue, 60.0), daemon=True).start()

    h = InputEventHandler(
        event_queue,
        screenshot_manager,
        move_interval=1.0,
        lookback_ms=LOOKBACK_MS,
        forward_delay_ms=FORWARD_DELAY_MS
    )

    print("Starting input listeners...")

    try:
        with keyboard.Listener(on_press=h.on_press, on_release=h.on_release) as kl, \
                mouse.Listener(on_click=h.on_click, on_move=h.on_move, on_scroll=h.on_scroll) as ml:

            print("Input listeners active. Press Ctrl+C to stop.")

            try:
                while True:
                    if stop_event.wait(timeout=1.0):
                        break
            except KeyboardInterrupt:
                print("\nReceived interrupt signal...")
                stop_event.set()

            print("Stopping listeners...")

    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        print("Cleaning up...")
        screenshot_manager.stop()

        try:
            event_queue.queue.join()
        except Exception as e:
            print(f"Error while waiting for queue to empty: {e}")

        print("Cleanup complete.")


if __name__ == "__main__":
    main()
