import datetime
import threading
from pathlib import Path
from pynput import keyboard, mouse
from record import EventQueue, io_worker, poll_worker, InputEventHandler, ScreenshotManager

stop_event = threading.Event()
SESSION_DIR = Path(__file__).parent.parent / "logs" / f"session_v2_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
SCREENSHOT_DIR = SESSION_DIR / "screenshots"
BUFFER_SCREENSHOTS_DIR = SESSION_DIR / "buffer_screenshots"
LOG_FILE = SESSION_DIR / "events.jsonl"
SSIM_LOG_FILE = SESSION_DIR / "img_similarities.jsonl"

SESSION_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

DEBOUNCING_THRESHOLDS = {
    "mouse_move": 2.0,
    "mouse_scroll": 3.0,
    "keyboard_press": 3.0,
    "keyboard_release": 3.0,
}

BUFFER_FPS = 6
BUFFER_SECONDS = 3.0
LOOKBACK_MS = 100
FORWARD_DELAY_MS = 100
SAVE_ALL_BUFFER = True
LOG_SSIM = True


def main():
    print(f"Session started: {SESSION_DIR}")
    print(f"Screenshots: {SCREENSHOT_DIR}")
    print(f"Event log: {LOG_FILE}")

    if SAVE_ALL_BUFFER:
        BUFFER_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Buffer screenshots will be saved to: {BUFFER_SCREENSHOTS_DIR}")

    if LOG_SSIM:
        print(f"SSIM similarities will be logged to: {SSIM_LOG_FILE}")

    print(f"Using buffered screenshots at {BUFFER_FPS} FPS, keeping {BUFFER_SECONDS}s in memory")

    event_queue = EventQueue(maxsize=1024, debouncing_thresholds=DEBOUNCING_THRESHOLDS)

    # Create enhanced screenshot manager with SSIM logging
    screenshot_manager = ScreenshotManager(
        fps=BUFFER_FPS,
        buffer_seconds=BUFFER_SECONDS,
        save_all_buffer=SAVE_ALL_BUFFER,
        buffer_save_dir=BUFFER_SCREENSHOTS_DIR if SAVE_ALL_BUFFER else None,
        log_ssim=LOG_SSIM,
        ssim_log_file=SSIM_LOG_FILE if LOG_SSIM else None
    )

    screenshot_manager.start()
    print("Buffered screenshot capture with SSIM logging started")

    # Start worker threads
    threading.Thread(target=io_worker, args=(event_queue, SCREENSHOT_DIR, LOG_FILE), daemon=True).start()
    threading.Thread(target=poll_worker, args=(screenshot_manager, event_queue, 60.0), daemon=True).start()

    # Create input event handler
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
            print("SSIM similarities are being calculated and logged in real-time...")

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
        print(f"SSIM log saved to: {SSIM_LOG_FILE}")


if __name__ == "__main__":
    main()
