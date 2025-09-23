import datetime
import threading
from pathlib import Path
from pynput import keyboard, mouse
from record import EventQueue, io_worker, InputEventHandler, ScreenshotManager

stop_event = threading.Event()
SESSION_DIR = Path(__file__).parent.parent / "logs" / f"session_v2_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
SCREENSHOT_DIR = SESSION_DIR / "screenshots"
BUFFER_SCREENSHOTS_DIR = SESSION_DIR / "buffer_screenshots"
LOG_FILE = SESSION_DIR / "events.jsonl"
SSIM_LOG_FILE = SESSION_DIR / "img_similarities.jsonl"

SESSION_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

BUFFER_FPS = 30
BUFFER_SECONDS = 6.0
SAVE_ALL_BUFFER = True
LOG_SSIM = True
THUMB_W = 320


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

    event_queue = EventQueue(maxsize=1024)

    # Create the ScreenshotManager with the capture-process enabled for reliable 60 FPS
    screenshot_manager = ScreenshotManager(
        fps=BUFFER_FPS,
        buffer_seconds=BUFFER_SECONDS,
        save_all_buffer=SAVE_ALL_BUFFER,
        buffer_save_dir=BUFFER_SCREENSHOTS_DIR if SAVE_ALL_BUFFER else None,
        log_ssim=LOG_SSIM,
        ssim_log_file=SSIM_LOG_FILE if LOG_SSIM else None,
        thumb_w=THUMB_W
    )

    screenshot_manager.start()
    print("Buffered screenshot capture with SSIM logging started")

    threading.Thread(target=io_worker, args=(event_queue, SCREENSHOT_DIR, LOG_FILE), daemon=True).start()

    h = InputEventHandler(event_queue, screenshot_manager)

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
            # if the io_worker uses event_queue.queue.join()
            event_queue.queue.join()
        except Exception as e:
            print(f"Error while waiting for queue to empty: {e}")
        print("Cleanup complete.")
        print(f"SSIM log saved to: {SSIM_LOG_FILE}")


if __name__ == "__main__":
    # On Windows, multiprocessing uses spawn and will re-import this module, so we must keep
    # top-level side-effects minimal and only run main here.
    main()
