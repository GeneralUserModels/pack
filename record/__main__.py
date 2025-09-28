import argparse
import datetime
import threading
from pathlib import Path
from record import EventQueue, io_worker, InputEventHandler, ScreenshotManager
from record.save_logic import SaveDeciderWorker

stop_event = threading.Event()
SESSION_DIR = Path(__file__).parent.parent / "logs" / f"session_v3_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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


def main(write_all_buffer: bool = False):
    print(f"Session started: {SESSION_DIR}")
    print(f"Screenshots directory: {SCREENSHOT_DIR}")
    print(f"Event log: {LOG_FILE}")
    if SAVE_ALL_BUFFER:
        BUFFER_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Buffer screenshots will be saved to: {BUFFER_SCREENSHOTS_DIR}")
    if LOG_SSIM:
        print(f"SSIM will be logged to: {SSIM_LOG_FILE}")
    print(f"Buffered screenshots at {BUFFER_FPS} FPS, keeping {BUFFER_SECONDS}s in memory")

    # Event queue (also holds in-memory time-buffer)
    event_queue = EventQueue(maxsize=4096, buffer_seconds=BUFFER_SECONDS)

    # Screenshot manager provides:
    # - screenshot_manager.buffer: deque of last frames
    # - screenshot_manager.ssim_buffer: deque of computed ssim entries (forwarded from ssim worker)
    screenshot_manager = ScreenshotManager(
        fps=BUFFER_FPS,
        buffer_seconds=BUFFER_SECONDS,
        save_all_buffer=write_all_buffer,
        buffer_save_dir=str(BUFFER_SCREENSHOTS_DIR) if SAVE_ALL_BUFFER else None,
        log_ssim=LOG_SSIM,
        ssim_log_file=str(SSIM_LOG_FILE) if LOG_SSIM else None,
        thumb_w=THUMB_W
    )

    screenshot_manager.start()
    print("ScreenshotManager started (capture + ssim processes)")

    # start io worker (persist events/screenshots as before)
    threading.Thread(target=io_worker, args=(event_queue, SCREENSHOT_DIR, LOG_FILE), daemon=True).start()

    # input handlers which will enqueue events in event_queue
    h = InputEventHandler(event_queue, screenshot_manager)

    # start the SaveDecider worker
    decider = SaveDeciderWorker(screenshot_manager, event_queue, SCREENSHOT_DIR, stop_event)
    decider.start()
    print("SaveDeciderWorker started (heuristic-driven saves)")

    # start input listeners
    from pynput import keyboard, mouse
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
                print("\nReceived interrupt, shutting down...")
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
    parser = argparse.ArgumentParser(description="Record user input events and screenshots.")
    parser.add_argument('-b', '--write-all-buffer', action='store_true', help="Do not save buffered screenshots.")
    args = parser.parse_args()
    main(args.write_all_buffer)
