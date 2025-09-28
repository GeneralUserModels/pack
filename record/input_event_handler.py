import time
from datetime import datetime
from pynput import mouse


class InputEventHandler:
    def __init__(self, queue, screenshot_manager, move_interval=0.1):
        self.queue = queue
        self.screenshot_manager = screenshot_manager
        self.move_interval = move_interval
        self._last_move_time = 0.0
        self.mouse_controller = mouse.Controller()

    def on_press(self, key):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        try:
            k = key.char
        except AttributeError:
            k = str(key)

        x, y = self.mouse_controller.position

        self._save_log(
            event_type="keyboard_press",
            details={"key": k},
            cursor_pos=[x, y],
            timestamp=ts
        )

    def on_release(self, key):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        try:
            k = key.char
        except AttributeError:
            k = str(key)

        x, y = self.mouse_controller.position
        self._save_log(
            event_type="keyboard_release",
            details={"key": k},
            cursor_pos=[x, y],
            timestamp=ts
        )

    def on_click(self, x, y, button, pressed):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

        evt = "mouse_down" if pressed else "mouse_up"
        print(f"Pressed {pressed} for button {button} at ({x}, {y})")
        self._save_log(
            event_type=evt,
            details={"button": str(button), "pressed": pressed},
            cursor_pos=[x, y],
            timestamp=ts
        )

    def on_move(self, x, y):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        now = time.time()
        if now - self._last_move_time < self.move_interval:
            return
        self._last_move_time = now

        self._save_log(
            event_type="mouse_move",
            details={},
            cursor_pos=[x, y],
            timestamp=ts
        )

    def on_scroll(self, x, y, dx, dy):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self._save_log(
            event_type="mouse_scroll",
            details={"dx": dx, "dy": dy},
            cursor_pos=[x, y],
            timestamp=ts
        )

    def _save_log(self, event_type, details, cursor_pos, timestamp):
        self.queue.enqueue(
            event_type=event_type,
            details=details,
            cursor_pos=cursor_pos,
            screenshot=None,
            timestamp=timestamp
        )
