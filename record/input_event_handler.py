import time
from pynput import mouse


class InputEventHandler:

    def __init__(self, queue, screenshot_manager, move_interval=1.0):
        self.queue = queue
        self.screenshot_manager = screenshot_manager
        self.move_interval = move_interval
        self._last_move_time = 0.0
        self.mouse_controller = mouse.Controller()

    def on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        x, y = self.mouse_controller.position
        mon = self.screenshot_manager.get_active_monitor(x, y)

        png, size = self.screenshot_manager.take_screenshot_for_monitor(mon)

        if png is None or size is None:
            return
        
        print(f"Screenshot taken for monitor: {mon} for keypress")

        self.queue.enqueue(
            event_type="keyboard_press",
            details={"key": k},
            monitor=mon,
            cursor_pos=[x, y],
            screenshot=(png, size)
        )

    def on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        x, y = self.mouse_controller.position
        mon = self.screenshot_manager.get_active_monitor(x, y)
        png, size = self.screenshot_manager.take_screenshot_for_monitor(mon)

        if png is None or size is None:
            return
        
        print(f"Screenshot taken for monitor: {mon} for keyrelease")

        self.queue.enqueue(
            event_type="keyboard_release",
            details={"key": k},
            monitor=mon,
            cursor_pos=[x, y],
            screenshot=(png, size)
        )

    def on_click(self, x, y, button, pressed):
        mon = self.screenshot_manager.get_active_monitor(x, y)
        png, size = self.screenshot_manager.take_screenshot_for_monitor(mon)

        if png is None or size is None:
            return
        
        print(f"Screenshot taken for monitor: {mon} for click")

        evt = "mouse_down" if pressed else "mouse_up"
        self.queue.enqueue(
            event_type=evt,
            details={"button": str(button)},
            monitor=mon,
            cursor_pos=[x, y],
            screenshot=(png, size)
        )

    def on_move(self, x, y):
        now = time.time()
        if now - self._last_move_time < self.move_interval:
            return
        self._last_move_time = now
        mon = self.screenshot_manager.get_active_monitor(x, y)
        png, size = self.screenshot_manager.take_screenshot_for_monitor(mon)

        if png is None or size is None:
            return
        
        print(f"Screenshot taken for monitor: {mon} for mouse move")

        self.queue.enqueue(
            event_type="mouse_move",
            details={},
            monitor=mon,
            cursor_pos=[x, y],
            screenshot=(png, size)
        )

    def on_scroll(self, x, y, dx, dy):
        mon = self.screenshot_manager.get_active_monitor(x, y)
        png, size = self.screenshot_manager.take_screenshot_for_monitor(mon)

        if png is None or size is None:
            return
        
        print(f"Screenshot taken for monitor: {mon} for mouse scroll")

        self.queue.enqueue(
            event_type="mouse_scroll",
            details={"scroll": [dx, dy]},
            monitor=mon,
            cursor_pos=[x, y],
            screenshot=(png, size)
        )
