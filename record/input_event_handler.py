import time
import threading
from pynput import mouse


class InputEventHandler:
    def __init__(self, queue, screenshot_manager, move_interval=1.0, lookback_ms=50, forward_delay_ms=30):
        self.queue = queue
        self.screenshot_manager = screenshot_manager
        self.move_interval = move_interval
        self.lookback_ms = lookback_ms
        self.forward_delay_ms = forward_delay_ms
        self._last_move_time = 0.0
        self.mouse_controller = mouse.Controller()

    def _capture_and_enqueue(self, event_type, details, mon, cursor_pos):
        png, size = self.screenshot_manager.take_screenshot_for_monitor(
            mon, quality=95, lookback_ms=0
        )

        if png is not None:
            print(f"Buffered screenshot retrieved for monitor {mon.get('monitor_id', '?')} for {event_type}")
        else:
            print(f"No buffered screenshot available for monitor {mon.get('monitor_id', '?')} for {event_type}")

        self.queue.enqueue(
            event_type=event_type,
            details=details,
            monitor=mon,
            cursor_pos=cursor_pos,
            screenshot=(png, size) if png is not None else None
        )

    def _schedule_delayed_capture(self, delay_ms, event_type, details, mon, cursor_pos):
        t = threading.Timer(delay_ms / 1000.0, self._capture_and_enqueue,
                            args=(event_type, details, mon, cursor_pos))
        t.daemon = True
        t.start()

    def on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)

        x, y = self.mouse_controller.position
        mon = self.screenshot_manager.get_active_monitor(x, y)

        png, size = self.screenshot_manager.take_screenshot_for_monitor(
            mon, quality=95, lookback_ms=self.lookback_ms
        )

        if png is not None:
            print(f"Buffered screenshot retrieved for monitor {mon.get('monitor_id', '?')} for keypress: {key}")
        else:
            print(f"No buffered screenshot available for monitor {mon.get('monitor_id', '?')} for keypress")

        self.queue.enqueue(
            event_type="keyboard_press",
            details={"key": k},
            monitor=mon,
            cursor_pos=[x, y],
            screenshot=(png, size) if png is not None else None
        )

    def on_release(self, key):
        """For key release: wait forward_delay_ms (non-blocking) then capture with lookback_ms=0."""
        try:
            k = key.char
        except AttributeError:
            k = str(key)

        x, y = self.mouse_controller.position
        mon = self.screenshot_manager.get_active_monitor(x, y)

        self._schedule_delayed_capture(
            self.forward_delay_ms,
            event_type="keyboard_release",
            details={"key": k},
            mon=mon,
            cursor_pos=[x, y]
        )

    def on_click(self, x, y, button, pressed):
        mon = self.screenshot_manager.get_active_monitor(x, y)

        evt = "mouse_down" if pressed else "mouse_up"
        print(f"Pressed {pressed} for button {button} at ({x}, {y}) on monitor {mon.get('monitor_id', '?')}")

        if not pressed:
            self._schedule_delayed_capture(
                self.forward_delay_ms,
                event_type=evt,
                details={"button": str(button)},
                mon=mon,
                cursor_pos=[x, y]
            )
        else:
            png, size = self.screenshot_manager.take_screenshot_for_monitor(
                mon, quality=95, lookback_ms=self.lookback_ms
            )

            if png is not None:
                print(f"Buffered screenshot retrieved for monitor {mon.get('monitor_id', '?')} for click: {button}")
            else:
                print(f"No buffered screenshot available for monitor {mon.get('monitor_id', '?')} for click")

            self.queue.enqueue(
                event_type=evt,
                details={"button": str(button)},
                monitor=mon,
                cursor_pos=[x, y],
                screenshot=(png, size) if png is not None else None
            )

    def on_move(self, x, y):
        now = time.time()
        if now - self._last_move_time < self.move_interval:
            return
        self._last_move_time = now

        mon = self.screenshot_manager.get_active_monitor(x, y)

        png, size = self.screenshot_manager.take_screenshot_for_monitor(
            mon, quality=95, lookback_ms=self.lookback_ms
        )

        if png is not None:
            print(f"Buffered screenshot retrieved for monitor {mon.get('monitor_id', '?')} for mouse move")
        else:
            print(f"No buffered screenshot available for monitor {mon.get('monitor_id', '?')} for mouse move")

        self.queue.enqueue(
            event_type="mouse_move",
            details={},
            monitor=mon,
            cursor_pos=[x, y],
            screenshot=(png, size) if png is not None else None
        )

    def on_scroll(self, x, y, dx, dy):
        mon = self.screenshot_manager.get_active_monitor(x, y)

        png, size = self.screenshot_manager.take_screenshot_for_monitor(
            mon, quality=95, lookback_ms=self.lookback_ms
        )

        if png is not None:
            print(f"Buffered screenshot retrieved for monitor {mon.get('monitor_id', '?')} for mouse scroll")
        else:
            print(f"No buffered screenshot available for monitor {mon.get('monitor_id', '?')} for mouse scroll")

        self.queue.enqueue(
            event_type="mouse_scroll",
            details={"scroll": [dx, dy]},
            monitor=mon,
            cursor_pos=[x, y],
            screenshot=(png, size) if png is not None else None
        )
