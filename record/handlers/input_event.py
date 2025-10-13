import time
from pynput import mouse
from screeninfo import get_monitors
from record.models.event import InputEvent, EventType
from record.models.event_queue import EventQueue


class InputEventHandler:
    """Handler for capturing and recording input events."""

    def __init__(self, event_queue: EventQueue):
        self.event_queue = event_queue
        self._monitors = list(get_monitors())

    def _get_monitor(self, x: int, y: int) -> int:
        """
        Get the monitor index for given coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Monitor index (0-based)
        """
        def to_monitor_dict(monitor):
            return {
                "left": monitor.x, "top": monitor.y, "width": monitor.width, "height": monitor.height
            }

        for idx, monitor in enumerate(self._monitors):
            if (monitor.x <= x < monitor.x + monitor.width and
                    monitor.y <= y < monitor.y + monitor.height):
                return idx, to_monitor_dict(monitor)
        return 0, to_monitor_dict(self._monitors[0])

    def on_move(self, x: int, y: int) -> None:
        """
        Callback for mouse move events.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        timestamp = time.time()
        monitor_idx, monitor = self._get_monitor(x, y)

        event = InputEvent(
            timestamp=timestamp,
            monitor_index=monitor_idx,
            monitor=monitor,
            event_type=EventType.MOUSE_MOVE,
            details={'x': x, 'y': y},
            cursor_position=(x, y)
        )
        self.event_queue.enqueue(event)

    def on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        """
        Callback for mouse click events.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button
            pressed: True if pressed, False if released
        """
        timestamp = time.time()
        monitor_idx, monitor = self._get_monitor(x, y)

        event = InputEvent(
            timestamp=timestamp,
            monitor_index=monitor_idx,
            monitor=monitor,
            event_type=EventType.MOUSE_DOWN if pressed else EventType.MOUSE_UP,
            details={
                'x': x,
                'y': y,
                'button': str(button),
            },
            cursor_position=(x, y)
        )
        self.event_queue.enqueue(event)

    def on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        """
        Callback for mouse scroll events.

        Args:
            x: X coordinate
            y: Y coordinate
            dx: Horizontal scroll amount
            dy: Vertical scroll amount
        """
        timestamp = time.time()
        monitor_idx, monitor = self._get_monitor(x, y)

        event = InputEvent(
            timestamp=timestamp,
            monitor_index=monitor_idx,
            monitor=monitor,
            event_type=EventType.MOUSE_SCROLL,
            details={
                'x': x,
                'y': y,
                'dx': dx,
                'dy': dy
            },
            cursor_position=(x, y)
        )
        self.event_queue.enqueue(event)

    def on_press(self, key) -> None:
        """
        Callback for keyboard press events.

        Args:
            key: Key that was pressed
        """
        timestamp = time.time()
        x, y = None, None

        controller = mouse.Controller()
        x, y = controller.position
        monitor_idx, monitor = self._get_monitor(x, y)

        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        event = InputEvent(
            timestamp=timestamp,
            monitor_index=monitor_idx,
            monitor=monitor,
            event_type=EventType.KEY_PRESS,
            details={'key': key_char},
            cursor_position=(x, y)
        )
        self.event_queue.enqueue(event)

    def on_release(self, key) -> None:
        """
        Callback for keyboard release events.

        Args:
            key: Key that was released
        """
        timestamp = time.time()
        x, y = None, None

        controller = mouse.Controller()
        x, y = controller.position
        monitor_idx, monitor = self._get_monitor(x, y)

        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        event = InputEvent(
            timestamp=timestamp,
            monitor_index=monitor_idx,
            monitor=monitor,
            event_type=EventType.KEY_RELEASE,
            details={'key': key_char},
            cursor_position=(x, y)
        )
        self.event_queue.enqueue(event)
