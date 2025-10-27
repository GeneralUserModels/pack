import time
from collections import defaultdict
from pynput import mouse
from screeninfo import get_monitors
from record.models.event import InputEvent, EventType
from record.models.event_queue import EventQueue


class InputEventHandler:
    """Handler for capturing and recording input events."""

    def __init__(self, event_queue: EventQueue, accessibility: bool = False):
        self.event_queue = event_queue
        self._monitors = list(get_monitors())
        self.accessibility_enabled = accessibility
        self.accessibility_handler = None
        
        self.perf_stats = {
            'handler_counts': defaultdict(int),
            'handler_times': defaultdict(float),
            'accessibility_counts': defaultdict(int),
            'accessibility_times': defaultdict(float),
        }
        self.session_start_time = time.time()
        
        if self.accessibility_enabled:
            try:
                from record.handlers.accessibility import AccessibilityHandler
                self.accessibility_handler = AccessibilityHandler()
            except ImportError as e:
                print(f"Warning: Could not import AccessibilityHandler: {e}")
                print("Accessibility features will be disabled.")
                self.accessibility_enabled = False

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
        handler_start = time.perf_counter()
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
        
        if self.accessibility_enabled and self.accessibility_handler:
            ax_start = time.perf_counter()
            ax_data = self.accessibility_handler(event)
            ax_elapsed = time.perf_counter() - ax_start
            self.perf_stats['accessibility_times']['mouse_move'] += ax_elapsed
            self.perf_stats['accessibility_counts']['mouse_move'] += 1
            if ax_data:
                event.details.update(ax_data)
        
        self.event_queue.enqueue(event)
        handler_elapsed = time.perf_counter() - handler_start
        self.perf_stats['handler_times']['on_move'] += handler_elapsed
        self.perf_stats['handler_counts']['on_move'] += 1

    def on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        """
        Callback for mouse click events.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button
            pressed: True if pressed, False if released
        """
        handler_start = time.perf_counter()
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
        
        if self.accessibility_enabled and self.accessibility_handler:
            ax_start = time.perf_counter()
            ax_data = self.accessibility_handler(event)
            ax_elapsed = time.perf_counter() - ax_start
            event_key = 'mouse_down' if pressed else 'mouse_up'
            self.perf_stats['accessibility_times'][event_key] += ax_elapsed
            self.perf_stats['accessibility_counts'][event_key] += 1
            if ax_data:
                event.details.update(ax_data)
        
        self.event_queue.enqueue(event)
        handler_elapsed = time.perf_counter() - handler_start
        self.perf_stats['handler_times']['on_click'] += handler_elapsed
        self.perf_stats['handler_counts']['on_click'] += 1

    def on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        """
        Callback for mouse scroll events.

        Args:
            x: X coordinate
            y: Y coordinate
            dx: Horizontal scroll amount
            dy: Vertical scroll amount
        """
        handler_start = time.perf_counter()
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
        
        if self.accessibility_enabled and self.accessibility_handler:
            ax_start = time.perf_counter()
            ax_data = self.accessibility_handler(event)
            ax_elapsed = time.perf_counter() - ax_start
            self.perf_stats['accessibility_times']['scroll'] += ax_elapsed
            self.perf_stats['accessibility_counts']['scroll'] += 1
            if ax_data:
                event.details.update(ax_data)
        
        self.event_queue.enqueue(event)
        handler_elapsed = time.perf_counter() - handler_start
        self.perf_stats['handler_times']['on_scroll'] += handler_elapsed
        self.perf_stats['handler_counts']['on_scroll'] += 1

    def on_press(self, key) -> None:
        """
        Callback for keyboard press events.

        Args:
            key: Key that was pressed
        """
        handler_start = time.perf_counter()
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
        
        if self.accessibility_enabled and self.accessibility_handler:
            ax_start = time.perf_counter()
            ax_data = self.accessibility_handler(event)
            ax_elapsed = time.perf_counter() - ax_start
            self.perf_stats['accessibility_times']['key_press'] += ax_elapsed
            self.perf_stats['accessibility_counts']['key_press'] += 1
            if ax_data:
                event.details.update(ax_data)
        
        self.event_queue.enqueue(event)
        handler_elapsed = time.perf_counter() - handler_start
        self.perf_stats['handler_times']['on_press'] += handler_elapsed
        self.perf_stats['handler_counts']['on_press'] += 1

    def on_release(self, key) -> None:
        """
        Callback for keyboard release events.

        Args:
            key: Key that was released
        """
        handler_start = time.perf_counter()
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
        
        if self.accessibility_enabled and self.accessibility_handler:
            ax_start = time.perf_counter()
            ax_data = self.accessibility_handler(event)
            ax_elapsed = time.perf_counter() - ax_start
            self.perf_stats['accessibility_times']['key_release'] += ax_elapsed
            self.perf_stats['accessibility_counts']['key_release'] += 1
            if ax_data:
                event.details.update(ax_data)
        
        self.event_queue.enqueue(event)
        handler_elapsed = time.perf_counter() - handler_start
        self.perf_stats['handler_times']['on_release'] += handler_elapsed
        self.perf_stats['handler_counts']['on_release'] += 1
    
    def print_performance_stats(self) -> None:
        session_duration = time.time() - self.session_start_time
        total_events = sum(self.perf_stats['handler_counts'].values())
        
        print("\n" + "="*70)
        print(">>>>                  Performance Summary                      <<<<")
        print("="*70)
        print(f"Session Duration: {session_duration:.1f}s")
        print(f"Total Events: {total_events:,}\n")
        
        print("Event Handler Performance:")
        total_handler_time = sum(self.perf_stats['handler_times'].values())
        handler_pct = (total_handler_time / session_duration * 100) if session_duration > 0 else 0
        print(f"  Total time in handlers: {total_handler_time:.2f}s ({handler_pct:.2f}% of session)")
        
        for handler_name in ['on_move', 'on_click', 'on_scroll', 'on_press', 'on_release']:
            count = self.perf_stats['handler_counts'][handler_name]
            elapsed = self.perf_stats['handler_times'][handler_name]
            if count > 0:
                avg_ms = (elapsed / count) * 1000
                print(f"  - {handler_name:12s}: {count:6,} events, {elapsed:6.2f}s ({avg_ms:6.3f}ms avg)")
        
        if self.accessibility_enabled:
            print("\nAccessibility Performance:")
            total_ax_time = sum(self.perf_stats['accessibility_times'].values())
            ax_pct = (total_ax_time / session_duration * 100) if session_duration > 0 else 0
            total_ax_calls = sum(self.perf_stats['accessibility_counts'].values())
            
            if total_ax_calls > 0:
                overhead = (total_ax_time / total_handler_time) if total_handler_time > 0 else 0
                print(f"  Total time: {total_ax_time:.2f}s ({ax_pct:.2f}% of session)")
                print(f"  Total calls: {total_ax_calls:,}")
                print(f"  Overhead vs baseline: {overhead:.1f}x")
                
                for event_type in ['mouse_move', 'mouse_down', 'mouse_up', 'scroll', 'key_press', 'key_release']:
                    count = self.perf_stats['accessibility_counts'][event_type]
                    elapsed = self.perf_stats['accessibility_times'][event_type]
                    if count > 0:
                        avg_ms = (elapsed / count) * 1000
                        print(f"  - {event_type:12s}: {count:6,} calls, {elapsed:6.2f}s ({avg_ms:6.3f}ms avg)")
        
        if total_events > 0 and session_duration > 0:
            throughput = total_events / session_duration
            print(f"\nThroughput: {throughput:.1f} events/sec")
        
        print("="*70)
