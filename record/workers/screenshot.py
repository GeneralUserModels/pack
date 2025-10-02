import numpy as np
from typing import Optional
import mss
from pynput import mouse
from screeninfo import get_monitors


def get_active_monitor() -> int:
    """
    Get the index of the monitor where the cursor is currently located.

    Returns:
        Monitor index (0-based)
    """
    def in_bounds(x: int, y: int, monitor) -> bool:
        return (monitor.x <= x < monitor.x + monitor.width and monitor.y <= y < monitor.y + monitor.height)

    try:
        controller = mouse.Controller()
        x, y = controller.position

        monitors = list(get_monitors())

        for idx, monitor in enumerate(monitors):
            if in_bounds(x, y, monitor):
                return idx

        return 0
    except Exception as e:
        print(f"Error getting active monitor: {e}")
        return 0


def capture_screenshot(monitor_index: Optional[int] = None) -> Optional[np.ndarray]:
    try:
        with mss.mss() as sct:
            # If no monitor specified, get the active one
            if monitor_index is None:
                monitor_index = get_active_monitor()

            monitor = sct.monitors[monitor_index + 1]

            screenshot = sct.grab(monitor)

            # Convert to numpy array (BGRA format from mss)
            img = np.array(screenshot)
            # Convert BGRA to RGB
            img_rgb = img[:, :, [2, 1, 0]]

            return img_rgb
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None
