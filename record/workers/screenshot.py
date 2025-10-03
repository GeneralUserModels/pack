import numpy as np
from typing import Optional


def is_active_monitor(mon: dict, x: int, y: int) -> bool:
    """Check if coordinates are within monitor bounds"""
    return (mon["left"] <= x < mon["left"] + mon["width"] and
            mon["top"] <= y < mon["top"] + mon["height"])


def get_active_monitor(x: int, y: int, sct) -> int:
    """
    Return the monitor index in sct.monitors that contains (x, y).
    sct.monitors[0] is the virtual/all-monitors image, physical monitors are 1..N.
    Returns an index suitable for sct.monitors (0..N).
    """
    # ensure ints
    x = int(x)
    y = int(y)

    for i, mon in enumerate(sct.monitors[1:], start=1):
        if is_active_monitor(mon, x, y):
            return i

    return 0


def capture_screenshot(sct, x: int, y: int) -> Optional[np.ndarray]:
    try:
        x = int(x)
        y = int(y)

        monitor_index = get_active_monitor(x, y, sct)

        max_idx = len(sct.monitors) - 1  # highest physical index; index 0 is allowed too
        if monitor_index < 0:
            monitor_index = 0
        elif monitor_index > max_idx:
            monitor_index = max_idx

        monitor = sct.monitors[monitor_index]
        screenshot = sct.grab(monitor)

        img = np.array(screenshot)
        img_rgb = img[:, :, [2, 1, 0]]

        return img_rgb, monitor_index
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None, None
