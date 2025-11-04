import time
import numpy as np
from typing import Optional, Tuple
from PIL import Image


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
    x = int(x)
    y = int(y)

    for i, mon in enumerate(sct.monitors[1:], start=1):
        if is_active_monitor(mon, x, y):
            return i

    return 0


def _resize_if_needed(img_rgb: np.ndarray, max_res) -> np.ndarray:
    """
    Resize (downscale only) a HxWx3 uint8 RGB numpy image so it fits within the appropriate
    FullHD box depending on orientation. Returns the (possibly) resized image.
    """
    h, w = img_rgb.shape[:2]
    landscape_res = (max_res[0], max_res[1])
    portrait_res = (max_res[1], max_res[0])
    if w >= h:
        target_w, target_h = landscape_res
    else:
        target_w, target_h = portrait_res

    scale = min(target_w / w, target_h / h, 1.0)

    if scale >= 1.0:
        return img_rgb

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    pil = Image.fromarray(img_rgb)
    pil_resized = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(pil_resized)


def capture_screenshot(sct, x: int, y: int, max_res: tuple[int, int] = None) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float], Optional[dict]]:
    """
    Capture a screenshot from sct that contains (x, y).
    Returns (img_rgb, monitor_index, timestamp, scale_factor, monitor_dict) or (None, None, None, None, None) on error.
    """
    try:
        x = int(x)
        y = int(y)
        monitor_index = get_active_monitor(x, y, sct)
        max_idx = len(sct.monitors) - 1
        if monitor_index < 0:
            monitor_index = 0
        elif monitor_index > max_idx:
            monitor_index = max_idx
        monitor = sct.monitors[monitor_index]

        time_before = time.time()
        screenshot = sct.grab(monitor)

        img = np.array(screenshot)
        img_rgb = img[:, :, [2, 1, 0]]

        scale_factor = 1.0
        if max_res is not None:
            h, w = img_rgb.shape[:2]
            img_rgb = _resize_if_needed(img_rgb, max_res)
            new_h, new_w = img_rgb.shape[:2]
            scale_factor = new_w / w

        return img_rgb, monitor_index, time_before, scale_factor, monitor
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None, None, None, None, None
