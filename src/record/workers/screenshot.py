import time
import numpy as np
from typing import Optional, Tuple, Union
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


def _resize_by_scale(img_rgb: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize a HxWx3 uint8 RGB numpy image by a scale factor.
    Only downscales (scale < 1.0). Returns the (possibly) resized image.
    """
    if scale >= 1.0:
        return img_rgb

    h, w = img_rgb.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    pil = Image.fromarray(img_rgb)
    pil_resized = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(pil_resized)


def capture_screenshot(
    sct,
    x: int,
    y: int,
    max_res: tuple[int, int] = None,
    scale: Union[float, dict[int, float]] = None
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float], Optional[dict]]:
    """
    Capture a screenshot from sct that contains (x, y).
    
    Args:
        sct: mss screenshot object
        x: cursor x coordinate
        y: cursor y coordinate
        max_res: optional max resolution to fit within (width, height)
        scale: optional scale factor (0.0-1.0) or dict mapping monitor index to scale
    
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
        h, w = img_rgb.shape[:2]
        
        # Resolve scale for this monitor
        effective_scale = None
        if scale is not None:
            if isinstance(scale, dict):
                # Per-monitor scale: look up by monitor_index
                # screeninfo uses 0-based indexing, mss uses 1-based (0 = all monitors)
                # So mss monitor 1 = screeninfo monitor 0
                screeninfo_idx = max(0, monitor_index - 1)
                effective_scale = scale.get(screeninfo_idx)
            else:
                effective_scale = scale
        
        # Apply scale factor if provided (takes priority over max_res)
        if effective_scale is not None and effective_scale < 1.0:
            img_rgb = _resize_by_scale(img_rgb, effective_scale)
            new_h, new_w = img_rgb.shape[:2]
            scale_factor = new_w / w
        elif max_res is not None:
            img_rgb = _resize_if_needed(img_rgb, max_res)
            new_h, new_w = img_rgb.shape[:2]
            scale_factor = new_w / w

        return img_rgb, monitor_index, time_before, scale_factor, monitor
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None, None, None, None, None
