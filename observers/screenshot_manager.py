import mss
import threading
from contextlib import contextmanager


class ScreenshotManager:
    def __init__(self):
        self._tls = threading.local()
        self._lock = threading.Lock()

    @contextmanager
    def _get_sct_context(self, with_cursor: bool = False):
        """Context manager that ensures proper cleanup of mss instances."""
        sct = mss.mss(with_cursor=with_cursor)
        try:
            yield sct
        finally:
            sct.close()

    def get_active_monitor(self, x: int, y: int) -> dict:
        with self._get_sct_context(with_cursor=False) as sct:
            for mon in sct.monitors[1:]:
                if (mon["left"] <= x < mon["left"] + mon["width"] and
                        mon["top"] <= y < mon["top"] + mon["height"]):
                    return mon
            return sct.monitors[0]

    def take_screenshot_for_monitor(self, mon: dict) -> tuple[bytes, tuple[int, int]]:
        with self._get_sct_context(with_cursor=True) as sct:
            img = sct.grab(mon)
            png = mss.tools.to_png(img.rgb, img.size)
            return png, img.size

    def take_virtual_screenshot(self) -> tuple[bytes, tuple[int, int]]:
        with self._get_sct_context(with_cursor=True) as sct:
            img = sct.grab(sct.monitors[0])
            png = mss.tools.to_png(img.rgb, img.size)
            return png, img.size

    def close(self):
        """Clean up any remaining resources."""
        pass
