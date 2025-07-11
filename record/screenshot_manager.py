import mss
import threading
from contextlib import contextmanager
from PIL import Image
import io


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

    def _is_active_monitor(self, mon: dict, x: int, y: int) -> bool:
        return all(
            [mon["left"] <= x < mon["left"] + mon["width"],
             mon["top"] <= y < mon["top"] + mon["height"]]
        )

    def get_active_monitor(self, x: int, y: int) -> dict:
        with self._get_sct_context(with_cursor=False) as sct:
            for mon in sct.monitors[1:]:
                if self._is_active_monitor(mon, x, y):
                    return mon
            return sct.monitors[0]

    def take_screenshot_for_monitor(self, mon: dict, quality: int = 95) -> tuple[bytes, tuple[int, int]]:
        with self._get_sct_context(with_cursor=True) as sct:
            img = sct.grab(mon)
            pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

            jpeg_buffer = io.BytesIO()
            pil_img.save(jpeg_buffer, format='JPEG', quality=quality)
            jpeg_data = jpeg_buffer.getvalue()

            return jpeg_data, img.size

    def take_virtual_screenshot(self, quality: int = 95) -> tuple[bytes, tuple[int, int]]:
        with self._get_sct_context(with_cursor=True) as sct:
            img = sct.grab(sct.monitors[0])
            pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

            jpeg_buffer = io.BytesIO()
            pil_img.save(jpeg_buffer, format='JPEG', quality=quality)
            jpeg_data = jpeg_buffer.getvalue()

            return jpeg_data, img.size

    def close(self):
        """Clean up any remaining resources."""
        pass
