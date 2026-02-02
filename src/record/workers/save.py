import json
from pathlib import Path
import cv2
from record.models.image import BufferImage
from record.models.event import InputEvent


class SaveWorker:
    """Worker for saving queue items to disk."""

    def __init__(
        self,
        session_dir: Path,
        buffer_all: bool = False,
        compression_quality: int = 70,
        lossless: bool = False,
        save_screenshots: bool = True
    ):
        """
        Initialize the save worker.

        Args:
            session_dir: Directory for the current session
            buffer_all: If True, save all buffer images
            compression_quality: JPEG compression quality (1-100)
            lossless: If True, save as PNG (lossless) instead of JPEG
            save_screenshots: If False, skip writing screenshot files to disk
        """
        self.session_dir = Path(session_dir)
        self.screenshots_dir = self.session_dir / "screenshots"
        self.buffer_all = buffer_all
        self.save_screenshots = save_screenshots
        if buffer_all:
            self.buffer_imgs_dir = self.session_dir / "buffer_imgs"
            self.buffer_imgs_dir.mkdir(exist_ok=True)

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)

        self.input_log = self.session_dir / "input_events.jsonl"
        self.screenshot_log = self.session_dir / "screenshots.jsonl"
        self.compression_quality = compression_quality
        self.lossless = lossless

    def save_input_event(self, event: InputEvent) -> None:
        """
        Save an input event to JSONL.

        Args:
            event: Input event to save
        """
        try:
            with open(self.input_log, 'a') as f:
                json.dump(event.to_dict(), f)
                f.write('\n')
        except Exception as e:
            print(f"Error saving input event: {e}")

    def save_image(self, image: BufferImage, buffer_dir: bool = False, force_save: bool = False, save_reason: str = "") -> str:
        """
        Save an image to disk and log the metadata.

        Args:
            image: BufferImage to save
            buffer_dir: If True, save to buffer_imgs, else to screenshots
            force_save: If True, always write the image file regardless of self.buffer_all

        Returns:
            Path to saved image (string), or empty string if save_screenshots is False
        """
        # Skip saving entirely if save_screenshots is disabled
        if not self.save_screenshots:
            return ""

        try:
            save_dir = self.buffer_imgs_dir if buffer_dir else self.screenshots_dir
            ext = ".png" if self.lossless else ".jpg"
            filename = f"{image.timestamp:.6f}_reason_{save_reason}{ext}"
            filepath = save_dir / filename

            if force_save or self.buffer_all:
                try:
                    img_bgr = cv2.cvtColor(image.data, cv2.COLOR_RGB2BGR)
                except Exception:
                    img_bgr = image.data
                
                if self.lossless:
                    cv2.imwrite(str(filepath), img_bgr)
                else:
                    cv2.imwrite(str(filepath), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.compression_quality])

            metadata = {
                'timestamp': image.timestamp,
                'path': str(filepath.relative_to(self.session_dir)),
                'monitor_index': image.monitor_index,
            }

            with open(self.screenshot_log, 'a') as f:
                json.dump(metadata, f)
                f.write('\n')

            return str(filepath)
        except Exception as e:
            print(f"Error saving image: {e}")
            return ""

    def save_buffer_image(self, image: BufferImage) -> str:
        """Save image to buffer_imgs directory."""
        return self.save_image(image, buffer_dir=True)

    def save_screenshot(self, image: BufferImage, force_save: bool = False, save_reason: str = "") -> str:
        """Save image to screenshots directory."""
        return self.save_image(image, buffer_dir=False, force_save=force_save, save_reason=save_reason)
