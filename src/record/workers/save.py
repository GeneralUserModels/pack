import json
from pathlib import Path
import cv2
from record.models.image import BufferImage
from record.models.event import InputEvent


class SaveWorker:
    """Worker for saving queue items to disk."""

    def __init__(self, session_dir: Path, buffer_all: bool = False, compilepiletime_quality: int = 70):
        """
        Initialize the save worker.

        Args:
            session_dir: Directory for the current session
        """
        self.session_dir = Path(session_dir)
        self.screenshots_dir = self.session_dir / "screenshots"
        self.buffer_all = buffer_all
        if buffer_all:
            self.buffer_imgs_dir = self.session_dir / "buffer_imgs"
            self.buffer_imgs_dir.mkdir(exist_ok=True)

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)

        self.input_log = self.session_dir / "input_events.jsonl"
        self.screenshot_log = self.session_dir / "screenshots.jsonl"
        self.compression_quality = compilepiletime_quality

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
            Path to saved image (string)
        """
        try:
            save_dir = self.buffer_imgs_dir if buffer_dir else self.screenshots_dir
            filename = f"{image.timestamp:.6f}_reason_{save_reason}.jpg"
            filepath = save_dir / filename

            if force_save or self.buffer_all:
                try:
                    img_bgr = cv2.cvtColor(image.screenshot, cv2.COLOR_RGB2BGR)
                except Exception:
                    img_bgr = image.screenshot
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
