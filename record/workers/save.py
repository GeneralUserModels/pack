import json
from pathlib import Path
import cv2
from record.models.image import BufferImage
from record.models.event import InputEvent


class SaveWorker:
    """Worker for saving queue items to disk."""

    def __init__(self, session_dir: Path):
        """
        Initialize the save worker.

        Args:
            session_dir: Directory for the current session
        """
        self.session_dir = Path(session_dir)
        self.screenshots_dir = self.session_dir / "screenshots"
        self.buffer_imgs_dir = self.session_dir / "buffer_imgs"

        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)
        self.buffer_imgs_dir.mkdir(exist_ok=True)

        # File paths for JSONL logs
        self.ssim_log = self.session_dir / "ssim.jsonl"
        self.input_log = self.session_dir / "input_events.jsonl"
        self.screenshot_log = self.session_dir / "screenshots.jsonl"

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

    def save_ssim_value(self, image: BufferImage) -> None:
        """
        Save SSIM value to JSONL.

        Args:
            image: BufferImage containing SSIM value
        """
        try:
            if image.ssim_value is not None:
                data = {
                    'timestamp': image.timestamp,
                    'ssim_value': image.ssim_value,
                    'monitor_index': image.monitor_index
                }
                with open(self.ssim_log, 'a') as f:
                    json.dump(data, f)
                    f.write('\n')
        except Exception as e:
            print(f"Error saving SSIM value: {e}")

    def save_image(self, image: BufferImage, buffer_dir: bool = False) -> str:
        """
        Save an image to disk and log the metadata.

        Args:
            image: BufferImage to save
            buffer_dir: If True, save to buffer_imgs, else to screenshots

        Returns:
            Path to saved image
        """
        try:
            # Determine save directory
            save_dir = self.buffer_imgs_dir if buffer_dir else self.screenshots_dir

            # Generate filename
            filename = f"{image.timestamp:.6f}_monitor{image.monitor_index}.jpg"
            filepath = save_dir / filename

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(image.screenshot, cv2.COLOR_RGB2BGR)

            # Save image
            cv2.imwrite(str(filepath), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Log metadata
            metadata = {
                'timestamp': image.timestamp,
                'path': str(filepath.relative_to(self.session_dir)),
                'monitor_index': image.monitor_index,
                'ssim_value': image.ssim_value
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

    def save_screenshot(self, image: BufferImage) -> str:
        """Save image to screenshots directory."""
        return self.save_image(image, buffer_dir=False)
