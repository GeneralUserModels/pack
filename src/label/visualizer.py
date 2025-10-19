from pathlib import Path
from typing import List, Optional
import json
import tempfile
import subprocess
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


class Visualizer:
    def __init__(self):
        self._load_fonts()

    def _load_fonts(self):
        try:
            self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            self.font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            self.font_large = self.font_medium = self.font_small = ImageFont.load_default()

    def visualize(self, session_dir: Path, output_path: Optional[Path] = None, fps: int = 1) -> Path:
        if not output_path:
            output_path = session_dir / "annotated.mp4"

        matched_path = session_dir / "data.jsonl"
        if not matched_path.exists():
            raise RuntimeError(f"matched_captions.jsonl not found in {session_dir}")

        matched_data = []
        with open(matched_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    matched_data.append(json.loads(line))

        valid_images = []
        for entry in matched_data:
            img_path = entry.get('img')
            if not img_path:
                continue

            p = Path(img_path)
            if not p.exists():
                p = session_dir / "screenshots" / p.name

            if p.exists():
                valid_images.append((p, entry))

        if not valid_images:
            raise RuntimeError("No valid images found")

        loaded = []
        widths, heights = [], []

        for p, entry in tqdm(valid_images, desc="Loading images"):
            try:
                img = Image.open(p).convert('RGB')
                widths.append(img.width)
                heights.append(img.height)
                loaded.append((img, entry))
            except Exception as e:
                print(f"Warning: failed to open {p}: {e}")

        if not loaded:
            raise RuntimeError("No images could be loaded")

        target_w, target_h = max(widths), max(heights)

        with tempfile.TemporaryDirectory(prefix="viz_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            for idx, (img, entry) in enumerate(tqdm(loaded, desc="Annotating")):
                canvas = self._resize_and_pad(img, target_w, target_h)
                annotated = self._annotate(canvas, entry)

                frame_path = tmpdir_path / f"{idx:06d}.jpg"
                annotated.save(frame_path)

            self._create_video(tmpdir_path, output_path, fps)

        return output_path

    def _resize_and_pad(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS) if (new_w, new_h) != (w, h) else img

        canvas = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        paste_x, paste_y = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))

        return canvas

    def _annotate(self, img: Image.Image, entry: dict) -> Image.Image:
        draw = ImageDraw.Draw(img)
        width, height = img.size

        caption = entry.get('caption', '')
        events = entry.get('raw_events', [])
        start_time = entry.get('start_formatted', '')
        end_time = entry.get('end_formatted', '')

        event_summary = self._summarize_events(events)

        self._draw_caption_box(draw, width, height, caption, start_time, end_time)

        if event_summary:
            self._draw_event_box(draw, width, height, event_summary)

        return img

    def _summarize_events(self, events: List[dict]) -> str:
        if not events:
            return ""

        compressed = self._compress_events(events)
        actions = []
        keys = []

        for event in compressed:
            etype = event.get("event_type")
            details = event.get("details", {})

            if etype == "key_press":
                key = details.get("key", "unknown").replace("Key.", "")
                if key:
                    keys.append(key)

            elif etype in ["mouse_down", "mouse_click"]:
                if keys:
                    actions.append(f"Key: {keys[0]}" if len(keys) == 1 else f"Keys: {'+'.join(keys)}")
                    keys.clear()

                button = details.get("button", "unknown").replace("Button.", "")
                double = details.get("double_click", False)
                click_text = f"Click {button}" + (" (2x)" if double else "")
                actions.append(click_text)

            elif etype == "mouse_scroll":
                if keys:
                    actions.append(f"Key: {keys[0]}" if len(keys) == 1 else f"Keys: {'+'.join(keys)}")
                    keys.clear()

                direction = event.get("_direction") or self._scroll_direction(details)
                actions.append(f"Scroll {direction}")

            elif etype == "mouse_move":
                if keys:
                    actions.append(f"Key: {keys[0]}" if len(keys) == 1 else f"Keys: {'+'.join(keys)}")
                    keys.clear()
                actions.append("Move mouse")

        if keys:
            actions.append(f"Key: {keys[0]}" if len(keys) == 1 else f"Keys: {'+'.join(keys)}")

        if len(actions) > 5:
            actions = actions[:5] + [f"... and {len(actions) - 5} more"]

        return "\n".join([f"• {a}" for a in actions])

    def _compress_events(self, events: List[dict]) -> List[dict]:
        if not events:
            return []

        compressed = []
        i, n = 0, len(events)

        while i < n:
            e = events[i]
            et = e.get("event_type")

            if et == "mouse_move":
                j = i + 1
                while j < n and events[j].get("event_type") == "mouse_move":
                    j += 1
                compressed.append(events[j - 1])
                i = j

            elif et == "mouse_scroll":
                j, last_dir = i, None
                while j < n and events[j].get("event_type") == "mouse_scroll":
                    details = events[j].get("details", {})
                    dir_ = self._scroll_direction(details)
                    if dir_ != last_dir:
                        compressed.append({"event_type": "mouse_scroll", "details": details, "_direction": dir_})
                        last_dir = dir_
                    j += 1
                i = j

            else:
                compressed.append(e)
                i += 1

        return compressed

    def _scroll_direction(self, scroll_data) -> str:
        if isinstance(scroll_data, dict):
            dx, dy = scroll_data.get("dx", 0), scroll_data.get("dy", 0)
        elif isinstance(scroll_data, (list, tuple)) and len(scroll_data) >= 2:
            dx, dy = scroll_data[0], scroll_data[1]
        else:
            return "unknown"

        dirs = []
        if dy > 0:
            dirs.append("up")
        elif dy < 0:
            dirs.append("down")
        if dx > 0:
            dirs.append("right")
        elif dx < 0:
            dirs.append("left")

        return " ".join(dirs) if dirs else "unknown"

    def _draw_caption_box(self, draw: ImageDraw.Draw, width: int, height: int,
                          caption: str, start_time: str, end_time: str):

        padding = 20
        time_str = f"[{start_time} - {end_time}]"
        full_text = f"{time_str}\n{caption}"
        lines = self._wrap_text(full_text, width - 100, self.font_medium)

        line_height = 30
        box_height = len(lines) * line_height + 2 * padding
        box_width = width - 40
        box_x, box_y = 20, 20

        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill=(0, 100, 200, 200))

        img_rgba = draw._image.convert('RGBA')
        img_rgba = Image.alpha_composite(img_rgba, overlay)
        draw._image.paste(img_rgba.convert('RGB'))

        y_offset = box_y + padding
        for i, line in enumerate(lines):
            font = self.font_large if i == 0 else self.font_medium
            draw.text((box_x + padding, y_offset), line, fill='white', font=font)
            y_offset += line_height

    def _draw_event_box(self, draw: ImageDraw.Draw, width: int, height: int, event_summary: str):
        padding = 15
        lines = event_summary.split('\n')
        line_height = 22
        box_height = len(lines) * line_height + 2 * padding
        box_width = width - 40
        box_x = 20
        box_y = height - box_height - 20

        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill=(50, 50, 50, 180))

        img_rgba = draw._image.convert('RGBA')
        img_rgba = Image.alpha_composite(img_rgba, overlay)
        draw._image.paste(img_rgba.convert('RGB'))

        y_offset = box_y + padding
        for line in lines:
            draw.text((box_x + padding, y_offset), line, fill='white', font=self.font_small)
            y_offset += line_height

    def _wrap_text(self, text: str, max_width: int, font) -> List[str]:
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph:
                lines.append('')
                continue

            words = paragraph.split()
            current = []

            for word in words:
                test = ' '.join(current + [word])
                bbox = font.getbbox(test)
                if bbox[2] - bbox[0] <= max_width:
                    current.append(word)
                else:
                    if current:
                        lines.append(' '.join(current))
                    current = [word]

            if current:
                lines.append(' '.join(current))

        return lines

    def _create_video(self, frames_dir: Path, output_path: Path, fps: int):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "%06d.jpg"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
