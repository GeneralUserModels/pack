from pathlib import Path
from typing import List, Optional
import json
import tempfile
import subprocess
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from label.video import annotate_image, scale_and_pad, apply_pending_movement, extract_pending_movement
from label.models import Aggregation


class Visualizer:
    def __init__(self, annotate: bool = True):
        self.annotate = annotate
        self._load_fonts()

    def _load_fonts(self):
        try:
            self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            self.font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            self.font_large = self.font_medium = self.font_small = ImageFont.load_default()

    def visualize(
        self,
        session_dir: Path,
        output_path: Optional[Path] = None,
        fps: int = 1,
        deduplicate_events: bool = True,
        min_event_count: int = 3
    ) -> Path:
        """
        Create an annotated video from session data.

        Args:
            session_dir: Path to the session directory
            output_path: Output video path (default: session_dir/annotated.mp4)
            fps: Frames per second for the output video
            deduplicate_events: If True, deduplicate events in text summaries
            min_event_count: Minimum consecutive events to summarize when deduplicating
        """
        if not output_path:
            output_path = session_dir / "annotated.mp4"

        matched_path = session_dir / "data.jsonl"
        if not matched_path.exists():
            raise RuntimeError(f"data.jsonl not found in {session_dir}")

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

            # Initialize pending movement tracking
            pending_movement = []

            for idx, (img, entry) in enumerate(tqdm(loaded, desc="Annotating")):
                # Scale and pad the image
                canvas, scale, x_offset, y_offset = scale_and_pad(img, target_w, target_h)

                # Apply arrow annotations if requested
                if self.annotate:
                    aggregations = self._reconstruct_aggregations(entry)
                    if aggregations:
                        # Apply pending movement from previous frame
                        agg = apply_pending_movement(aggregations[0], pending_movement)

                        # Use the aggregation with pending movements for visual annotation
                        canvas = annotate_image(canvas, agg, scale, x_offset, y_offset)

                        # Extract pending movement for next frame
                        pending_movement = extract_pending_movement(agg)
                    else:
                        # Reset pending movement if no aggregations
                        pending_movement = []
                else:
                    # Reset pending movement if annotation is disabled
                    pending_movement = []

                # Add text overlays
                annotated = self._add_text_overlays(canvas, entry, deduplicate_events, min_event_count)

                frame_path = tmpdir_path / f"{idx:06d}.jpg"
                annotated.save(frame_path)

            self._create_video(tmpdir_path, output_path, fps)

        return output_path

    def _reconstruct_aggregations(self, entry: dict) -> List[Aggregation]:
        """Reconstruct Aggregation objects from the matched data entry."""
        try:
            # Get raw events from the entry
            raw_events = entry.get('raw_events', [])
            if not raw_events:
                return []

            # Create a single aggregation with all events
            agg_data = {
                'timestamp': entry.get('start_time', 0),
                'end_timestamp': entry.get('end_time', 0),
                'reason': 'reconstructed',
                'event_type': 'mixed',
                'request_state': True,
                'screenshot_path': entry.get('img'),
                'events': raw_events,
                'monitor': raw_events[0].get('monitor') if raw_events else None
            }

            return [Aggregation.from_dict(agg_data)]
        except Exception as e:
            print(f"Warning: failed to reconstruct aggregations: {e}")
            return []

    def _add_text_overlays(self, img: Image.Image, entry: dict,
                           deduplicate: bool, min_count: int) -> Image.Image:
        """Add caption and event summary text overlays to the image."""
        draw = ImageDraw.Draw(img)
        width, height = img.size

        caption = entry.get('caption', '')
        start_time = entry.get('start_formatted', '')
        end_time = entry.get('end_formatted', '')

        # Generate event summary using to_prompt if we have aggregations
        aggregations = self._reconstruct_aggregations(entry)
        if aggregations:
            # Use the Aggregation's to_prompt method
            time_str = f"[{start_time} - {end_time}]"
            prompt = aggregations[0].to_prompt(time_str, deduplicate=deduplicate, min_count=min_count)

            # Extract just the actions part from the prompt
            event_summary = self._extract_actions_from_prompt(prompt)
        else:
            event_summary = ""

        self._draw_caption_box(draw, width, height, caption, start_time, end_time)

        if event_summary:
            self._draw_event_box(draw, width, height, event_summary)

        return img

    def _extract_actions_from_prompt(self, prompt: str) -> str:
        """Extract the actions section from the to_prompt output."""
        lines = prompt.strip().split('\n')
        actions = []
        in_actions = False

        for line in lines:
            if 'Actions:' in line:
                in_actions = True
                continue
            if in_actions and line.strip() and line.strip() != 'No actions recorded.':
                # Remove leading tabs and format as bullet points
                action = line.strip()
                if action:
                    actions.append(f"• {action}")

        return '\n'.join(actions) if actions else ""

    def _draw_caption_box(self, draw: ImageDraw.Draw, width: int, height: int,
                          caption: str, start_time: str, end_time: str):
        """Draw the caption box at the top of the image."""
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
        """Draw the event summary box at the bottom of the image."""
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
        """Wrap text to fit within a maximum width."""
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
        """Create a video from a directory of frames using FFmpeg."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "%06d.jpg"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
