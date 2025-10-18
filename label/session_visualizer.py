import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


class SessionVisualizer:

    def __init__(self):
        self._load_fonts()

    def _load_fonts(self):
        """Load fonts with fallback to default."""
        try:
            self.font_large = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
            )
            self.font_medium = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
            )
            self.font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
            )
        except Exception:
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def visualize_session(
        self,
        session_dir: Path,
        output_video_path: Optional[Path] = None,
        fps: int = 1
    ) -> Path:
        """
        Create annotated video from session's matched_captions.jsonl.

        Args:
            session_dir: Path to session directory
            output_video_path: Path for output video (default: session_dir/annotated_session.mp4)
            fps: Frames per second

        Returns:
            Path to created video
        """
        if output_video_path is None:
            output_video_path = session_dir / "annotated_session.mp4"

        matched_captions_path = session_dir / "matched_captions.jsonl"
        if not matched_captions_path.exists():
            raise RuntimeError(f"matched_captions.jsonl not found in {session_dir}")

        matched_data = []
        with open(matched_captions_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    matched_data.append(json.loads(line))

        print(f"[Visualizer] Processing {len(matched_data)} frames")

        with tempfile.TemporaryDirectory(prefix="session_viz_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            for idx, entry in enumerate(tqdm(matched_data, desc="Annotating frames")):
                img_path = entry.get('img')

                if img_path is None or not Path(img_path).exists():
                    if img_path:
                        img_path = session_dir / img_path

                    if not img_path or not Path(img_path).exists():
                        print(f"[Visualizer] Warning: Image {img_path} not found for frame {idx}")
                        continue

                img = Image.open(img_path).convert('RGB')
                annotated_img = self._annotate_frame(
                    img,
                    entry['caption'],
                    entry['raw_events'],
                    entry.get('start_formatted', ''),
                    entry.get('end_formatted', '')
                )

                frame_path = tmpdir_path / f"{idx:06d}.jpg"
                annotated_img.save(frame_path)

            print("[Visualizer] Creating video...")
            self._create_video_from_frames(tmpdir_path, output_video_path, fps)

        print(f"[Visualizer] ✓ Video saved to {output_video_path}")
        return output_video_path

    def _annotate_frame(
        self,
        img: Image.Image,
        caption: str,
        events: List[Dict],
        start_time: str,
        end_time: str
    ) -> Image.Image:
        """Annotate a single frame with caption and event summary."""
        draw = ImageDraw.Draw(img)
        width, height = img.size

        event_summary = self._generate_event_summary(events)

        self._draw_caption_box(draw, width, height, caption, start_time, end_time)

        if event_summary:
            self._draw_event_box(draw, width, height, event_summary)

        return img

    def _generate_event_summary(self, events: List[Dict]) -> str:
        """Generate human-readable event summary."""
        if not events:
            return ""

        compressed = self._compress_events(events)

        actions = []
        keys_pressed = []

        for event in compressed:
            event_type = event.get("event_type")

            if event_type == "key_press":
                key = event.get("details", {}).get("key", "unknown")
                key = key.replace("Key.", "")
                if key:
                    keys_pressed.append(key)

            elif event_type == "mouse_down":
                if keys_pressed:
                    actions.append(
                        f"Key: {keys_pressed[0]}" if len(keys_pressed) == 1
                        else f"Keys: {'+'.join(keys_pressed)}"
                    )
                    keys_pressed.clear()

                details = event.get("details", {})
                button = details.get("button", "unknown").replace("Button.", "")
                double_click = details.get("double_click", False)

                click_text = f"Click {button}"
                if double_click:
                    click_text += " (2x)"
                actions.append(click_text)

            elif event_type == "mouse_scroll":
                if keys_pressed:
                    actions.append(
                        f"Key: {keys_pressed[0]}" if len(keys_pressed) == 1
                        else f"Keys: {'+'.join(keys_pressed)}"
                    )
                    keys_pressed.clear()

                direction = event.get("_direction") or self._convert_scroll_direction(
                    event.get("details", {})
                )
                actions.append(f"Scroll {direction}")

            elif event_type == "mouse_move":
                if keys_pressed:
                    actions.append(
                        f"Key: {keys_pressed[0]}" if len(keys_pressed) == 1
                        else f"Keys: {'+'.join(keys_pressed)}"
                    )
                    keys_pressed.clear()

                actions.append("Move mouse")

        if keys_pressed:
            actions.append(
                f"Key: {keys_pressed[0]}" if len(keys_pressed) == 1
                else f"Keys: {'+'.join(keys_pressed)}"
            )

        # Limit to most important actions
        if len(actions) > 5:
            actions = actions[:5] + [f"... and {len(actions) - 5} more"]

        return "\n".join([f"• {action}" for action in actions])

    def _compress_events(self, events: List[Dict]) -> List[Dict]:
        """Compress consecutive similar events."""
        if not events:
            return []

        compressed = []
        i = 0
        n = len(events)

        while i < n:
            e = events[i]
            et = e.get("event_type")

            if et == "mouse_move":
                # Skip to last mouse_move in sequence
                j = i + 1
                while j < n and events[j].get("event_type") == "mouse_move":
                    j += 1
                compressed.append(events[j - 1])
                i = j

            elif et == "mouse_scroll":
                # Compress scrolls by direction
                j = i
                last_dir = None
                while j < n and events[j].get("event_type") == "mouse_scroll":
                    details = events[j].get("details", {})
                    dir_ = self._convert_scroll_direction(details)
                    if dir_ != last_dir:
                        compressed.append({
                            "event_type": "mouse_scroll",
                            "details": details,
                            "_direction": dir_,
                        })
                        last_dir = dir_
                    j += 1
                i = j

            else:
                compressed.append(e)
                i += 1

        return compressed

    def _convert_scroll_direction(self, scroll_data) -> str:
        """Convert scroll data to readable direction."""
        if isinstance(scroll_data, dict):
            dx = scroll_data.get("dx", 0)
            dy = scroll_data.get("dy", 0)
        elif isinstance(scroll_data, (list, tuple)) and len(scroll_data) >= 2:
            dx, dy = scroll_data[0], scroll_data[1]
        else:
            return "unknown"

        directions = []
        if dy > 0:
            directions.append("up")
        elif dy < 0:
            directions.append("down")
        if dx > 0:
            directions.append("right")
        elif dx < 0:
            directions.append("left")

        return " ".join(directions) if directions else "unknown"

    def _draw_caption_box(
        self,
        draw: ImageDraw.Draw,
        width: int,
        height: int,
        caption: str,
        start_time: str,
        end_time: str
    ):
        """Draw caption box at top of frame."""
        padding = 20

        # Add timestamp to caption
        time_str = f"[{start_time} - {end_time}]"
        full_text = f"{time_str}\n{caption}"

        # Wrap text
        lines = self._wrap_text(full_text, width - 100, self.font_medium)

        line_height = 30
        box_height = len(lines) * line_height + 2 * padding
        box_width = width - 40

        box_x = 20
        box_y = 20

        # Draw semi-transparent background
        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [box_x, box_y, box_x + box_width, box_y + box_height],
            fill=(0, 100, 200, 200)
        )

        # Composite overlay
        img_rgba = draw._image.convert('RGBA')
        img_rgba = Image.alpha_composite(img_rgba, overlay)
        draw._image.paste(img_rgba.convert('RGB'))

        # Draw text
        y_offset = box_y + padding
        for i, line in enumerate(lines):
            font = self.font_large if i == 0 else self.font_medium
            draw.text((box_x + padding, y_offset), line, fill='white', font=font)
            y_offset += line_height

    def _draw_event_box(
        self,
        draw: ImageDraw.Draw,
        width: int,
        height: int,
        event_summary: str
    ):
        """Draw event summary box at bottom of frame."""
        padding = 15

        lines = event_summary.split('\n')
        line_height = 22
        box_height = len(lines) * line_height + 2 * padding
        box_width = width - 40

        box_x = 20
        box_y = height - box_height - 20

        # Draw semi-transparent background
        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [box_x, box_y, box_x + box_width, box_y + box_height],
            fill=(50, 50, 50, 180)
        )

        # Composite overlay
        img_rgba = draw._image.convert('RGBA')
        img_rgba = Image.alpha_composite(img_rgba, overlay)
        draw._image.paste(img_rgba.convert('RGB'))

        # Draw text
        y_offset = box_y + padding
        for line in lines:
            draw.text((box_x + padding, y_offset), line, fill='white', font=self.font_small)
            y_offset += line_height

    def _wrap_text(self, text: str, max_width: int, font) -> List[str]:
        """Wrap text to fit within max_width."""
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph:
                lines.append('')
                continue

            words = paragraph.split()
            current_line = []

            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = font.getbbox(test_line)
                if bbox[2] - bbox[0] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]

            if current_line:
                lines.append(' '.join(current_line))

        return lines

    def _create_video_from_frames(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: int
    ):
        """Create video from numbered frames."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "%06d.jpg"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Visualizer] FFmpeg error: {result.stderr}")
            raise RuntimeError("Failed to create video")


def visualize_session_main(session_dir: Path, output_video: Optional[Path] = None):
    """Main function to visualize a session."""
    session_dir = Path(session_dir)

    if not session_dir.exists():
        raise RuntimeError(f"Session directory not found: {session_dir}")

    visualizer = SessionVisualizer()
    output_path = visualizer.visualize_session(session_dir, output_video)

    print(f"\n✓ Session visualization complete: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize session with captions and events"
    )
    parser.add_argument("session_dir", type=Path, help="Session directory")
    parser.add_argument("--output", type=Path, default=None, help="Output video path")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second")

    args = parser.parse_args()

    visualizer = SessionVisualizer()
    visualizer.visualize_session(args.session_dir, args.output, args.fps)
