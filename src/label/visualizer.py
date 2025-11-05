from pathlib import Path
from typing import List, Optional, Dict
import json
import tempfile
import subprocess
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from label.video import annotate_image, scale_and_pad, apply_pending_movement, extract_pending_movement, compute_max_size
from label.models import Aggregation, ImagePath


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

        # Load the original aggregations
        aggregations_path = session_dir / "aggregations.jsonl"
        aggs = []
        with open(aggregations_path, 'r') as f:
            for line in f:
                if line.strip():
                    aggs.append(Aggregation.from_dict(json.loads(line)))
        # add up all aggregations with identical screenshot_paths
        all_aggregations = []
        for agg in aggs:
            if all_aggregations and all_aggregations[-1].screenshot_path == agg.screenshot_path:
                combined_agg = all_aggregations[-1] + agg
                all_aggregations[-1] = combined_agg
            else:
                all_aggregations.append(agg)

        # Load matched data for captions
        matched_data = []
        with open(matched_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    matched_data.append(json.loads(line))

        # Build aggregation index for quick lookup
        agg_by_screenshot: Dict[str, Aggregation] = {}
        for agg in all_aggregations:
            if agg.screenshot_path:
                agg_by_screenshot[str(Path(agg.screenshot_path).name)] = agg

        # Collect frames to render based on matched data
        frames_to_render = []
        for entry in matched_data:
            img_path = entry.get('img')
            if not img_path:
                continue

            p = Path(img_path)
            if not p.exists():
                p = session_dir / "screenshots" / p.name

            if not p.exists():
                continue

            # Find the aggregation for this screenshot
            agg = agg_by_screenshot.get(p.name)
            if not agg:
                print(f"Warning: No aggregation found for {p.name}")
                continue

            # Get all aggregations in the matched range
            start_idx = entry.get('start_index')
            end_idx = entry.get('end_index')

            if start_idx is not None and end_idx is not None:
                matched_aggs = all_aggregations[start_idx:end_idx + 1]
            else:
                matched_aggs = [agg]

            frames_to_render.append((p, agg, matched_aggs, entry))

        if not frames_to_render:
            raise RuntimeError("No valid frames to render")

        # Compute target size from ALL original screenshots (like master video does)
        all_screenshot_paths = [Path(agg.screenshot_path) for agg in all_aggregations
                                if agg.screenshot_path and Path(agg.screenshot_path).exists()]

        # If paths are relative, resolve them relative to session_dir
        resolved_paths = []
        for p in all_screenshot_paths:
            if not p.exists():
                p = session_dir / "screenshots" / p.name
            if p.exists():
                resolved_paths.append(p)

        target_w, target_h = compute_max_size(resolved_paths)

        with tempfile.TemporaryDirectory(prefix="viz_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Initialize pending movement tracking
            pending_movement = []

            for idx, (img_path, primary_agg, matched_aggs, caption_entry) in enumerate(tqdm(frames_to_render, desc="Annotating")):
                # Load the original screenshot using ImagePath (handles relative paths)
                img_path_obj = ImagePath(img_path, session_dir)
                img = img_path_obj.load()

                # Scale and pad to target size (same as master video creation)
                canvas, scale, x_offset, y_offset = scale_and_pad(img, target_w, target_h)

                # Apply arrow annotations if requested
                if self.annotate:
                    # Apply pending movement from previous frame to the primary aggregation
                    agg_with_pending = apply_pending_movement(primary_agg, pending_movement)

                    # Annotate with the primary aggregation's events (with pending movement)
                    canvas = annotate_image(canvas, agg_with_pending, scale, x_offset, y_offset)

                    # Extract pending movement from the LAST aggregation in the matched range
                    # This ensures continuous cursor tracking across caption boundaries
                    if matched_aggs:
                        last_agg = matched_aggs[-1]
                        # Apply pending movement to last agg to get accurate extraction
                        if last_agg == primary_agg:
                            pending_movement = extract_pending_movement(agg_with_pending)
                        else:
                            pending_movement = extract_pending_movement(last_agg)
                    else:
                        pending_movement = extract_pending_movement(agg_with_pending)
                else:
                    pending_movement = []

                # Add text overlays
                annotated = self._add_text_overlays(canvas, caption_entry, matched_aggs,
                                                    deduplicate_events, min_event_count)

                frame_path = tmpdir_path / f"{idx:06d}.jpg"
                annotated.save(frame_path)

            self._create_video(tmpdir_path, output_path, fps)

        return output_path

    def _add_text_overlays(self, img: Image.Image, caption_entry: dict,
                           matched_aggs: List[Aggregation],
                           deduplicate: bool, min_count: int) -> Image.Image:
        """Add caption and event summary text overlays to the image."""
        draw = ImageDraw.Draw(img)
        width, height = img.size

        caption = caption_entry.get('caption', '')
        start_time = caption_entry.get('start_formatted', '')
        end_time = caption_entry.get('end_formatted', '')

        # Generate event summary from all matched aggregations
        time_str = f"[{start_time} - {end_time}]"

        # Combine events from all matched aggregations
        if matched_aggs:
            # Create a combined prompt from all aggregations
            all_prompts = []
            for agg in matched_aggs:
                prompt = agg.to_prompt("", deduplicate=deduplicate, min_count=min_count)
                actions = self._extract_actions_from_prompt(prompt)
                if actions:
                    all_prompts.append(actions)

            event_summary = '\n'.join(all_prompts)
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
                if action and not action.startswith('•'):
                    actions.append(f"• {action}")
                elif action:
                    actions.append(action)

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
