from __future__ import annotations
from typing import List, Optional, Tuple
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from record.models import ProcessedAggregation

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

# Constants for video annotation
CLICK_MARKER_RADIUS = 8
BUTTON_COLORS = {
    'Button.left': 'red',
    'left': 'red',
    'Button.right': 'blue',
    'right': 'blue',
    'Button.middle': 'green',
    'middle': 'green'
}


def list_screenshots(session_folder: Path) -> List[Path]:
    """List all screenshot files in the session folder."""
    screenshots = []
    for p in sorted((session_folder / "screenshots").iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            screenshots.append(p)
    return screenshots


def extract_timestamp_from_filename(p: Path) -> Optional[float]:
    """Extract timestamp from filename or use file modification time."""
    _timestamp_re = re.compile(r"(\d+\.\d+)")
    m = _timestamp_re.search(p.name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    try:
        return p.stat().st_mtime
    except Exception:
        return None


def compute_max_image_size(images: Iterable[Path]) -> Tuple[int, int]:
    """Compute the maximum dimensions across all images."""
    max_w = 0
    max_h = 0
    if Image is None:
        return (1920, 1200)

    for p in images:
        try:
            with Image.open(p) as im:
                w, h = im.size
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h
        except Exception:
            continue
    if max_w == 0 or max_h == 0:
        return (1920, 1200)
    return (max_w, max_h)


def extract_mouse_events(events):
    """Extract mouse click/press/release events."""
    mouse_events = []
    for event in events:
        event_type = event.get('event_type', '')
        details = event.get('details', {})
        cursor_pos = event.get('cursor_position', [])

        if event_type in ['mouse_click', 'mouse_press', 'mouse_release']:
            button = details.get('button', 'Button.left')
            mouse_events.append({
                'button': button,
                'position': cursor_pos,
                'event_type': event_type
            })

    return mouse_events


def get_cursor_movements(events):
    """Extract cursor movement from events."""
    movements = []
    prev_pos = None

    for event in events:
        cursor_pos = event.get('cursor_position', [])
        event_type = event.get('event_type', '')

        if cursor_pos and len(cursor_pos) >= 2:
            if prev_pos and prev_pos != cursor_pos:
                movements.append({
                    'start': prev_pos,
                    'end': cursor_pos,
                    'event_type': event_type
                })
            prev_pos = cursor_pos

    return movements


def screen_to_scaled_coords(screen_pos, monitor, scale, x_offset, y_offset):
    """Convert screen coordinates to scaled image coordinates."""
    x, y = screen_pos
    img_x = x - monitor['left']
    img_y = y - monitor['top']

    scaled_x = int(img_x * scale) + x_offset
    scaled_y = int(img_y * scale) + y_offset

    return scaled_x, scaled_y


def draw_cursor_arrow(img: Image.Image, start_pos, end_pos, monitor, color='orange',
                      scale=1.0, x_offset=0, y_offset=0) -> Image.Image:
    """Draw a cursor movement arrow on the image."""
    if Image is None or ImageDraw is None:
        return img

    import numpy as np
    draw = ImageDraw.Draw(img)

    start_x, start_y = screen_to_scaled_coords(start_pos, monitor, scale, x_offset, y_offset)
    end_x, end_y = screen_to_scaled_coords(end_pos, monitor, scale, x_offset, y_offset)

    if (start_x < 0 or start_y < 0 or start_x >= img.width or start_y >= img.height or
            end_x < 0 or end_y < 0 or end_x >= img.width or end_y >= img.height):
        return img

    if abs(start_x - end_x) < 2 and abs(start_y - end_y) < 2:
        return img

    line_width = max(1, int(3 * scale))
    draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=line_width)

    arrow_length = int(15 * scale)
    arrow_angle = 25

    dx = end_x - start_x
    dy = end_y - start_y
    angle = np.arctan2(dy, dx)

    arrow_angle_rad = np.radians(arrow_angle)
    x1 = end_x - arrow_length * np.cos(angle - arrow_angle_rad)
    y1 = end_y - arrow_length * np.sin(angle - arrow_angle_rad)
    x2 = end_x - arrow_length * np.cos(angle + arrow_angle_rad)
    y2 = end_y - arrow_length * np.sin(angle + arrow_angle_rad)

    draw.polygon([(end_x, end_y), (x1, y1), (x2, y2)], fill=color, outline='darkorange')

    marker_size = int(4 * scale)
    draw.ellipse([(start_x - marker_size, start_y - marker_size),
                  (start_x + marker_size, start_y + marker_size)],
                 fill='lime', outline='darkgreen', width=2)

    return img


def draw_clicks(img: Image.Image, click_positions, monitor, marker_radius: int,
                scale=1.0, x_offset=0, y_offset=0) -> Image.Image:
    """Draw click markers on the image."""
    if Image is None or ImageDraw is None:
        return img

    draw = ImageDraw.Draw(img)
    for click in click_positions:
        button = click.get('button', 'Button.left')
        position = click.get('position', click)

        img_x, img_y = screen_to_scaled_coords(position, monitor, scale, x_offset, y_offset)

        if img_x < 0 or img_y < 0 or img_x >= img.width or img_y >= img.height:
            continue

        color = BUTTON_COLORS.get(button, 'yellow')
        scaled_radius = int(marker_radius * scale)

        draw.ellipse(
            [(img_x - scaled_radius, img_y - scaled_radius),
             (img_x + scaled_radius, img_y + scaled_radius)],
            fill=color, outline='black', width=2
        )
    return img


def annotate_image(img: Image.Image, events, monitor, scale=1.0, x_offset=0, y_offset=0) -> Image.Image:
    """Annotate image with cursor movements and clicks."""
    movements = get_cursor_movements(events)
    for movement in [movements[0], movements[-1]] if len(movements) >= 2 else movements:
        img = draw_cursor_arrow(img, movement['start'], movement['end'], monitor,
                                'orange', scale, x_offset, y_offset)

    mouse_events = extract_mouse_events(events)
    img = draw_clicks(img, mouse_events, monitor, CLICK_MARKER_RADIUS, scale, x_offset, y_offset)

    return img


def scale_and_pad_image(img, target_width, target_height, background_color=(0, 0, 0)):
    """Scale and pad image to target dimensions."""
    if Image is None:
        return img, 1.0, 0, 0

    original_width, original_height = img.size

    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = min(scale_x, scale_y)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    result = Image.new('RGB', (target_width, target_height), background_color)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    result.paste(scaled_img, (x_offset, y_offset))

    return result, scale, x_offset, y_offset


def create_video_from_images(
    images: List[Path],
    output_path: Path,
    fps: int = 1,
    pad_to: Optional[Tuple[int, int]] = None,
    label_video: bool = False,
    aggregations: Optional[List[ProcessedAggregation]] = None
) -> None:
    """
    Create a video from a list of images with optional annotation.
    If label_video is True and aggregations provided, annotates each frame.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [str(p) for p in images if not p.exists()]
    if missing:
        raise RuntimeError(f"Some images are missing / unreadable (first 5): {missing[:5]}")

    with tempfile.TemporaryDirectory(prefix="imgseq_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        print(f"[video] tmpdir: {tmpdir_path} (will contain {len(images)} files)")

        for idx, src in enumerate(images):
            dst = tmpdir_path / f"{idx:06d}.jpg"

            if label_video and aggregations and idx < len(aggregations):
                agg = aggregations[idx]
                img = Image.open(src).convert('RGB')
                if agg.request.screenshot_path and hasattr(agg.request, 'monitor'):
                    monitor = getattr(agg.request, 'monitor', {'left': 0, 'top': 0})
                    if pad_to:
                        img, scale, x_offset, y_offset = scale_and_pad_image(img, pad_to[0], pad_to[1])
                    else:
                        scale, x_offset, y_offset = 1.0, 0, 0

                    img = annotate_image(img, agg.events, monitor, scale, x_offset, y_offset)
                img.save(dst)
            else:
                shutil.copy2(src, dst)

        expected_first = tmpdir_path / "000000.jpg"
        if not expected_first.exists():
            raise RuntimeError(f"Expected first file {expected_first} does not exist in temp dir")

        vf_parts = []
        if pad_to:
            pad_w, pad_h = pad_to
            vf_parts.append(
                f"scale=iw*min({pad_w}/iw\\,{pad_h}/ih):ih*min({pad_w}/iw\\,{pad_h}/ih),pad={pad_w}:{pad_h}:(ow-iw)/2:(oh-ih)/2"
            )
        vf = ",".join(vf_parts) if vf_parts else None

        cmd = [
            "ffmpeg",
            "-y",
            "-start_number", "0",
            "-framerate", str(fps),
            "-i", str(tmpdir_path / "%06d.jpg"),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd.append(str(output_path))

        print(f"[video] Running ffmpeg -> {output_path} ({len(images)} frames, fps={fps})")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("[video] ffmpeg failed. stderr:\n", res.stderr)
            print("[video] ffmpeg stdout:\n", res.stdout)
            raise RuntimeError("ffmpeg failed creating video")
        else:
            print("[video] ffmpeg finished successfully")


def split_video(video_path: Path, chunk_duration: int, out_dir: Path) -> List[Path]:
    """Split a video into chunks of specified duration."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(video_path)
    if duration is None:
        raise RuntimeError("Could not get video duration")
    num_chunks = math.ceil(duration / float(chunk_duration))
    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_duration
        out_path = out_dir / f"chunk_{i:03d}.mp4"
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start),
            '-i', str(video_path),
            '-t', str(chunk_duration),
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-an',
            str(out_path)
        ]
        print(f"[video] Creating chunk #{i} -> {out_path}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"ffmpeg chunking error: {r.stderr}")
            continue
        chunk_paths.append(out_path)
    return chunk_paths


def get_video_duration(video_path: Path) -> Optional[float]:
    """Get the duration of a video file in seconds."""
    try:
        r = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(video_path)
        ], capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return None
