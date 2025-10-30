from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
import tempfile
import shutil
import math
from PIL import Image, ImageDraw
import numpy as np

from label.models import Aggregation, ImagePath


BUTTON_COLORS = {
    'left': 'red',
    'right': 'blue',
    'middle': 'green'
}


def get_video_duration(video_path: Path) -> Optional[float]:
    try:
        r = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(video_path)
        ], capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def split_video(video_path: Path, chunk_duration: int, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(video_path)
    if duration is None:
        raise RuntimeError("Could not get video duration")

    num_chunks = math.ceil(duration / float(chunk_duration))
    chunk_paths = []

    for i in range(num_chunks):
        start = i * chunk_duration
        out_path = out_dir / f"{i:03d}.mp4"

        cmd = [
            'ffmpeg', '-y', '-ss', str(start), '-i', str(video_path),
            '-t', str(chunk_duration), '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            '-an', str(out_path)
        ]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            chunk_paths.append(out_path)

    return chunk_paths


def compute_max_size(image_paths: List[Path]) -> Tuple[int, int]:
    max_w, max_h = 0, 0

    for p in image_paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
            max_w = max(max_w, w)
            max_h = max(max_h, h)
        except Exception:
            continue

    return (max_w, max_h) if max_w > 0 else (1920, 1200)


def scale_and_pad(img: Image.Image, target_w: int, target_h: int) -> Tuple[Image.Image, float, int, int]:
    orig_w, orig_h = img.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    result = Image.new('RGB', (target_w, target_h), (0, 0, 0))

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result.paste(scaled, (x_offset, y_offset))

    return result, scale, x_offset, y_offset


def screen_to_image_coords(screen_pos, monitor, scale, x_offset, y_offset):
    x, y = screen_pos
    img_x = (x - monitor['left']) * scale + x_offset
    img_y = (y - monitor['top']) * scale + y_offset
    return int(img_x), int(img_y)


def annotate_image(img: Image.Image, agg: Aggregation, scale: float = 1.0,
                   x_offset: int = 0, y_offset: int = 0) -> Image.Image:
    if not agg.monitor or not agg.events:
        return img

    draw = ImageDraw.Draw(img)
    monitor = agg.monitor

    movements = []
    prev_pos = None
    mpos_events = [e for e in agg.events if e.cursor_position and len(e.cursor_position) >= 2]
    for event in ([mpos_events[0], mpos_events[-1]] if len(mpos_events) >= 2 else mpos_events):
        if prev_pos and prev_pos != event.cursor_position:
            movements.append({'start': prev_pos, 'end': event.cursor_position})
        prev_pos = event.cursor_position

    print(f"Drawing movements: {movements}")
    for mv in movements:
        draw_arrow(draw, img.size, mv['start'], mv['end'], monitor, scale, x_offset, y_offset)

    clicks = [e for e in agg.events if e.is_mouse_event]
    for click in clicks:
        pos = click.cursor_position
        if not pos or len(pos) < 2:
            continue

        img_x, img_y = screen_to_image_coords(pos, monitor, scale, x_offset, y_offset)
        if 0 <= img_x < img.width and 0 <= img_y < img.height:
            color = BUTTON_COLORS.get(click.details.button, 'yellow')
            radius = int(8 * scale)
            draw.ellipse(
                [(img_x - radius, img_y - radius), (img_x + radius, img_y + radius)],
                fill=color, outline='black', width=2
            )

    return img


def draw_arrow(draw, img_size, start_pos, end_pos, monitor, scale, x_offset, y_offset):
    start_x, start_y = screen_to_image_coords(start_pos, monitor, scale, x_offset, y_offset)
    end_x, end_y = screen_to_image_coords(end_pos, monitor, scale, x_offset, y_offset)

    width, height = img_size
    if not (0 <= start_x < width and 0 <= start_y < height and
            0 <= end_x < width and 0 <= end_y < height):
        return

    if abs(start_x - end_x) < 2 and abs(start_y - end_y) < 2:
        return

    line_width = max(1, int(3 * scale))
    draw.line([(start_x, start_y), (end_x, end_y)], fill='orange', width=line_width)

    arrow_length = int(25 * scale)
    dx, dy = end_x - start_x, end_y - start_y
    angle = np.arctan2(dy, dx)
    arrow_angle_rad = np.radians(40)

    x1 = end_x - arrow_length * np.cos(angle - arrow_angle_rad)
    y1 = end_y - arrow_length * np.sin(angle - arrow_angle_rad)
    x2 = end_x - arrow_length * np.cos(angle + arrow_angle_rad)
    y2 = end_y - arrow_length * np.sin(angle + arrow_angle_rad)

    draw.polygon([(end_x, end_y), (x1, y1), (x2, y2)], fill='orange', outline='darkorange')

    marker_size = int(8 * scale)
    draw.ellipse(
        [(start_x - marker_size, start_y - marker_size),
         (start_x + marker_size, start_y + marker_size)],
        fill='lime', outline='darkgreen', width=2
    )


def create_video(
    image_paths: List[Path],
    output_path: Path,
    fps: int = 1,
    pad_to: Optional[Tuple[int, int]] = None,
    annotate: bool = False,
    aggregations: Optional[List[Aggregation]] = None,
    session_dir: Optional[Path] = None
):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="video_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for idx, src in enumerate(image_paths):
            dst = tmpdir_path / f"{idx:06d}.jpg"

            if annotate and aggregations and idx < len(aggregations):
                agg = aggregations[idx]
                img_path = ImagePath(src, session_dir)
                img = img_path.load()

                if pad_to:
                    img, scale, x_off, y_off = scale_and_pad(img, pad_to[0], pad_to[1])
                else:
                    scale, x_off, y_off = 1.0, 0, 0

                img = annotate_image(img, agg, scale, x_off, y_off)
                img.save(dst)
            else:
                shutil.copy2(src, dst)

        vf_parts = []
        if pad_to:
            w, h = pad_to
            vf_parts.append(f"scale=iw*min({w}/iw\\,{h}/ih):ih*min({w}/iw\\,{h}/ih),pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")

        cmd = [
            'ffmpeg', '-y', '-start_number', '0', '-framerate', str(fps),
            '-i', str(tmpdir_path / '%06d.jpg'), '-c:v', 'libx264',
            '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart'
        ]

        if vf_parts:
            cmd += ['-vf', ','.join(vf_parts)]

        cmd.append(str(output_path))

        subprocess.run(cmd, capture_output=True, check=True)
