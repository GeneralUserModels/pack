from __future__ import annotations
from typing import List, Optional, Tuple
import argparse
import json
import math
import os
import shutil
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable
from dotenv import load_dotenv

from label.clients import PromptClient, GeminiPromptClient
from label.clients.qwen_3_vl_client import Qwen3VLPromptClient
from label.vllm_server_manager import VLLMServerManager
from record.models import ProcessedAggregation, AggregationRequest

load_dotenv()

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
    screenshots = []
    for p in sorted((session_folder / "screenshots").iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            screenshots.append(p)
    return screenshots


_timestamp_re = re.compile(r"(\d+\.\d+)")


def extract_timestamp_from_filename(p: Path) -> Optional[float]:
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
        cursor_pos = event.get('cursor_pos', [])

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
        cursor_pos = event.get('cursor_pos', [])
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


def draw_cursor_arrow(img: Image.Image, start_pos, end_pos, monitor, color='orange', scale=1.0, x_offset=0, y_offset=0) -> Image.Image:
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
    draw.ellipse([(start_x - marker_size, start_y - marker_size), (start_x + marker_size, start_y + marker_size)],
                 fill='lime', outline='darkgreen', width=2)

    return img


def draw_clicks(img: Image.Image, click_positions, monitor, marker_radius: int, scale=1.0, x_offset=0, y_offset=0) -> Image.Image:
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
    for movement in movements:
        img = draw_cursor_arrow(img, movement['start'], movement['end'], monitor, 'orange', scale, x_offset, y_offset)

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


def create_video_from_images(images: List[Path], output_path: Path, fps: int = 1, pad_to: Optional[Tuple[int, int]] = None, label_video: bool = False, aggregations: Optional[List[ProcessedAggregation]] = None) -> None:
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
    try:
        r = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(video_path)
        ], capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def load_aggregations_jsonl(path: Path) -> List[ProcessedAggregation]:
    out = []
    with open(path, 'r') as f:
        for raw_line in f:
            try:
                line = json.loads(raw_line.strip())
            except Exception:
                print(f"Skipping invalid json line")
                continue

            r = AggregationRequest(
                timestamp=line['timestamp'],
                end_timestamp=None,
                reason=line['reason'],
                event_type=line['event_type'],
                is_start=line['is_start'],
                screenshot=None,
                screenshot_path=line['screenshot_path']
            )
            events = line['events']
            out.append(ProcessedAggregation(request=r, events=events))
    return out


def chunk_aggregations(aggs: List[ProcessedAggregation], chunk_start: float, chunk_duration: int) -> List[List[ProcessedAggregation]]:
    if not aggs:
        return []
    min_ts = min(a.request.timestamp for a in aggs)
    max_ts = max(a.request.timestamp for a in aggs)
    if chunk_start is None:
        chunk_start = min_ts
    num_chunks = max(1, math.ceil((max_ts - chunk_start) / float(chunk_duration)))
    chunks: List[List[ProcessedAggregation]] = [[] for _ in range(num_chunks)]
    for a in aggs:
        idx = int(math.floor((a.request.timestamp - chunk_start) / float(chunk_duration)))
        if idx < 0:
            idx = 0
        if idx >= len(chunks):
            extend_by = idx - len(chunks) + 1
            chunks.extend([[] for _ in range(extend_by)])
        chunks[idx].append(a)
    return chunks


def process_session(
    session_folder: Path,
    agg_jsonl: Path,
    out_chunks_dir: Path,
    chunk_duration: int = 180,
    fps: int = 1,
    prompt_client: PromptClient = None,
    use_existing_video: Optional[Path] = None,
    label_video: bool = False,
):
    """Main orchestration:
    - Reads screenshots from session_folder/screenshots
    - Builds a full video showing each screenshot for 1s
    - Splits into chunks
    - Loads aggregations.jsonl and chunk them according to epoch time windows
    - For each chunk: upload video chunk and send prompt built from aggregations
    - Save aggregations and generation results in separate files
    """
    screenshots = list_screenshots(session_folder)
    if not screenshots:
        raise RuntimeError("No screenshots found in session folder")

    images_and_ts = [(p, extract_timestamp_from_filename(p)) for p in screenshots]
    images_and_ts = [it for it in images_and_ts if it[1] is not None]
    images_and_ts.sort(key=lambda x: x[1])

    global_start = images_and_ts[0][1]

    master_video = out_chunks_dir / "full_session.mp4"
    pad_to = compute_max_image_size([p for p, _ in images_and_ts])
    image_paths = [p for p, _ in images_and_ts]

    if use_existing_video and use_existing_video.exists():
        print(f"Using existing video: {use_existing_video}")
        master_video = use_existing_video
    else:
        print(f"Creating master video with {len(image_paths)} images; pad_to={pad_to}")
        aggs = load_aggregations_jsonl(agg_jsonl)
        create_video_from_images(image_paths, master_video, fps=fps, pad_to=pad_to, label_video=label_video, aggregations=aggs if label_video else None)

    chunks_dir = out_chunks_dir / "video_chunks"
    video_chunks = split_video(master_video, chunk_duration, chunks_dir)

    aggs = load_aggregations_jsonl(agg_jsonl)
    agg_chunks = chunk_aggregations(aggs, chunk_start=global_start, chunk_duration=chunk_duration)

    results = []
    for i, vpath in enumerate(video_chunks):
        print(f"Processing chunk {i} -> {vpath}")
        chunk_out_dir = out_chunks_dir / f"chunk_{i:03d}"
        chunk_out_dir.mkdir(parents=True, exist_ok=True)
        this_aggs = agg_chunks[i] if i < len(agg_chunks) else []

        prompts = []
        for j, a in enumerate(this_aggs):
            mss = f"{j // 60:02}:{j % 60:02}"
            prompts.append(a.to_prompt(mss))
        full_prompt = "".join(prompts)
        with open(Path(__file__).parent / "prompt.txt", 'r') as f:
            prompt_template = f.read()
        full_prompt = prompt_template.replace("{{LOGS}}", full_prompt)

        # Save aggregations separately
        agg_output = chunk_out_dir / "aggregations.json"
        with open(agg_output, 'w') as f:
            json.dump([a.to_dict() for a in this_aggs], f, indent=2, ensure_ascii=False)

        file_descriptor = None
        if prompt_client is not None:
            try:
                file_descriptor = prompt_client.upload_file(str(vpath))
            except Exception as e:
                print(f"Warning: upload failed for chunk {i}: {e}")
                file_descriptor = None

        saved_resp = None
        if prompt_client is not None:
            try:
                resp = prompt_client.generate_content(full_prompt, file_descriptor=file_descriptor, response_mime_type="application/json")
                if hasattr(resp, 'json'):
                    try:
                        body = resp.json
                        if callable(body):
                            body = body()
                    except Exception:
                        body = None
                else:
                    try:
                        body = json.loads(getattr(resp, 'text', str(resp)))
                    except Exception:
                        body = getattr(resp, 'text', str(resp))
                saved_resp = body
            except Exception as e:
                print(f"Model call failed for chunk {i}: {e}")
                saved_resp = {"error": str(e)}
        else:
            print("No prompt client configured; skipping model call")

        # Save generation result separately
        result_output = chunk_out_dir / "generation_result.json"
        with open(result_output, 'w') as f:
            json.dump({
                "chunk_index": i,
                "video_chunk": str(vpath),
                "aggregations_count": len(this_aggs),
                "result": saved_resp,
            }, f, indent=2, ensure_ascii=False)

        # Save prompt for reference
        prompt_output = chunk_out_dir / "prompt.txt"
        with open(prompt_output, 'w') as f:
            f.write(full_prompt)

        results.append({
            "chunk_index": i,
            "video_chunk": str(vpath),
            "aggregations_count": len(this_aggs),
            "aggregations_file": str(agg_output),
            "result_file": str(result_output),
            "prompt_file": str(prompt_output),
        })

    with open(out_chunks_dir / "all_chunks_summary.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done. Results saved to {out_chunks_dir}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Process session recordings with VLM labeling")
    p.add_argument("--session", required=True, help="Path to session folder (contains screenshots/ and aggregations.jsonl)")
    p.add_argument("--agg-jsonl", default="aggregations.jsonl", help="Filename of aggregations jsonl inside session folder")
    p.add_argument("--chunk-duration", type=int, default=60, help="Chunk duration in seconds")
    p.add_argument("--fps", type=int, default=1, help="Frames per second for master video")
    p.add_argument("--prompt-client", choices=["gemini", "qwen3vl"], default="gemini", help="Which VLM client to use")

    # Qwen3VL options
    p.add_argument("--qwen-model-path", default="Qwen/Qwen3-VL-30B-A3B-Thinking-FP8", help="Qwen model path or HuggingFace ID")
    p.add_argument("--vllm-port", type=int, default=8000, help="Port for vLLM server")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization (0.0-1.0)")
    p.add_argument("--max-model-len", type=int, default=None, help="Maximum model context length")
    p.add_argument("--server-startup-timeout", type=int, default=600, help="Timeout for vLLM server startup (seconds)")

    # Video options
    p.add_argument("--label-video", action="store_true", help="Annotate video frames with mouse movements and clicks")

    return p.parse_args()


def main():
    args = parse_args()
    session = Path(args.session)
    agg_path = session / args.agg_jsonl
    out = session / f"chunks_{args.chunk_duration}"
    out.mkdir(parents=True, exist_ok=True)

    if args.prompt_client == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key is None:
            raise RuntimeError('GEMINI_API_KEY environment variable not set (required for Gemini client)')

        print("[Main] Using Gemini client")
        client = GeminiPromptClient(api_key=api_key)
        process_session(
            session, agg_path, out,
            chunk_duration=args.chunk_duration,
            fps=args.fps,
            prompt_client=client,
            label_video=args.label_video
        )

    elif args.prompt_client == 'qwen3vl':
        print("[Main] Using Qwen3-VL client with managed vLLM server")

        # Start managed vLLM server
        with VLLMServerManager(
            model_path=args.qwen_model_path,
            port=args.vllm_port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        ) as server:
            # Wait for server with custom timeout
            server._wait_for_server(timeout=args.server_startup_timeout)

            # Create client pointing to the managed server
            client = Qwen3VLPromptClient(
                base_url=server.get_base_url(),
                model_name=args.qwen_model_path,
            )

            # Process the session
            process_session(
                session, agg_path, out,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                prompt_client=client,
                label_video=args.label_video
            )

        print("[Main] vLLM server stopped automatically")

    else:
        raise ValueError(f"Unknown prompt client: {args.prompt_client}")


if __name__ == '__main__':
    main()
