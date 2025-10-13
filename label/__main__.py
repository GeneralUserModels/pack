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

from label.clients import PromptClient, GeminiPromptClient, LocalQwenPromptClient
from record.models import ProcessedAggregation, AggregationRequest

load_dotenv()

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


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


def create_video_from_images(images: List[Path], output_path: Path, fps: int = 1, pad_to: Optional[Tuple[int, int]] = None) -> None:
    """
    Create a video from a list of images by copying them into a temporary folder as
    000000.jpg, 000001.jpg, ... then calling ffmpeg with the %06d pattern.
    Includes debug prints and a safety check to ensure files exist before ffmpeg runs.
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
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                raise RuntimeError(f"Failed to copy {src} -> {dst}: {e}")

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
                print(f"Skipping invalid json line: {line[:80]}")
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
):
    """Main orchestration:
    - Reads screenshots from session_folder/screenshots
    - Builds a full video (1 fps default) showing each screenshot for 1s
    - Splits into chunks
    - Loads aggregations.jsonl and chunk them according to epoch time windows
    - For each chunk: upload video chunk (if client supports it) and send prompt built from aggregations
    - Save JSON results under out_chunks_dir
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
        create_video_from_images(image_paths, master_video, fps=fps, pad_to=pad_to)

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
        for i, a in enumerate(this_aggs):
            mss = f"{i // 60:02}:{i % 60:02}"
            prompts.append(a.to_prompt(mss))
        full_prompt = "".join(prompts)
        with open(Path(__file__).parent / "prompt.txt", 'r') as f:
            prompt_template = f.read()
        full_prompt = prompt_template.replace("{{LOGS}}", full_prompt)

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

        info = {
            "chunk_index": i,
            "video_chunk": str(vpath),
            "aggregations_count": len(this_aggs),
            "aggregations": [a.to_dict() for a in this_aggs],
            "prompt": full_prompt,
            "result": saved_resp,
        }
        with open(chunk_out_dir / "result.json", 'w') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        results.append(info)

    with open(out_chunks_dir / "all_chunks_summary.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done. Results saved to {out_chunks_dir}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True, help="Path to session folder (contains screenshots/ and aggregations.jsonl)")
    p.add_argument("--agg-jsonl", default="aggregations.jsonl", help="Filename of aggregations jsonl inside session folder")
    p.add_argument("--chunk-duration", type=int, default=60, help="Chunk duration in seconds")
    p.add_argument("--fps", type=int, default=1, help="Frames per second for master video (1 => 1s per screenshot)")
    p.add_argument("--prompt-client", choices=["gemini", "local"], default="gemini")
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
        client = GeminiPromptClient(api_key=api_key)
    else:
        client = LocalQwenPromptClient()

    process_session(session, agg_path, out, chunk_duration=args.chunk_duration, fps=args.fps, prompt_client=client)


if __name__ == '__main__':
    main()
