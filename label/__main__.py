from __future__ import annotations
from typing import List, Optional
import argparse
import json
import math
import os
import re
from pathlib import Path
from dotenv import load_dotenv

from label.clients import PromptClient, GeminiPromptClient, Qwen3VLPromptClient
from label.video import (
    create_video_from_images,
    split_video,
    compute_max_image_size
)
from record.models import ProcessedAggregation, AggregationRequest

load_dotenv()


def list_screenshots(session_folder: Path) -> List[Path]:
    """List all screenshot files in the session folder."""
    screenshots = []
    for p in sorted((session_folder / "screenshots").iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            screenshots.append(p)
    return screenshots


_timestamp_re = re.compile(r"(\d+\.\d+)")


def extract_timestamp_from_filename(p: Path) -> Optional[float]:
    """Extract timestamp from filename or use file modification time."""
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


def load_aggregations_jsonl(path: Path) -> List[ProcessedAggregation]:
    """Load aggregations from a JSONL file."""
    out = []
    with open(path, 'r') as f:
        for raw_line in f:
            try:
                line = json.loads(raw_line.strip())
            except Exception:
                print("Skipping invalid json line")
                continue

            r = AggregationRequest(
                timestamp=line['timestamp'],
                end_timestamp=None,
                reason=line['reason'],
                event_type=line['event_type'],
                is_start=line['is_start'],
                screenshot=None,
                screenshot_path=line['screenshot_path'],
                monitor=line.get('monitor'),
                burst_id=line.get('burst_id'),
            )
            events = line['events']
            out.append(ProcessedAggregation(request=r, events=events))
    return out


def chunk_aggregations(
    aggs: List[ProcessedAggregation],
    chunk_start: float,
    chunk_duration: int
) -> List[List[ProcessedAggregation]]:
    """Partition aggregations into time-based chunks."""
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
        create_video_from_images(
            image_paths,
            master_video,
            fps=fps,
            pad_to=pad_to,
            label_video=label_video,
            aggregations=aggs if label_video else None
        )

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
                resp = prompt_client.generate_content(
                    full_prompt,
                    file_descriptor=file_descriptor,
                    response_mime_type="application/json"
                )
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
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True,
                   help="Path to session folder (contains screenshots/ and aggregations.jsonl)")
    p.add_argument("--agg-jsonl", default="aggregations.jsonl",
                   help="Filename of aggregations jsonl inside session folder")
    p.add_argument("--chunk-duration", type=int, default=60,
                   help="Chunk duration in seconds")
    p.add_argument("--fps", type=int, default=1,
                   help="Frames per second for master video")
    p.add_argument("--prompt-client", choices=["gemini", "qwen3vl", "local"],
                   default="gemini")
    p.add_argument("--qwen-model-path", default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                   help="Path to Qwen3-VL model")
    p.add_argument("--label-video", action="store_true",
                   help="Annotate video frames with mouse movements and clicks")
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
    client = None

    if args.prompt_client == 'gemini':
        client = GeminiPromptClient(api_key=api_key)
    elif args.prompt_client == 'qwen3vl':
        client = Qwen3VLPromptClient(model_path=args.qwen_model_path)

    process_session(
        session,
        agg_path,
        out,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        prompt_client=client,
        label_video=args.label_video
    )


if __name__ == '__main__':
    main()
