import json
import os
from pathlib import Path
from datetime import datetime

from prompt.prompt import process_video_chunks
from prompt.annotate import label_video_with_captions

API_KEY = os.getenv("GEMINI_API_KEY")
PERCENTILE = 85
VIDEO_LENGTH = 60


def prompt_one_chunk(api_key, percentile, video_length, chunks_dir):
    SESSION_FOLDER = Path(__file__).parent / "data"
    VIDEO_PATH = SESSION_FOLDER / f"event_logs_video_{percentile}.mp4"
    AGG_JSON = SESSION_FOLDER / f'aggregated_logs_{percentile}.json'

    results = process_video_chunks(
        API_KEY,
        VIDEO_PATH,
        AGG_JSON,
        video_length=VIDEO_LENGTH,
        session_folder=SESSION_FOLDER,
        percentile=PERCENTILE,
        chunks_dir=chunks_dir,
        num_workers=6
    )

    if results:
        print(f"\nProcessed {len(results)} chunks successfully")

        summary_path = chunks_dir / "all_chunks_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Summary saved to: {summary_path}")
    else:
        print("No chunks processed successfully")
    return results


def time_str_to_seconds(t: str) -> int:
    """Convert a time string MM:SS to total seconds."""
    return int(datetime.strptime(t, "%M:%S").minute) * 60 + int(datetime.strptime(t, "%M:%S").second)


def print_tricky_timestamps(results):
    res = []
    for r in results:
        res.extend(r["result"])
    results = res
    tricky_timestamps = [
        {
            "start": "00:24",
            "end": "00:31",
            "comment": "Three different images are shown: 1. Exterior, 2. Exterior, 3. Interior; User clicked on (first) three images ==> Three separate logs"
        },
        {
            "start": "02:14",
            "end": "02:52",
            "comment": "How is the prompt summarized?"
        },
        {
            "start": "03:26",
            "end": "03:39",
            "comment": "Tried to connect but failed. Tried different address and worked (Does it figure out that the right address has the 'lx01' appended?)"
        },
        {
            "start": "04:35",
            "end": "05:13",
            "comment": "User is in spreadsheet, does the model tell what the context is, what a cell contains?"
        }
    ]

    def caption_overlaps_range(caption, start_sec, end_sec):
        cap_start = time_str_to_seconds(caption["start"])
        cap_end = time_str_to_seconds(caption["end"])
        return cap_start < end_sec and cap_end > start_sec

    for ts in tricky_timestamps:
        start_sec = time_str_to_seconds(ts["start"])
        end_sec = time_str_to_seconds(ts["end"])
        print("\n" + "=" * 80)
        print(f"Comment: {ts['comment']}")
        print(f"Time Range: {ts['start']} - {ts['end']}\n")
        print("=" * 80)

        matching_captions = [cap for cap in results if caption_overlaps_range(cap, start_sec, end_sec)]
        if not matching_captions:
            print("⚠️ No matching captions found.")
        else:
            for cap in matching_captions:
                print(f"[{cap['start']} - {cap['end']}] {cap['caption']}")


def main():
    sessions = (Path(__file__).parent / "data").glob("chunks_*")
    sessions = [s for s in sessions if f"chunks_{PERCENTILE}_{VIDEO_LENGTH}" in s.name]
    last_session = max(sessions, key=lambda x: x.name, default=None)
    last_version = int(str(last_session).split("_")[-1]) if last_session else 0
    next_version = last_version + 1
    chunks_dir = Path(__file__).parent / "data" / f"chunks_{PERCENTILE}_{VIDEO_LENGTH}_{next_version}"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing session: {chunks_dir.name}")
    results = prompt_one_chunk(API_KEY, PERCENTILE, VIDEO_LENGTH, chunks_dir)
    prompt_file = Path(__file__).parent.parent / "prompt" / "prompt.txt"
    if prompt_file.exists():
        with open(prompt_file, "r") as f:
            prompt_content = f.read()
        with open(chunks_dir / "prompt.txt", "w") as f:
            f.write(prompt_content)
    print(f"Results saved in: {chunks_dir}")
    print_tricky_timestamps(results)
    video_path = Path(__file__).parent / "data" / f"event_logs_video_{PERCENTILE}.mp4"
    output_video_path = chunks_dir / f"annotated_video_{PERCENTILE}_{VIDEO_LENGTH}.mp4"
    label_video_with_captions(video_path, chunks_dir, output_video_path)


if __name__ == "__main__":
    main()
