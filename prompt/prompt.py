import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from modules import AggregatedLog
import subprocess
import math

load_dotenv()


def setup_gemini_api(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')


def upload_video_file(video_path):
    print(f"Uploading file: {video_path}")
    video_file = genai.upload_file(path=video_path)
    print(f"Completed upload: {video_file.uri}")
    while video_file.state.name == "PROCESSING":
        print("Processing video...")
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    print("Video processing completed!")
    return video_file


def get_video_duration(video_path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(video_path)
        ], capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return None


def split_video(video_path, output_dir, chunk_duration_seconds):
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(video_path)
    if duration is None:
        raise ValueError("Could not determine video duration")

    num_chunks = math.ceil(duration / chunk_duration_seconds)
    chunk_paths = []

    for i in range(num_chunks):
        start_time = i * chunk_duration_seconds
        output_path = output_dir / f"chunk_{i:03d}.mp4"

        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(chunk_duration_seconds),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            str(output_path),
            '-y'
        ]

        print(f"Creating chunk {i + 1}/{num_chunks}: {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error creating chunk {i}: {result.stderr}")
            continue

        chunk_paths.append(output_path)

    return chunk_paths


def split_logs(logs, chunk_duration_seconds):
    chunks = []
    current_chunk = []

    for log in logs:
        current_chunk.append(log)

        if len(current_chunk) >= chunk_duration_seconds // 2:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def seconds_to_mmss(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def parse_jsonl_response(response_text):
    try:
        cleaned_text = response_text.strip()

        if cleaned_text.startswith('```jsonl'):
            cleaned_text = cleaned_text.replace('```jsonl', '', 1).strip()
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text.replace('```', '', 1).strip()

        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3].strip()

        cleaned_text = cleaned_text.strip()

        parsed_objects = []
        lines = cleaned_text.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                parsed_obj = json.loads(line)
                parsed_objects.append(parsed_obj)
            except json.JSONDecodeError as e:
                print(f"Failed to parse line {line_num}: {line}")
                print(f"JSON error: {e}")
                return {
                    'success': False,
                    'parsed_data': None,
                    'raw_text': response_text
                }

        return {
            'success': True,
            'parsed_data': parsed_objects,
            'raw_text': response_text
        }

    except Exception as e:
        print(f"Error parsing JSONL response: {e}")
        return {
            'success': False,
            'parsed_data': None,
            'raw_text': response_text
        }


def prompt_gemini_with_annotated_video(api_key, video_path, agg_logs, chunk_start_seconds=0):
    try:
        logs_prompt = [a.to_prompt(2 * i + chunk_start_seconds * 2) for i, a in enumerate(agg_logs)]

        prompt_template = f"""
I got this annotated video composed of a series of annotated screenshots, one each second. Thereby the screenshots have annotations baked into them:
Mouse Interactions
Click Markers (Circular indicators):
Red circles: Left mouse button clicks
Blue circles: Right mouse button clicks
Green circles: Middle mouse button clicks
Yellow circles: Other/unknown mouse button clicks
Each click marker appears as a filled circle with a black outline at the exact location where the mouse click occurred.
Cursor Movement Annotations
Movement Arrows:
Orange arrows: Standard cursor movements during regular interactions
Magenta arrows: Cursor movements between different interaction sequences or logs
Lime green dots: Starting positions of cursor movements
Dark green outline: Border around start position markers
Arrow Components:
Line: Shows the path of cursor movement
Arrowhead: Points to the final position of the cursor
Start marker: Small lime green circle indicating where the movement began
Additionally I got a log file, logging all user interactions with the system. They are in MM:SS format corresponding to the video timestamps and list all actions, which are user inputs, in an aggregated form. Keys are delimited by '|' pipe symbol:
{''.join(logs_prompt)}
Given this information, please generate a list of higher level user interactions, so called tasks. Each task describes what the user is doing.
Therefore use this JSONL format:
<explanation>
    {{
        "start": "MM:SS",
        "end": "MM:SS",
        "caption": "describe the action the user did, some context, the location and KEEP NAMES! DONT provide information without context, like 'clicked at (200, 300)', instead describe what the user did (e.g. user clicked on 'start search' button). Similarely don't provide raw information for keyboard inputs, so instead of e.g. 'User hold shift and ctrl then pressed left arrow', state what the user did, like 'user selected 7 cells in statistics sheet in analysis excle file'",
        "confidence": "Rate between 1 and 10, where 1 means you are very unsure and 10 means you are incredibly sure about the caption.",
    }}
</explanation>
<examples>
    {{
        "start": "01:12",
        "end": "01:17",
        "caption": "typed 'google.com' into the search bar in google chrome",
        "confidence": "7",

    }},
    {{
        "start": "03:15",
        "end": "03:21",
        "caption": "Switch to google chrome window displaying LoRA paper",
        "confidence": "9",

    }},
    {{
        "start": "05:12",
        "end": "05:22",
        "caption": "Switched to 'analysis.ipynb' notebook in VSCode",
        "confidence": "6",

    }},

</examples>
        """

        print("Generating content with Gemini...")
        model = setup_gemini_api(api_key)
        video_file = upload_video_file(str(video_path))
        print(f"Video file: {video_file.uri}")
        print(f"Prompt template:\n{agg_logs[0].start_timestamp} - {agg_logs[-1].end_timestamp}")
        response = model.generate_content([prompt_template, video_file])
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None


def process_video_chunks(api_key, video_path, agg_json_path, video_length=180, session_folder=None, percentile=90):
    try:
        with open(agg_json_path, "r") as f:
            logs_data = json.load(f)
        agg_logs = [AggregatedLog.from_dict(log) for log in logs_data]

        if video_length is None:
            print("Processing entire video...")
            result = prompt_gemini_with_annotated_video(api_key, video_path, agg_logs)
            if result:
                print("Gemini's Response:")
                print(result)
            return [result] if result else []

        if session_folder is None:
            session_folder = video_path.parent
        chunks_dir = Path(session_folder) / f"chunks_{PERCENTILE}_{video_length}"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        print(f"Splitting video into {video_length}-second chunks...")
        video_chunks = split_video(video_path, chunks_dir, video_length)

        print("Splitting logs into chunks...")
        log_chunks = split_logs(agg_logs, video_length)

        results = []
        for i, (video_chunk_path, log_chunk) in enumerate(zip(video_chunks, log_chunks)):
            print(f"\nProcessing chunk {i + 1}/{len(video_chunks)}: {video_chunk_path}")

            chunk_start_seconds = i * video_length
            raw_result = prompt_gemini_with_annotated_video(
                api_key,
                video_chunk_path,
                log_chunk,
                chunk_start_seconds
            )

            if raw_result:
                parsed_result = parse_jsonl_response(raw_result)

                chunk_result_path = chunks_dir / f"chunk_{i:03d}_result.json"
                chunk_info = {
                    "chunk_index": i,
                    "start_time": seconds_to_mmss(chunk_start_seconds),
                    "end_time": seconds_to_mmss(chunk_start_seconds + len(log_chunk)),
                    "video_chunk": str(video_chunk_path),
                }

                if parsed_result['success']:
                    chunk_info["result"] = parsed_result['parsed_data']
                    chunk_info["result_type"] = "jsonl"
                    print(f"Successfully parsed {len(parsed_result['parsed_data'])} JSONL objects")
                else:
                    chunk_info["result"] = parsed_result['raw_text']
                    chunk_info["result_type"] = "text"
                    print("Failed to parse JSONL, storing as raw text")

                with open(chunk_result_path, "w") as f:
                    json.dump(chunk_info, f, indent=2)

                print(f"Saved chunk result to: {chunk_result_path}")
                print(f"Chunk {i + 1} response preview:")
                if parsed_result['success']:
                    print(f"Parsed {len(parsed_result['parsed_data'])} tasks")
                    for task in parsed_result['parsed_data'][:2]:
                        print(f"  {task.get('start', '??')} - {task.get('end', '??')}: {task.get('caption', 'No caption')[:100]}")
                else:
                    print(f"Raw text (first 200 chars): {raw_result[:200]}...")

                results.append(chunk_info)
            else:
                print(f"Failed to get response for chunk {i + 1}")

        return results

    except Exception as e:
        print(f"Error processing chunks: {e}")
        return []


if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    PERCENTILE = 90
    SESSION = "session_2025-07-17_10-06-32"
    VIDEO_PATH = Path(__file__).parent.parent / "logs" / SESSION / f"event_logs_video_{PERCENTILE}.mp4"
    AGG_JSON = Path(__file__).parent.parent / "logs" / SESSION / f'aggregated_logs_{PERCENTILE}.json'
    SESSION_FOLDER = Path(__file__).parent.parent / "logs" / SESSION

    VIDEO_LENGTH = 180

    results = process_video_chunks(
        API_KEY,
        VIDEO_PATH,
        AGG_JSON,
        video_length=VIDEO_LENGTH,
        session_folder=SESSION_FOLDER
    )

    if results:
        print(f"\nProcessed {len(results)} chunks successfully")

        summary_path = SESSION_FOLDER / f"chunks_{PERCENTILE}_{VIDEO_LENGTH}" / "all_chunks_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Summary saved to: {summary_path}")
    else:
        print("No chunks processed successfully")
