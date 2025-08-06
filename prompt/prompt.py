import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from modules import AggregatedLog
import subprocess
import math
import multiprocessing as mp
from functools import partial


TASK_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "string"},
            "end": {"type": "string"},
            "caption": {"type": "string"}
        },
        "required": ["start", "end", "caption"]
    }
}

load_dotenv()


def get_available_sessions(logs_dir="logs"):
    """Get list of available sessions from the logs directory."""
    logs_path = Path(__file__).parent.parent / logs_dir
    if not logs_path.exists():
        print(f"Logs directory not found: {logs_path}")
        return []
    
    sessions = []
    for item in logs_path.iterdir():
        if item.is_dir() and item.name.startswith("session_"):
            sessions.append(item.name)
    
    return sorted(sessions)


def display_session_options(sessions):
    """Display available sessions as numbered options."""
    if not sessions:
        print("No sessions found in logs directory.")
        return None
    
    print("\nAvailable sessions:")
    for i, session in enumerate(sessions, 1):
        print(f"{i}. {session}")
    
    while True:
        try:
            choice = input(f"\nSelect a session (1-{len(sessions)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(sessions):
                return sessions[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(sessions)}")
        except ValueError:
            print("Please enter a valid number")


def get_available_percentiles(session_folder):
    """Get list of available percentiles from video files in the session directory."""
    session_path = Path(session_folder)
    if not session_path.exists():
        print(f"Session directory not found: {session_path}")
        return []
    
    percentiles = []
    for item in session_path.iterdir():
        if item.is_file() and item.name.startswith("event_logs_video_") and item.name.endswith(".mp4"):
            # Extract percentile from filename like "event_logs_video_87.mp4"
            try:
                percentile = int(item.name.replace("event_logs_video_", "").replace(".mp4", ""))
                percentiles.append(percentile)
            except ValueError:
                continue
    
    return sorted(percentiles)


def display_percentile_options(percentiles):
    """Display available percentiles as numbered options."""
    if not percentiles:
        print("No video files found in session directory.")
        return None
    
    print("\nAvailable percentiles:")
    for i, percentile in enumerate(percentiles, 1):
        print(f"{i}. {percentile}%")
    
    while True:
        try:
            choice = input(f"\nSelect a percentile (1-{len(percentiles)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(percentiles):
                return percentiles[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(percentiles)}")
        except ValueError:
            print("Please enter a valid number")


def setup_gemini_api(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-pro')


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


def mmss_to_seconds(mmss):
    try:
        parts = mmss.split(':')
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
    except (ValueError, AttributeError):
        pass
    return 0


def adjust_timestamps_in_tasks(tasks, chunk_start_seconds):
    adjusted_tasks = []

    for task in tasks:
        adjusted_task = task.copy()

        if 'start' in task and task['start']:
            start_seconds = mmss_to_seconds(task['start'])
            adjusted_start_seconds = start_seconds + chunk_start_seconds
            adjusted_task['start'] = seconds_to_mmss(adjusted_start_seconds)

        if 'end' in task and task['end']:
            end_seconds = mmss_to_seconds(task['end'])
            adjusted_end_seconds = end_seconds + chunk_start_seconds
            adjusted_task['end'] = seconds_to_mmss(adjusted_end_seconds)

        adjusted_tasks.append(adjusted_task)

    return adjusted_tasks


def prompt_gemini_with_annotated_video(api_key, video_path, agg_logs, chunk_start_seconds=0):
    try:
        logs_prompt = [a.to_prompt(2 * i) for i, a in enumerate(agg_logs)]

        with open(Path(__file__).parent / "prompt.txt", "r") as f:
            prompt_template = f.read()

        prompt_template = prompt_template.replace("{{LOGS}}", ''.join(logs_prompt))
        print("Generating content with Gemini...")
        model = setup_gemini_api(api_key)
        video_file = upload_video_file(str(video_path))
        print(f"Video file: {video_file.uri}")
        print(f"Prompt template:\n{agg_logs[0].start_timestamp} - {agg_logs[-1].end_timestamp}")

        response = model.generate_content(
            [prompt_template, video_file],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=TASK_SCHEMA,
                temperature=0.0,
            )
        )
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            print("Token Usage:")
            print(f"  Input tokens: {usage.prompt_token_count}")
            print(f"  Output tokens: {usage.candidates_token_count}")
            print(f"  Total tokens: {usage.total_token_count}")
        else:
            print("Token usage information not available")
        return response
    except Exception as e:
        print(f"Error in Gemini call: {e}")
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            print(f"Prompt Feedback: {response.prompt_feedback}")
        return None


def process_single_chunk(chunk_data, api_key, chunks_dir, video_length):
    """Process a single video chunk - this function will be called by worker processes"""
    i, video_chunk_path, log_chunk = chunk_data

    # Check if chunk already processed
    chunk_result_path = chunks_dir / f"chunk_{i:03d}_result.json"
    if chunk_result_path.exists():
        print(f"Chunk {i + 1} already processed, skipping...")
        return None

    print(f"Processing chunk {i + 1}: {video_chunk_path}")
    chunk_start_seconds = i * video_length

    try:
        response = prompt_gemini_with_annotated_video(
            api_key,
            video_chunk_path,
            log_chunk,
            chunk_start_seconds
        )

        if response:
            chunk_info = {
                "chunk_index": i,
                "start_time": seconds_to_mmss(chunk_start_seconds),
                "video_chunk": str(video_chunk_path),
            }

            try:
                result = json.loads(response.text)
                result_type = "json"
            except json.JSONDecodeError:
                result = response.text
                result_type = "text"

            try:
                adjusted_result = adjust_timestamps_in_tasks(result, chunk_start_seconds)
            except Exception as e:
                print(f"Error adjusting timestamps for chunk {i + 1}: {e}")
                adjusted_result = result

            chunk_info["result"] = adjusted_result
            chunk_info["result_type"] = result_type

            with open(chunk_result_path, "w") as f:
                json.dump(chunk_info, f, indent=2, ensure_ascii=False)

            print(f"Saved chunk result to: {chunk_result_path}")
            return chunk_info
        else:
            print(f"Failed to get response for chunk {i + 1}")
            return None

    except Exception as e:
        print(f"Error processing chunk {i + 1}: {e}")
        return None


def process_video_chunks(
    api_key,
    video_path,
    agg_json_path,
    video_length=180,
    session_folder=None,
    percentile=90,
    start_chunk=0,
    end_chunk=None,
    chunks_dir=None,
    num_workers=1
):
    if session_folder is None:
        session_folder = video_path.parent
    if chunks_dir is None:
        chunks_dir = Path(session_folder) / f"chunks_{percentile}_{video_length}"
    chunks_dir.mkdir(parents=True, exist_ok=True)

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

        print(f"Splitting video into {video_length}-second chunks...")
        video_chunks = split_video(video_path, chunks_dir, video_length)
        print("Splitting logs into chunks...")
        log_chunks = split_logs(agg_logs, video_length)

        if end_chunk is None:
            end_chunk = len(video_chunks) - 1

        chunk_data = []
        for i, (video_chunk_path, log_chunk) in enumerate(zip(video_chunks, log_chunks)):
            if start_chunk <= i <= end_chunk:
                chunk_data.append((i, video_chunk_path, log_chunk))

        if not chunk_data:
            print("No chunks to process in the specified range.")
            return []

        print(f"Processing {len(chunk_data)} chunks with {num_workers} worker(s)...")

        results = []

        if num_workers == 1:
            for data in chunk_data:
                result = process_single_chunk(data, api_key, chunks_dir, video_length)
                if result:
                    results.append(result)
        else:
            if num_workers is None:
                num_workers = mp.cpu_count()

            process_func = partial(process_single_chunk,
                                   api_key=api_key,
                                   chunks_dir=chunks_dir,
                                   video_length=video_length)

            with mp.Pool(processes=num_workers) as pool:
                chunk_results = pool.map(process_func, chunk_data)

                results = [result for result in chunk_results if result is not None]

        print(f"Successfully processed {len(results)} chunks")
        return results

    except Exception as e:
        print(f"Error processing chunks: {e}")
        return []


if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    sessions = get_available_sessions()
    selected_session = display_session_options(sessions)

    if not selected_session:
        print("No session selected. Exiting.")
        exit()
    
    SESSION_FOLDER = Path(__file__).parent.parent / "logs" / selected_session
    video_files = get_available_percentiles(SESSION_FOLDER)
    selected_percentile = display_percentile_options(video_files)

    if not selected_percentile:
        print("No percentile selected. Exiting.")
        exit()
    
    VIDEO_PATH = SESSION_FOLDER / f"event_logs_video_{selected_percentile}.mp4"
    AGG_JSON = SESSION_FOLDER / f'aggregated_logs_{selected_percentile}.json'

    VIDEO_LENGTH = 60

    results = process_video_chunks(
        API_KEY,
        VIDEO_PATH,
        AGG_JSON,
        video_length=VIDEO_LENGTH,
        session_folder=SESSION_FOLDER,
        percentile=selected_percentile
    )

    if not results:
        print("No chunks processed successfully")
        exit()

    print(f"\nProcessed {len(results)} chunks successfully")
    summary_path = SESSION_FOLDER / f"chunks_{selected_percentile}_{VIDEO_LENGTH}" / "all_chunks_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_path}")
