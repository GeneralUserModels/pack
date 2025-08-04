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
    PERCENTILE = 87
    # SESSION = "session_2025-07-17_10-06-32"
    SESSION = "session_2025-07-11_02-51-30-768112"
    VIDEO_PATH = Path(__file__).parent.parent / "logs" / SESSION / f"event_logs_video_{PERCENTILE}.mp4"
    AGG_JSON = Path(__file__).parent.parent / "logs" / SESSION / f'aggregated_logs_{PERCENTILE}.json'
    SESSION_FOLDER = Path(__file__).parent.parent / "logs" / SESSION

    VIDEO_LENGTH = 60

    results = process_video_chunks(
        API_KEY,
        VIDEO_PATH,
        AGG_JSON,
        video_length=VIDEO_LENGTH,
        session_folder=SESSION_FOLDER,
        percentile=PERCENTILE
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
