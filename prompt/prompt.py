import os
import time
import json
import codecs
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


def parse_json_response(response_text):
    try:
        response_text = codecs.decode(response_text, "unicode_escape")

        cleaned_text = response_text.strip()

        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text.replace('```json', '', 1).strip()
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text.replace('```', '', 1).strip()
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3].strip()

        cleaned_text = cleaned_text.strip()

        try:
            parsed_data = json.loads(cleaned_text)

            if not isinstance(parsed_data, list):
                parsed_data = [parsed_data]

            return {
                'success': True,
                'parsed_data': parsed_data,
                'raw_text': response_text
            }
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {cleaned_text[:200]}...")
            print(f"JSON error: {e}")
            return {
                'success': False,
                'parsed_data': None,
                'raw_text': response_text,
                'error': str(e)
            }

    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return {
            'success': False,
            'parsed_data': None,
            'raw_text': response_text,
            'error': str(e)
        }


def prompt_gemini_with_annotated_video(api_key, video_path, agg_logs, chunk_start_seconds=0):
    try:
        logs_prompt = [a.to_prompt(2 * i) for i, a in enumerate(agg_logs)]

        prompt_template = f"""I have an annotated video composed of a series of screenshots, one per second, with the following baked-in annotations:

Mouse Interactions  
- **Click Markers (circular indicators)**:  
  - Red circles: Left mouse button clicks  
  - Blue circles: Right mouse button clicks  
  - Green circles: Middle mouse button clicks  
  - Yellow circles: Other/unknown mouse button clicks  
  - Each marker has a black outline and appears at the exact click location  

Cursor Movement Annotations  
- **Movement Arrows**:  
  - Orange arrows: Regular cursor movements  
  - Magenta arrows: Cursor transitions between different interaction sequences/logs  
  - Lime green dots with dark green outline: Starting positions of cursor movements  
- **Arrow Components**:  
  - Line: Cursor path  
  - Arrowhead: Final cursor position  
  - Start marker: Small lime green circle at movement start  

Additionally, I have a log file listing all user inputs in MM:SS format matching the video timestamps. Actions are aggregated and keys are delimited by a pipe symbol (`|`):  

## Logs

{''.join(logs_prompt)}

## Task

Using this data, generate a list of higher-level user interactions (called **tasks**) that describe what the user is doing.  

**Output format**: Return a JSON array of task objects in this exact structure:

[
    {
        "start": "MM:SS",
        "end": "MM:SS",
        "caption": "Describe the user’s action with context and names.",
    }
]

## Guidelines

- DON'T give raw input like coordinates or raw key presses. 
- Instead, provide a description of the user’s intention (e.g., 'User clicked on start search button' or 'User selected 7 cells in statistics sheet in analysis Excel file').
- The caption should be a single action, not a sequence of actions.
- Be specific. Mention the name of the application, file, website, etc.

**Examples**:
[
    {
        "start": "01:12",
        "end": "01:13",
        "caption": "User double-clicked the 'Google Chrome' shortcut on the desktop"
    },
    {
        "start": "01:15",
        "end": "01:17",
        "caption": "User typed 'google.com' into the address bar and pressed Enter"
    },
    {
        "start": "01:17",
        "end": "01:20",
        "caption": "User typed 'best budget noise cancelling headphones 2024' into the Google search bar"
    },
    {
        "start": "01:20",
        "end": "01:25",
        "caption": "User scrolled through Google search results for 'best budget noise cancelling headphones 2024'"
    },
    {
        "start": "03:17",
        "end": "03:21",
        "caption": "User clicked on the tab titled 'LoRA: Low-Rank Adaptation of Large Language Models - arXiv' in Google Chrome"
    },
    {
        "start": "04:21",
        "end": "04:23",
        "caption": "User scrolled through the article titled 'The Perfect Ice Cream: A Summer Recipe Guide' on the New York Times website"
    },
    {
        "start": "04:23",
        "end": "04:25",
        "caption": "User scrolled to the bottom of the article 'The Perfect Ice Cream: A Summer Recipe Guide'"
    },
    {
        "start": "04:30",
        "end": "04:33",
        "caption": "User clicked on the messaging tab titled 'Team Chat – Product Launch'"
    },
    {
        "start": "04:33",
        "end": "04:36",
        "caption": "User scrolled through messages in 'Team Chat – Product Launch'"
    },
    {
        "start": "05:14",
        "end": "05:17",
        "caption": "User clicked the tab titled 'analysis.ipynb' in Visual Studio Code"
    },
    {
        "start": "05:17",
        "end": "05:22",
        "caption": "User viewed the notebook 'analysis.ipynb' in Visual Studio Code"
    }
]

Return ONLY the JSON array of tasks. No other text or explanation."""

        print("Generating content with Gemini...")
        print(prompt_template + "...")
        model = setup_gemini_api(api_key)
        video_file = upload_video_file(str(video_path))
        print(f"Video file: {video_file.uri}")
        print(f"Prompt template:\n{agg_logs[0].start_timestamp} - {agg_logs[-1].end_timestamp}")
        response = model.generate_content([prompt_template, video_file])
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None


def process_video_chunks(api_key, video_path, agg_json_path, video_length=180, session_folder=None, percentile=90, start_chunk=0, end_chunk=None):
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
        chunks_dir = Path(session_folder) / f"chunks_{percentile}_{video_length}"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        print(f"Splitting video into {video_length}-second chunks...")
        video_chunks = split_video(video_path, chunks_dir, video_length)

        print("Splitting logs into chunks...")
        log_chunks = split_logs(agg_logs, video_length)

        if end_chunk is None:
            end_chunk = len(video_chunks) - 1

        results = []
        for i, (video_chunk_path, log_chunk) in enumerate(zip(video_chunks, log_chunks)):
            if i < start_chunk or i > end_chunk:
                print(f"Skipping chunk {i + 1} (out of range)")
                continue
            print(f"\nProcessing chunk {i + 1}/{len(video_chunks)}: {video_chunk_path}")

            chunk_start_seconds = i * video_length
            raw_result = prompt_gemini_with_annotated_video(
                api_key,
                video_chunk_path,
                log_chunk,
                chunk_start_seconds
            )

            if raw_result:
                parsed_result = parse_json_response(raw_result)

                chunk_result_path = chunks_dir / f"chunk_{i:03d}_result.json"
                chunk_info = {
                    "chunk_index": i,
                    "start_time": seconds_to_mmss(chunk_start_seconds),
                    "end_time": seconds_to_mmss(chunk_start_seconds + len(log_chunk)),
                    "video_chunk": str(video_chunk_path),
                }

                if parsed_result['success']:
                    adjusted_tasks = adjust_timestamps_in_tasks(parsed_result['parsed_data'], chunk_start_seconds)

                    chunk_info["result"] = adjusted_tasks
                    chunk_info["result_type"] = "json"
                    print(f"Successfully parsed {len(adjusted_tasks)} JSON task objects")

                    for task in adjusted_tasks[:2]:
                        print(f"  {task.get('start', '??')} - {task.get('end', '??')}: {task.get('caption', 'No caption')[:100]}")
                else:
                    chunk_info["result"] = parsed_result['raw_text']
                    chunk_info["result_type"] = "text"
                    chunk_info["error"] = parsed_result.get('error', 'Unknown parsing error')
                    print("Failed to parse JSON, storing as raw text")

                with open(chunk_result_path, "w") as f:
                    json.dump(chunk_info, f, indent=2)

                print(f"Saved chunk result to: {chunk_result_path}")
                print(f"Chunk {i + 1} response preview:")
                if parsed_result['success']:
                    print(f"Parsed {len(adjusted_tasks)} tasks with adjusted timestamps")
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
    PERCENTILE = 87
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
