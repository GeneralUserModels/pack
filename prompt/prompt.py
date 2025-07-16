import os
import time
import json
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

from modules import AggregatedLog

load_dotenv()

PERCENTILE = 95


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


def prompt_gemini_with_annotated_video(api_key, video_path, agg_json_path):
    try:
        with open(agg_json_path, "r") as f:
            logs = json.load(f)
        agg_logs = [AggregatedLog.from_dict(log) for log in logs]
        logs_prompt = [a.to_prompt(2 * i) for i, a in enumerate(agg_logs)]

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
Therefor use this JSON format:
[
    {{
        "start": "MM:SS",
        "end": "MM:SS",
        "task": "Description of the task, e.g. user pairs bluetooth earbuds in settings",
        "confidence": "between 0 and 1, how confident you are that this task is correct"
    }}
]
        """

        print("Generating content with Gemini...")
        print(f"Prompt: {prompt_template}")
        model = setup_gemini_api(api_key)
        video_file = upload_video_file(str(video_path))
        print(f"Video file: {video_file.uri}")

        response = model.generate_content([prompt_template, video_file])
        return response.text

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    SESSION = "session_2025-07-14_20-15-27-182853"
    VIDEO_PATH = Path(__file__).parent.parent / "logs" / SESSION / "event_logs_video.mp4"
    AGG_JSON = Path(__file__).parent.parent / "logs" / SESSION / f'aggregated_logs_{PERCENTILE}.json'

    result = prompt_gemini_with_annotated_video(API_KEY, VIDEO_PATH, AGG_JSON)

    if result:
        print("Gemini's Response:")
        print(result)
    else:
        print("Failed to get response from Gemini")
