import json
import subprocess
import tempfile
from pathlib import Path
import os


def mmss_to_seconds(mmss_str):
    """Convert MM:SS format to seconds"""
    try:
        minutes, seconds = map(int, mmss_str.split(':'))
        return minutes * 60 + seconds
    except Exception as e:
        print(f"Warning: Could not parse time format: {mmss_str}: {e}")
        return 0


def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes:02d}:{seconds:02d}"


def create_subtitle_file(tasks, output_path):
    """Create an SRT subtitle file from tasks"""
    with open(output_path, 'w', encoding='utf-8') as f:
        subtitle_index = 1
        for task in tasks:
            start_seconds = mmss_to_seconds(task['start'])
            end_seconds = mmss_to_seconds(task['end'])
            caption = task['caption']

            subtitle_text = f"{task['start']} - {task['end']}: {caption}"

            f.write(f"{subtitle_index}\n")
            f.write(f"{format_srt_time(start_seconds)} --> {format_srt_time(end_seconds)}\n")
            f.write(f"{subtitle_text}\n\n")
            subtitle_index += 1


def format_srt_time(seconds):
    """Format seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def create_annotated_video(video_path, all_tasks, output_video_path):
    if not all_tasks:
        print("No valid tasks found in summary file")
        return False

    print(f"Found {len(all_tasks)} tasks to annotate")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as temp_srt:
        temp_srt_path = temp_srt.name

    try:
        create_subtitle_file(all_tasks, temp_srt_path)
        print(f"Created subtitle file: {temp_srt_path}")

        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f"subtitles='{temp_srt_path}':force_style='FontName=Arial,FontSize=8,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1'",
            '-c:a', 'copy',
            '-y',
            str(output_video_path)
        ]

        print("Running FFmpeg command...")
        print(" ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully created annotated video: {output_video_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False

    finally:
        try:
            os.unlink(temp_srt_path)
            print(f"Cleaned up temporary file: {temp_srt_path}")
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")


def label_video_with_captions(video_path, summary_dir, output_video_path):

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    jsons = list(summary_dir.glob("*_result.json"))
    if not jsons:
        print(f"No JSON files found in directory: {summary_dir}")
        return

    annotations = []
    for json_file in jsons:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)["result"]
                annotations = annotations + data
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file {json_file}: {e}")
            continue

    success = create_annotated_video(video_path, annotations, output_video_path)

    if success:
        print("\n✅ Annotation complete!")
        print("📹 Annotated video saved to: {output_video_path}")
    else:
        print("\n❌ Annotation failed!")


if __name__ == "__main__":
    PERCENTILE = 87
    SESSION = "session_2025-07-17_10-06-32"
    VIDEO_LENGTH = 60
    base_path = Path(__file__).parent.parent / "logs" / SESSION
    video_path = base_path / f"event_logs_video_{PERCENTILE}.mp4"
    summary_dir = base_path / f"chunks_{PERCENTILE}_{VIDEO_LENGTH}"
    output_video_path = base_path / f"annotated_video_{PERCENTILE}_{VIDEO_LENGTH}.mp4"
    label_video_with_captions(video_path, summary_dir, output_video_path)
