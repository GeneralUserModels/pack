import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompt.prompt import process_video_chunks


def prompt_one_chunk(api_key, session_name, percentile, video_length):
    VIDEO_PATH = Path(__file__).parent.parent / "logs" / session_name / f"event_logs_video_{percentile}.mp4"
    AGG_JSON = Path(__file__).parent.parent / "logs" / session_name / f'aggregated_logs_{percentile}.json'
    SESSION_FOLDER = Path(__file__).parent.parent / "logs" / session_name

    results = process_video_chunks(
        api_key,
        VIDEO_PATH,
        AGG_JSON,
        video_length=video_length,
        session_folder=SESSION_FOLDER,
        percentile=percentile
    )

    if results:
        print(f"\nProcessed {len(results)} chunks successfully")

        summary_path = SESSION_FOLDER / f"chunks_{percentile}_{video_length}" / "all_chunks_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Summary saved to: {summary_path}")
    else:
        print("No chunks processed successfully")


if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    PERCENTILE = 85
    VIDEO_LENGTH = 60
    sessions_path = Path(__file__).parent.parent / "logs"

    sessions = [
        session.name for session in sessions_path.iterdir()
        if session.is_dir()
        and session.name.startswith("session_")
        and f"aggregated_logs_{PERCENTILE}.json" in os.listdir(session)
        and f"event_logs_video_{PERCENTILE}.mp4" in os.listdir(session)
    ]

    print(f"Found {len(sessions)} valid sessions")

    # Run in 8 threads max
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(prompt_one_chunk, API_KEY, session, PERCENTILE, VIDEO_LENGTH): session
            for session in sessions
        }

        for future in as_completed(futures):
            session = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error in session {session}: {e}")
