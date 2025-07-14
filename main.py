import json
from pathlib import Path
from aggregate.logs import main as aggregate_logs
from aggregate.video import convert_to_video, render_images
from analyze.plot_raw_logs import plot_interactive


def main():
    SESSION = 'session_2025-07-13_17-09-59-136228'
    PERCENTILE = 95
    session_path = Path(__file__).parent / 'logs' / SESSION

    aggregated_logs = aggregate_logs(session_path, PERCENTILE)
    with open(session_path / f'aggregated_logs_{PERCENTILE}.json', 'w') as f:
        json.dump([log.to_dict() for log in aggregated_logs], f, indent=4, ensure_ascii=False)

    plot_interactive(session_path / 'events.jsonl')

    # convert_to_video(aggregated_logs, "event_logs_video", session_path, should_annotate=True, seconds_per_frame=1)
    visualization_path = session_path / f"agg_{PERCENTILE}_visualizations"

    visualization_path.mkdir(parents=True, exist_ok=True)
    render_images(aggregated_logs, visualization_path)


if __name__ == "__main__":
    main()
