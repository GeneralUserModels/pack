from pathlib import Path
from aggregation_analysis.format_logs import create_regression_dataset
from aggregation_analysis.aggregate_logs import aggregate_and_plot
from aggregation_analysis.to_video import main as logs_to_video


def main():
    sessions_dir = Path(__file__).parent.parent / "logs"
    last_session = max((d for d in sessions_dir.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, default=None)
    _ = create_regression_dataset(
        time_window=5.0,
        output_file='./aggregation_analysis/regression_dataset.json',
        base_dir=last_session
    )

    aggregate_and_plot()
    logs_to_video(last_session, False)


if __name__ == "__main__":
    main()
