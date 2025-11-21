from pathlib import Path
import argparse
from dotenv import load_dotenv

from label.discovery import discover_sessions, discover_screenshots_sessions, create_single_config
from label.clients import create_client
from label.processor import Processor
from label.visualizer import Visualizer

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Process session recordings with VLM")

    session_group = p.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--session", type=Path)
    session_group.add_argument("--sessions-root", type=Path)

    p.add_argument("--chunk-duration", type=int, default=60, help="Chunk duration in seconds")
    p.add_argument("--fps", type=int, default=1, help="Frames per second for video processing")

    p.add_argument("--screenshots-only", action="store_true", help="Process screenshots folder only without aggregations or annotations")
    p.add_argument("--image-extensions", nargs="+", default=[".jpg", ".jpeg", ".png"], help="Image file extensions to consider")
    p.add_argument("--max-time-gap", type=float, default=300.0, help="Maximum time gap (seconds) between images before forcing a video split (default: 120 = 2 minutes)")
    p.add_argument("--prompt-file", default=None, help="Path to prompt file (default: prompts/default.txt or prompts/screenshots_only.txt if screenshots only)")
    p.add_argument("--annotate", action="store_true", help="Annotate videos with cursor positions and clicks (only for standard processing)")
    p.add_argument("--skip-existing", action="store_true", help="Skip sessions that have already been processed")
    p.add_argument("--visualize", action="store_true", help="Create annotated video visualizations after processing")

    p.add_argument("--client", choices=["gemini", "vllm", "bigquery"], default="gemini")
    p.add_argument("--model", default="")
    p.add_argument("--num-workers", type=int, default=4, help="Number of concurrent workers (set to 1 to disable concurrency)")

    vllm_group = p.add_argument_group("vLLM Options")
    vllm_group.add_argument("--vllm-url")

    bq_group = p.add_argument_group("BigQuery Options")
    bq_group.add_argument("--bq-bucket-name", help="GCS bucket name for uploading videos")
    bq_group.add_argument("--bq-gcs-prefix", default="video_chunks", help="Prefix/folder path in GCS bucket")
    bq_group.add_argument("--bq-object-table-location", default="us", help="Object table location (e.g., 'us' or 'us.screenomics-gemini')")

    args = p.parse_args()

    if not args.model:
        if args.client == 'gemini':
            args.model = 'gemini-2.5-flash'
        elif args.client == 'vllm':
            args.model = 'Qwen/Qwen3-VL-8B-Thinking-FP8'
        elif args.client == 'bigquery':
            args.model = 'dataset.model'  # Placeholder - user must provide full model reference
    if not args.prompt_file:
        args.prompt_file = "prompts/screenshots_only.txt" if args.screenshots_only else "prompts/default.txt"

    return args


def setup_configs(args):
    if args.session:
        configs = [create_single_config(
            args.session,
            args.chunk_duration,
            args.screenshots_only,
            tuple(args.image_extensions),
        )]
    else:
        if args.screenshots_only:
            configs = discover_screenshots_sessions(
                args.sessions_root,
                args.chunk_duration,
                tuple(args.image_extensions),
            )
        else:
            configs = discover_sessions(
                args.sessions_root,
                args.chunk_duration,
                args.skip_existing,
            )

        if not configs:
            print(f"No sessions found in {args.sessions_root}")
            return []

    return configs


def process_with_gemini(args, configs):
    client = create_client(
        'gemini',
        model_name=args.model,
    )

    processor = Processor(
        client=client,
        num_workers=args.num_workers,
        screenshots_only=args.screenshots_only,
        prompt_file=args.prompt_file,
        max_time_gap=args.max_time_gap,
    )

    return processor.process_sessions(
        configs,
        fps=args.fps,
        annotate=args.annotate and not args.screenshots_only,
    )


def process_with_vllm(args, configs):
    client = create_client(
        'vllm',
        base_url=args.vllm_url if args.vllm_url.endswith('/v1') else f"{args.vllm_url}/v1",
        model_name=args.model
    )

    processor = Processor(
        client=client,
        num_workers=args.num_workers,
        screenshots_only=args.screenshots_only,
        prompt_file=args.prompt_file,
        max_time_gap=args.max_time_gap,
    )

    return processor.process_sessions(
        configs,
        fps=args.fps,
        annotate=args.annotate and not args.screenshots_only,
    )


def process_with_bigquery(args, configs):
    client = create_client(
        'bigquery',
        model_name=args.model,
        bucket_name=args.bq_bucket_name,
        gcs_prefix=args.bq_gcs_prefix,
        object_table_location=args.bq_object_table_location,
    )

    processor = Processor(
        client=client,
        num_workers=args.num_workers,
        screenshots_only=args.screenshots_only,
        prompt_file=args.prompt_file,
        max_time_gap=args.max_time_gap,
    )

    return processor.process_sessions(
        configs,
        fps=args.fps,
        annotate=args.annotate and not args.screenshots_only,
    )


def main():
    args = parse_args()

    configs = setup_configs(args)
    if not configs:
        return

    print(f"Processing {len(configs)} sessions")

    if args.client == 'gemini':
        results = process_with_gemini(args, configs)
    elif args.client == 'vllm':
        results = process_with_vllm(args, configs)
    elif args.client == 'bigquery':
        results = process_with_bigquery(args, configs)
    else:
        raise ValueError(f"Unknown client: {args.client}")

    print(f"✓ Processed {len(results)} sessions")

    if args.visualize:
        print("\nCreating visualizations...")
        visualizer = Visualizer(args.annotate)

        for config in configs:
            if not config.matched_captions_jsonl.exists():
                print(f"Skipping Visualizing {config.session_id}: no data.jsonl")
                continue

            try:
                output = config.session_folder / "annotated.mp4"
                visualizer.visualize(config.session_folder, output, args.fps)
                print(f"✓ {config.session_id}: {output}")
            except Exception as e:
                print(f"✗ {config.session_id}: {e}")


if __name__ == '__main__':
    main()
