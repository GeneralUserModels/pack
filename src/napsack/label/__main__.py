from pathlib import Path
import argparse
from dotenv import load_dotenv

from napsack.label.discovery import discover_sessions, discover_screenshots_sessions, create_single_config
from napsack.label.clients import create_client
from napsack.label.processor import Processor
from napsack.label.visualizer import Visualizer

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Process session recordings with VLM")

    session_group = p.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--session-dir", type=Path)
    session_group.add_argument("--sessions-root", type=Path)

    p.add_argument("--chunk-duration", type=int, default=60, help="Chunk duration in seconds")
    p.add_argument("--fps", type=int, default=1, help="Frames per second for video processing")

    p.add_argument("--screenshots-only", action="store_true", help="Process screenshots folder only without aggregations or annotations")
    p.add_argument("--image-extensions", nargs="+", default=[".jpg", ".jpeg", ".png"], help="Image file extensions to consider")
    p.add_argument("--max-time-gap", type=float, default=300.0, help="Maximum time gap (seconds) between images before forcing a video split (default: 120 = 2 minutes)")
    p.add_argument("--prompt-file", default=None, help="Path to prompt file (default: prompts/default.txt or prompts/screenshots_only.txt if screenshots only)")
    p.add_argument("--hash-cache", type=str, default=None, help="Path to hash_cache.json for deduplicating consecutive similar images")
    p.add_argument("--dedupe-threshold", type=int, default=1, help="Hamming distance threshold for deduplication (drop if <= threshold, default: 1)")
    p.add_argument("--annotate", action="store_true", help="Annotate videos with cursor positions and clicks (only for standard processing)")
    p.add_argument("--image-mode", action="store_true", help="Send frames as individual images instead of video (for models that don't support video input)")
    p.add_argument("--dense-caption", action="store_true", help="Include a dense text caption per chunk describing important text the user focused on, for retrieval")
    p.add_argument("--skip-existing", action="store_true", help="Skip sessions that have already been processed")
    p.add_argument("--visualize", action="store_true", help="Create annotated video visualizations after processing")
    p.add_argument("--encode-only", action="store_true", help="Only encode videos (create chunks), skip labeling. Useful for pre-processing before running the full pipeline.")

    p.add_argument("--client", choices=["litellm", "bigquery", "tinfoil"], default="litellm")
    p.add_argument("--model", default="gemini/gemini-2.5-flash",
                   help="Full litellm model string, e.g. gemini/gemini-2.5-flash, openai/gpt-4o, "
                        "anthropic/claude-3-5-sonnet-20241022, hosted_vllm/Qwen3-VL-8B, "
                        "or a Tinfoil model name e.g. meta-llama/Llama-3.2-11B-Vision-Instruct")
    p.add_argument("--api-base", default=None,
                   help="API base URL for local/custom endpoints (e.g. http://localhost:8000/v1 for vLLM)")

    tinfoil_group = p.add_argument_group("Tinfoil Options")
    tinfoil_group.add_argument("--tinfoil-enclave", default="", help="Tinfoil enclave address (auto-selected if omitted)")
    tinfoil_group.add_argument("--tinfoil-repo", default="tinfoilsh/confidential-model-router",
                               help="GitHub repo used for Tinfoil attestation")
    p.add_argument("--encode-workers", type=int, default=8, help="Number of parallel workers for video encoding")
    p.add_argument("--label-workers", type=int, default=4, help="Number of parallel workers for VLM labeling")

    bq_group = p.add_argument_group("BigQuery Options")
    bq_group.add_argument("--bq-project", help="GCP project ID for AI Platform endpoint")
    bq_group.add_argument("--bq-bucket-name", help="GCS bucket name for uploading videos")
    bq_group.add_argument("--bq-gcs-prefix", default="video_chunks", help="Prefix/folder path in GCS bucket")
    bq_group.add_argument("--bq-object-table-location", default="us", help="Object table location (e.g., 'us' or 'us.screenomics-gemini')")

    args = p.parse_args()

    if not args.model:
        if args.client == 'bigquery':
            args.model = 'dataset.model'  # Placeholder - user must provide full model reference
    
    # gemini, hosted_vllm, and tinfoil support video input; everything else falls back to image mode
    provider = args.model.split("/")[0] if args.model and "/" in args.model else ""
    if not args.image_mode and args.client not in ("tinfoil",) and provider not in ("gemini", "hosted_vllm"):
        args.image_mode = True

    if not args.prompt_file:
        if args.image_mode:
            args.prompt_file = "prompts/image_mode.txt"
        elif args.screenshots_only:
            args.prompt_file = "prompts/screenshots_only.txt"
        else:
            args.prompt_file = "prompts/default.txt"

    return args


def setup_configs(args):
    if args.session_dir:
        configs = [create_single_config(
            args.session_dir,
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


def process_with_litellm(args, configs):
    client = create_client(
        'litellm',
        model_name=args.model,
        api_base=args.api_base,
    )

    processor = Processor(
        client=client,
        encode_workers=args.encode_workers,
        label_workers=args.label_workers,
        screenshots_only=args.screenshots_only,
        prompt_file=args.prompt_file,
        max_time_gap=args.max_time_gap,
        hash_cache_path=args.hash_cache,
        dedupe_threshold=args.dedupe_threshold,
        image_mode=args.image_mode,
        dense_caption=args.dense_caption,
    )

    return processor.process_sessions(
        configs,
        fps=args.fps,
        annotate=args.annotate and not args.screenshots_only,
        encode_only=args.encode_only,
    )


def process_with_bigquery(args, configs):
    client = create_client(
        'bigquery',
        model_name=args.model,
        bucket_name=args.bq_bucket_name,
        gcs_prefix=args.bq_gcs_prefix,
        object_table_location=args.bq_object_table_location,
        project_id=args.bq_project,
    )

    processor = Processor(
        client=client,
        encode_workers=args.encode_workers,
        label_workers=args.label_workers,
        screenshots_only=args.screenshots_only,
        prompt_file=args.prompt_file,
        max_time_gap=args.max_time_gap,
        hash_cache_path=args.hash_cache,
        dedupe_threshold=args.dedupe_threshold,
        image_mode=args.image_mode,
        dense_caption=args.dense_caption,
    )

    return processor.process_sessions(
        configs,
        fps=args.fps,
        annotate=args.annotate and not args.screenshots_only,
        encode_only=args.encode_only,
    )


def process_with_tinfoil(args, configs):
    client = create_client(
        'tinfoil',
        model_name=args.model,
        enclave=args.tinfoil_enclave,
        repo=args.tinfoil_repo,
    )

    processor = Processor(
        client=client,
        encode_workers=args.encode_workers,
        label_workers=args.label_workers,
        screenshots_only=args.screenshots_only,
        prompt_file=args.prompt_file,
        max_time_gap=args.max_time_gap,
        hash_cache_path=args.hash_cache,
        dedupe_threshold=args.dedupe_threshold,
        image_mode=args.image_mode,
        dense_caption=args.dense_caption,
    )

    return processor.process_sessions(
        configs,
        fps=args.fps,
        annotate=args.annotate and not args.screenshots_only,
        encode_only=args.encode_only,
    )


def main():
    args = parse_args()

    configs = setup_configs(args)
    if not configs:
        return

    print(f"Processing {len(configs)} sessions")

    if args.client == 'litellm':
        results = process_with_litellm(args, configs)
    elif args.client == 'bigquery':
        results = process_with_bigquery(args, configs)
    elif args.client == 'tinfoil':
        results = process_with_tinfoil(args, configs)
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
