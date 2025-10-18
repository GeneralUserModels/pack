from __future__ import annotations
from typing import List
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from label.clients import GeminiPromptClient, Qwen3VLPromptClient
from label.vllm_server_manager import VLLMServerManager
from label.session_processor import SessionProcessor, SessionConfig

load_dotenv()


def discover_sessions(
    sessions_root: Path,
    agg_filename: str = "aggregations.jsonl",
    chunk_duration: int = 60
) -> List[SessionConfig]:
    """Discover sessions with screenshots and aggregation logs."""
    configs = []

    if not sessions_root.exists():
        raise RuntimeError(f"Sessions root directory not found: {sessions_root}")

    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue

        screenshots_dir = session_dir / "screenshots"
        agg_path = session_dir / agg_filename

        if screenshots_dir.exists() and agg_path.exists():
            has_images = any(
                p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                for p in screenshots_dir.iterdir()
            )

            if has_images:
                configs.append(SessionConfig(
                    session_folder=session_dir,
                    agg_jsonl=agg_path,
                    out_chunks_dir=session_dir / f"chunks_{chunk_duration}",
                ))
                print(f"[Discovery] Found session: {session_dir.name}")
            else:
                print(f"[Discovery] Skipping {session_dir.name}: no images in screenshots/")
        else:
            missing = []
            if not screenshots_dir.exists():
                missing.append("screenshots/")
            if not agg_path.exists():
                missing.append(agg_filename)
            print(f"[Discovery] Skipping {session_dir.name}: missing {', '.join(missing)}")

    return configs


def discover_video_only_sessions(
    sessions_root: Path,
    chunk_duration: int = 60,
    video_extensions: tuple = (".mp4", ".avi", ".mov", ".mkv")
) -> List[SessionConfig]:
    """
    Discover sessions containing only video files (no logs).

    Looks for video files directly in session folders or in a 'video' subdirectory.
    """
    configs = []

    if not sessions_root.exists():
        raise RuntimeError(f"Sessions root directory not found: {sessions_root}")

    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue

        # Look for video file in session directory
        video_files = [
            f for f in session_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]

        # Also check 'video' subdirectory
        video_subdir = session_dir / "video"
        if video_subdir.exists():
            video_files.extend([
                f for f in video_subdir.iterdir()
                if f.is_file() and f.suffix.lower() in video_extensions
            ])

        if video_files:
            # Use the first video file found
            video_path = video_files[0]
            configs.append(SessionConfig(
                session_folder=session_dir,
                agg_jsonl=None,  # No aggregation logs
                out_chunks_dir=session_dir / f"chunks_{chunk_duration}",
                video_path=video_path,
            ))
            print(f"[Discovery] Found video-only session: {session_dir.name} (video: {video_path.name})")
        else:
            print(f"[Discovery] Skipping {session_dir.name}: no video files found")

    return configs


def parse_args():
    p = argparse.ArgumentParser(
        description="Process session recordings with VLM labeling (supports multiple sessions and parallelization)"
    )

    # Session input
    session_group = p.add_mutually_exclusive_group(required=True)
    session_group.add_argument(
        "--session",
        type=Path,
        help="Path to single session folder"
    )
    session_group.add_argument(
        "--sessions-root",
        type=Path,
        help="Path to directory containing multiple session folders"
    )

    p.add_argument(
        "--agg-jsonl",
        default="aggregations.jsonl",
        help="Filename of aggregations jsonl inside session folder(s)"
    )
    p.add_argument(
        "--chunk-duration",
        type=int,
        default=60,
        help="Chunk duration in seconds"
    )
    p.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second for master video"
    )

    # Video-only mode
    p.add_argument(
        "--video-only",
        action="store_true",
        help="Process sessions containing only video files (no aggregation logs)"
    )
    p.add_argument(
        "--video-only-prompt",
        default="prompts/video_only_prompt.txt",
        help="Prompt file to use for video-only mode"
    )
    p.add_argument(
        "--video-extensions",
        nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv"],
        help="Video file extensions to look for in video-only mode"
    )

    # VLM client selection
    p.add_argument(
        "--prompt-client",
        choices=["gemini", "qwen3vl"],
        default="gemini",
        help="Which VLM client to use"
    )

    # Parallelization options
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of concurrent workers for Gemini API calls"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for vLLM batch processing"
    )

    # Qwen3VL options
    p.add_argument(
        "--qwen-model-path",
        default="Qwen/Qwen3-VL-8B-Thinking-FP8",
        help="Qwen model path or HuggingFace ID"
    )
    p.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="Port for vLLM server"
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)"
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length"
    )
    p.add_argument(
        "--server-startup-timeout",
        type=int,
        default=600,
        help="Timeout for vLLM server startup (seconds)"
    )

    # Video options (for standard mode)
    p.add_argument(
        "--label-video",
        action="store_true",
        help="Annotate video frames with mouse movements and clicks (standard mode only)"
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Discover sessions based on mode
    if args.session:
        # Single session mode
        if args.video_only:
            # Find video file in session
            video_files = [
                f for f in args.session.iterdir()
                if f.is_file() and f.suffix.lower() in tuple(args.video_extensions)
            ]
            video_subdir = args.session / "video"
            if video_subdir.exists():
                video_files.extend([
                    f for f in video_subdir.iterdir()
                    if f.is_file() and f.suffix.lower() in tuple(args.video_extensions)
                ])

            if not video_files:
                raise RuntimeError(f"No video files found in {args.session}")

            session_configs = [SessionConfig(
                session_folder=args.session,
                agg_jsonl=None,
                out_chunks_dir=args.session / f"chunks_{args.chunk_duration}",
                video_path=video_files[0],
            )]
            print(f"[Main] Processing single video-only session: {args.session.name}")
        else:
            session_configs = [SessionConfig(
                session_folder=args.session,
                agg_jsonl=args.session / args.agg_jsonl,
                out_chunks_dir=args.session / f"chunks_{args.chunk_duration}",
            )]
            print(f"[Main] Processing single session: {args.session.name}")
    else:
        # Multiple sessions mode
        if args.video_only:
            session_configs = discover_video_only_sessions(
                args.sessions_root,
                args.chunk_duration,
                tuple(args.video_extensions)
            )
        else:
            session_configs = discover_sessions(
                args.sessions_root,
                args.agg_jsonl,
                args.chunk_duration
            )

        if not session_configs:
            print(f"[Main] No valid sessions found in {args.sessions_root}")
            return
        print(f"[Main] Found {len(session_configs)} valid sessions")

    # Create output directories
    for config in session_configs:
        config.out_chunks_dir.mkdir(parents=True, exist_ok=True)

    # Process based on client type
    if args.prompt_client == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key is None:
            raise RuntimeError('GEMINI_API_KEY environment variable not set')

        print(f"[Main] Using Gemini client with {args.num_workers} concurrent workers")
        client = GeminiPromptClient(api_key=api_key)

        processor = SessionProcessor(
            prompt_client=client,
            num_workers=args.num_workers,
            use_batching=False,
            video_only_mode=args.video_only,
            video_only_prompt_file=args.video_only_prompt,
        )

        results = processor.process_multiple_sessions(
            session_configs,
            chunk_duration=args.chunk_duration,
            fps=args.fps,
            label_video=args.label_video if not args.video_only else False,
        )

        print(f"[Main] Processed {len(results)} sessions successfully")

    elif args.prompt_client == 'qwen3vl':
        print(f"[Main] Using Qwen3-VL client with batch size {args.batch_size}")

        with VLLMServerManager(
            model_path=args.qwen_model_path,
            port=args.vllm_port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        ) as server:
            server._wait_for_server(timeout=args.server_startup_timeout)

            client = Qwen3VLPromptClient(
                base_url=server.get_base_url(),
                model_name=args.qwen_model_path,
            )

            processor = SessionProcessor(
                prompt_client=client,
                batch_size=args.batch_size,
                use_batching=True,
                video_only_mode=args.video_only,
                video_only_prompt_file=args.video_only_prompt,
            )

            results = processor.process_multiple_sessions(
                session_configs,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                label_video=args.label_video if not args.video_only else False,
            )

            print(f"[Main] Processed {len(results)} sessions successfully")

        print("[Main] vLLM server stopped automatically")

    else:
        raise ValueError(f"Unknown prompt client: {args.prompt_client}")


if __name__ == '__main__':
    main()
