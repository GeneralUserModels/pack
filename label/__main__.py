from pathlib import Path
import argparse

from label.discovery import discover_sessions, discover_video_only_sessions
from label.clients import create_client
from label.session_processor import SessionProcessor
from label.vllm_server_manager import VLLMServerManager
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(
        description="Process session recordings with VLM labeling"
    )

    # Session input
    session_group = p.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--session", type=Path, help="Single session folder")
    session_group.add_argument("--sessions-root", type=Path, help="Multiple session folders")

    # Session processing options
    p.add_argument("--agg-jsonl", default="aggregations.jsonl", help="Aggregation filename")
    p.add_argument("--chunk-duration", type=int, default=60, help="Chunk duration (seconds)")
    p.add_argument("--fps", type=int, default=1, help="Frames per second")

    # Video-only mode
    p.add_argument("--video-only", action="store_true", help="Process videos without logs")
    p.add_argument("--video-only-prompt", default="prompts/video_only_prompt.txt")
    p.add_argument("--video-extensions", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv"])
    p.add_argument("--label-video", action="store_true", help="Annotate video frames")
    p.add_argument("--skip-existing", action="store_true", help="Skip already processed sessions")

    # Visualization
    p.add_argument("--visualize-session", action="store_true",
                   help="Create annotated video with captions and events after processing")

    # VLM client selection
    p.add_argument("--client", choices=["gemini", "qwen3vl"], default="gemini")
    p.add_argument("--model-id", default="", help="Model path or ID")

    # Parallelization
    p.add_argument("--num-workers", type=int, default=4, help="Concurrent workers (Gemini)")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (Qwen3VL)")

    # Qwen3VL server options
    qwen_group = p.add_argument_group("Qwen3VL Server Options")
    qwen_group.add_argument("--vllm-url", help="Connect to existing vLLM server (e.g., http://localhost:8000)")
    qwen_group.add_argument("--vllm-port", type=int, default=8000, help="Port for new vLLM server")
    qwen_group.add_argument("--tensor-parallel-size", type=int, default=1)
    qwen_group.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    qwen_group.add_argument("--max-model-len", type=int, default=None)
    qwen_group.add_argument("--server-startup-timeout", type=int, default=600)
    qwen_group.add_argument("--moe-expert-parallel", action="store_true")

    args = p.parse_args()

    if not args.model_id:
        args.model_id = 'gemini-2.5-flash' if args.client == 'gemini' else 'Qwen/Qwen3-VL-8B-Thinking-FP8'

    return args


def setup_session_configs(args):
    """Discover and prepare session configurations."""
    if args.session:
        from label.discovery import create_single_session_config
        configs = [create_single_session_config(
            args.session,
            args.agg_jsonl,
            args.chunk_duration,
            args.video_only,
            tuple(args.video_extensions)
        )]
        print(f"[Main] Processing single session: {args.session.name}")
    else:
        if args.video_only:
            configs = discover_video_only_sessions(
                args.sessions_root,
                args.chunk_duration,
                tuple(args.video_extensions)
            )
        else:
            configs = discover_sessions(
                args.sessions_root,
                args.agg_jsonl,
                args.chunk_duration,
                skip_existing=args.skip_existing,
            )

        if not configs:
            print(f"[Main] No valid sessions found in {args.sessions_root}")
            return []
        print(f"[Main] Found {len(configs)} sessions")

    for config in configs:
        config.out_chunks_dir.mkdir(parents=True, exist_ok=True)

    return configs


def process_with_gemini(args, session_configs):
    """Process sessions using Gemini client."""
    print(f"[Main] Using Gemini with {args.num_workers} workers")

    client = create_client(
        'gemini',
        model_name=args.model_id,
    )

    processor = SessionProcessor(
        prompt_client=client,
        num_workers=args.num_workers,
        use_batching=False,
        video_only_mode=args.video_only,
        video_only_prompt_file=args.video_only_prompt,
    )

    return processor.process_multiple_sessions(
        session_configs,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        label_video=args.label_video if not args.video_only else False,
    )


def process_with_qwen3vl(args, session_configs):
    """Process sessions using Qwen3VL client."""
    print(f"[Main] Using Qwen3VL with batch size {args.batch_size}")

    # Connect to existing server or start new one
    if args.vllm_url:
        print(f"[Main] Connecting to existing vLLM server at {args.vllm_url}")
        client = create_client(
            'qwen3vl',
            base_url=args.vllm_url if args.vllm_url.endswith('/v1') else f"{args.vllm_url}/v1",
            model_name=args.model_id
        )

        processor = SessionProcessor(
            prompt_client=client,
            batch_size=args.batch_size,
            use_batching=True,
            video_only_mode=args.video_only,
            video_only_prompt_file=args.video_only_prompt,
        )

        return processor.process_multiple_sessions(
            session_configs,
            chunk_duration=args.chunk_duration,
            fps=args.fps,
            label_video=args.label_video if not args.video_only else False,
        )
    else:
        print(f"[Main] Starting new vLLM server on port {args.vllm_port}")
        with VLLMServerManager(
            model_path=args.model_id,
            port=args.vllm_port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            moe_expert_parallel=args.moe_expert_parallel,
            startup_timeout=args.server_startup_timeout,
        ) as server:
            client = create_client(
                'qwen3vl',
                base_url=server.get_base_url(),
                model_name=args.model_id
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

        print("[Main] vLLM server stopped")
        return results


def main():
    args = parse_args()

    session_configs = setup_session_configs(args)
    if not session_configs:
        return

    if args.client == 'gemini':
        results = process_with_gemini(args, session_configs)
    elif args.client == 'qwen3vl':
        results = process_with_qwen3vl(args, session_configs)
    else:
        raise ValueError(f"Unknown client: {args.client}")

    print(f"[Main] ✓ Processed {len(results)} sessions successfully")

    if args.visualize_session:
        print("\n[Main] Creating visualizations...")
        from label.session_visualizer import SessionVisualizer

        visualizer = SessionVisualizer()

        for config in session_configs:
            session_id = config.session_folder.name

            matched_captions_path = config.session_folder / "matched_captions.jsonl"
            if not matched_captions_path.exists():
                print(f"[Main] Skipping visualization for {session_id}: no matched_captions.jsonl")
                continue

            try:
                output_video = config.session_folder / "annotated_session.mp4"
                visualizer.visualize_session(
                    config.session_folder,
                    output_video,
                    fps=args.fps
                )
                print(f"[Main] ✓ Created visualization: {output_video}")
            except Exception as e:
                print(f"[Main] ✗ Failed to visualize {session_id}: {e}")


if __name__ == '__main__':
    main()
