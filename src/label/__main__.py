from pathlib import Path
import argparse
from dotenv import load_dotenv

from label.discovery import discover_sessions, discover_video_sessions, create_single_config
from label.clients import create_client
from label.processor import Processor
from label.vllm_server import VLLMServer
from label.visualizer import Visualizer

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Process session recordings with VLM")

    session_group = p.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--session", type=Path)
    session_group.add_argument("--sessions-root", type=Path)

    p.add_argument("--chunk-duration", type=int, default=60, help="Chunk duration in seconds")
    p.add_argument("--fps", type=int, default=1, help="Frames per second for video processing")

    p.add_argument("--video-only", action="store_true", help="Process video only without screenshots or annotations")
    p.add_argument("--video-extensions", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv"], help="Video file extensions to consider")
    p.add_argument("--prompt-file", default=None, help="Path to prompt file (default: prompts/default.txt or prompts/video_only.txt if video only)")
    p.add_argument("--annotate", action="store_true", help="Annotate videos with cursor positions and clicks (only for standard processing)")
    p.add_argument("--skip-existing", action="store_true", help="Skip sessions that have already been processed")
    p.add_argument("--visualize", action="store_true", help="Create annotated video visualizations after processing")

    p.add_argument("--client", choices=["gemini", "vllm"], default="gemini")
    p.add_argument("--model", default="")
    p.add_argument("--num-workers", type=int, default=4)

    vllm_group = p.add_argument_group("vLLM Options")
    vllm_group.add_argument("--vllm-url")
    vllm_group.add_argument("--vllm-port", type=int, default=8000)
    vllm_group.add_argument("--tensor-parallel", type=int, default=1)
    vllm_group.add_argument("--gpu-memory", type=float, default=0.9)
    vllm_group.add_argument("--max-model-len", type=int)
    vllm_group.add_argument("--expert-parallel", action="store_true")
    vllm_group.add_argument("--enforce-eager", action="store_true")
    vllm_group.add_argument("--startup-timeout", type=int, default=600)

    args = p.parse_args()

    if not args.model:
        args.model = 'gemini-2.5-flash' if args.client == 'gemini' else 'Qwen/Qwen3-VL-8B-Thinking-FP8'
    if not args.prompt_file:
        args.prompt_file = "prompts/video_only.txt" if args.video_only else "prompts/default.txt"

    return args


def setup_configs(args):
    if args.session:
        configs = [create_single_config(
            args.session,
            args.chunk_duration,
            args.video_only,
            tuple(args.video_extensions),
        )]
    else:
        if args.video_only:
            configs = discover_video_sessions(
                args.sessions_root,
                args.chunk_duration,
                tuple(args.video_extensions),
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
        video_only=args.video_only,
        prompt_file=args.prompt_file,
    )

    return processor.process_sessions(
        configs,
        fps=args.fps,
        annotate=args.annotate and not args.video_only,
    )


def process_with_vllm(args, configs):
    if args.vllm_url:
        client = create_client(
            'vllm',
            base_url=args.vllm_url if args.vllm_url.endswith('/v1') else f"{args.vllm_url}/v1",
            model_name=args.model
        )

        processor = Processor(
            client=client,
            num_workers=args.num_workers,
            video_only=args.video_only,
            prompt_file=args.prompt_file,
        )

        return processor.process_sessions(
            configs,
            fps=args.fps,
            annotate=args.annotate and not args.video_only,
        )
    else:
        with VLLMServer(
            model_path=args.model,
            port=args.vllm_port,
            tensor_parallel=args.tensor_parallel,
            gpu_memory=args.gpu_memory,
            max_model_len=args.max_model_len,
            expert_parallel=args.expert_parallel,
            startup_timeout=args.startup_timeout,
            enforce_eager=args.enforce_eager,
        ) as server:
            client = create_client(
                'vllm',
                base_url=server.get_api_url(),
                model_name=args.model
            )

            processor = Processor(
                client=client,
                num_workers=args.num_workers,
                video_only=args.video_only,
                prompt_file=args.prompt_file,
            )

            return processor.process_sessions(
                configs,
                fps=args.fps,
                annotate=args.annotate and not args.video_only,
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
