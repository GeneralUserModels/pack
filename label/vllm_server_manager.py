import subprocess
import time
import requests
from pathlib import Path
from typing import Optional
import atexit


class VLLMServerManager:
    """Manages a vLLM server subprocess."""

    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        guided_decoding_backend: str = "outlines",
        host: str = "127.0.0.1",
    ):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.guided_decoding_backend = guided_decoding_backend
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"

        # Register cleanup on exit
        atexit.register(self.stop)

    def start(self, wait_for_ready: bool = True, timeout: int = 300):
        """Start the vLLM server."""
        if self.process is not None:
            print("[VLLMServer] Server already running")
            return

        cmd = [
            "vllm", "serve", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--guided-decoding-backend", self.guided_decoding_backend,
        ]

        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        print(f"[VLLMServer] Starting vLLM server on {self.base_url}")
        print(f"[VLLMServer] Command: {' '.join(cmd)}")

        # Start the server process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if wait_for_ready:
            print(f"[VLLMServer] Waiting for server to be ready (timeout: {timeout}s)...")
            self._wait_for_server(timeout)
            print("[VLLMServer] Server is ready!")
        else:
            print("[VLLMServer] Server started (not waiting for ready status)")

    def _wait_for_server(self, timeout: int = 300):
        """Wait for the server to become ready by polling the health endpoint."""
        start_time = time.time()
        health_url = f"{self.base_url}/health"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    return
            except (requests.RequestException, ConnectionError):
                pass

            # Check if process died
            if self.process.poll() is not None:
                # Process terminated
                stdout, _ = self.process.communicate()
                print("[VLLMServer] Server process terminated unexpectedly!")
                print("[VLLMServer] Output:")
                print(stdout)
                raise RuntimeError("vLLM server process died during startup")

            time.sleep(2)

        raise TimeoutError(f"vLLM server did not become ready within {timeout} seconds")

    def stop(self):
        """Stop the vLLM server."""
        if self.process is None:
            return

        print("[VLLMServer] Stopping vLLM server...")

        # Try graceful shutdown first
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
            print("[VLLMServer] Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("[VLLMServer] Server did not stop gracefully, forcing shutdown...")
            self.process.kill()
            self.process.wait()
            print("[VLLMServer] Server killed")

        self.process = None

    def is_running(self) -> bool:
        """Check if the server is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_base_url(self) -> str:
        """Get the base URL for the OpenAI API."""
        return f"{self.base_url}/v1"

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main_with_managed_server():
    """Example usage with context manager."""
    import argparse
    from label.clients.qwen3vl_openai_client import Qwen3VLOpenAIClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--agg-jsonl", default="aggregations.jsonl")
    parser.add_argument("--chunk-duration", type=int, default=60)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-30B-A3B-Thinking-FP8")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--label-video", action="store_true")
    parser.add_argument("--server-startup-timeout", type=int, default=300,
                        help="Timeout in seconds for server startup")
    args = parser.parse_args()

    session = Path(args.session)
    agg_path = session / args.agg_jsonl
    out = session / f"chunks_{args.chunk_duration}"
    out.mkdir(parents=True, exist_ok=True)

    # Start vLLM server and process session
    with VLLMServerManager(
        model_path=args.model_path,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    ) as server:
        # Create client pointing to the managed server
        client = Qwen3VLOpenAIClient(
            base_url=server.get_base_url(),
            model_name=args.model_path,
        )

        # Import process_session from your main module
        from your_module import process_session

        process_session(
            session, agg_path, out,
            chunk_duration=args.chunk_duration,
            fps=args.fps,
            prompt_client=client,
            label_video=args.label_video
        )

    print("Done! Server will be automatically stopped.")


if __name__ == "__main__":
    main_with_managed_server()
