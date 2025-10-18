import subprocess
import time
import requests
import atexit
from typing import Optional


class VLLMServerManager:
    """Manages a vLLM server subprocess with automatic cleanup."""

    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        host: str = "127.0.0.1",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        guided_decoding_backend: str = "outlines",
        moe_expert_parallel: bool = False,
        startup_timeout: int = 600,
    ):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.guided_decoding_backend = guided_decoding_backend
        self.moe_expert_parallel = moe_expert_parallel
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"

        atexit.register(self.stop)

    def start(self):
        """Start the vLLM server and wait for it to be ready."""
        if self.process is not None:
            print("[vLLM] Server already running")
            return

        cmd = [
            "vllm", "serve", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--guided-decoding-backend", self.guided_decoding_backend,
        ]

        if self.moe_expert_parallel:
            cmd.append("--moe-expert-parallel")

        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        print(f"[vLLM] Starting server on {self.base_url}")
        print(f"[vLLM] Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        print(f"[vLLM] Waiting for server (timeout: {self.startup_timeout}s)...")
        self._wait_for_server()
        print("[vLLM] Server is ready!")

    def _wait_for_server(self):
        """Wait for server to become ready."""
        start_time = time.time()
        health_url = f"{self.base_url}/health"

        while time.time() - start_time < self.startup_timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    return
            except (requests.RequestException, ConnectionError):
                pass

            # Check if process died
            if self.process.poll() is not None:
                stdout, _ = self.process.communicate()
                print("[vLLM] Server process terminated unexpectedly!")
                print("[vLLM] Output:", stdout)
                raise RuntimeError("vLLM server died during startup")

            time.sleep(2)

        raise TimeoutError(f"vLLM server not ready within {self.startup_timeout}s")

    def stop(self):
        """Stop the vLLM server."""
        if self.process is None:
            return

        print("[vLLM] Stopping server...")

        self.process.terminate()
        try:
            self.process.wait(timeout=10)
            print("[vLLM] Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("[vLLM] Forcing shutdown...")
            self.process.kill()
            self.process.wait()
            print("[vLLM] Server killed")

        self.process = None

    def is_running(self) -> bool:
        """Check if server is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_base_url(self) -> str:
        """Get OpenAI API base URL."""
        return f"{self.base_url}/v1"

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
