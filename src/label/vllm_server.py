import subprocess
import time
import requests
import atexit
from typing import Optional


class VLLMServer:
    def __init__(self, model_path: str, port: int = 8000, host: str = "127.0.0.1",
                 tensor_parallel: int = 1, gpu_memory: float = 0.9,
                 max_model_len: Optional[int] = None, expert_parallel: bool = False,
                 startup_timeout: int = 600):

        self.model_path = model_path
        self.port = port
        self.host = host
        self.tensor_parallel = tensor_parallel
        self.gpu_memory = gpu_memory
        self.max_model_len = max_model_len
        self.expert_parallel = expert_parallel
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"

        atexit.register(self.stop)

    def start(self):
        if self.process is not None:
            return

        cmd = [
            "vllm", "serve", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel),
            "--gpu-memory-utilization", str(self.gpu_memory),
            "--guided-decoding-backend", "outlines",
        ]

        if self.expert_parallel:
            cmd.append("--enable-expert-parallel")
        print("Starting vLLM server with command:", " ".join(cmd))

        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._wait_for_ready()

    def _wait_for_ready(self):
        start_time = time.time()
        health_url = f"{self.base_url}/health"

        while time.time() - start_time < self.startup_timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    return
            except (requests.RequestException, ConnectionError):
                pass

            if self.process.poll() is not None:
                stdout, _ = self.process.communicate()
                raise RuntimeError(f"vLLM died during startup: {stdout}")

            time.sleep(2)

        raise TimeoutError(f"vLLM not ready within {self.startup_timeout}s")

    def stop(self):
        if self.process is None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

        self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def get_api_url(self) -> str:
        return f"{self.base_url}/v1"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
