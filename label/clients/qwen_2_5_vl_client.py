from label.clients.prompt_client import PromptClient
from typing import Any, Dict, Optional


class LocalQwenPromptClient(PromptClient):
    def __init__(self):
        pass

    def upload_file(self, path: str) -> Any:
        raise NotImplementedError("LocalQwenPromptClient isn't implemented yet.")

    def generate_content(self, prompt: str, file_descriptor: Optional[Any] = None, *, response_mime_type: str = "application/json", schema: Optional[Dict] = None) -> Any:
        raise NotImplementedError("LocalQwenPromptClient isn't implemented yet.")
