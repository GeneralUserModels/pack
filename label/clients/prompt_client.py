from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

TASK_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "string"},
            "end": {"type": "string"},
            "caption": {"type": "string"}
        },
        "required": ["start", "end", "caption"]
    }
}


class PromptClient(ABC):
    """Abstract prompt client. Implement upload_file and generate_content for each backend."""

    @abstractmethod
    def upload_file(self, path: str) -> Any:
        pass

    @abstractmethod
    def generate_content(
        self, prompt: str,
        file_descriptor: Optional[Any] = None,
        *,
        response_mime_type: str = "application/json",
        schema: Optional[Dict] = None
    ) -> Any:
        pass
