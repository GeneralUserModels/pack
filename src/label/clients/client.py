from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union

CAPTION_SCHEMA = {
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


class VLMClient(ABC):
    @abstractmethod
    def upload_file(self, path: str, session_id: str = None) -> Any:
        pass

    @abstractmethod
    def generate(self, prompt: Union[str, List[str]],
                 file_descriptor: Optional[Union[Any, List[Any]]] = None,
                 schema: Optional[Dict] = None) -> Union[Any, List[Any]]:
        pass
