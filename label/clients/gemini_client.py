from __future__ import annotations
import os
import time
import google.generativeai as genai
from typing import Any, Dict, Optional

from label.clients.prompt_client import PromptClient


class GeminiPromptClient(PromptClient):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        if genai is None:
            raise RuntimeError("google.generativeai not installed or importable. Install with pip install google-generativeai")
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not provided via constructor or environment")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    def upload_file(self, path: str) -> Any:
        print(f"[Gemini] Uploading file: {path}")
        video_file = genai.upload_file(path=str(path))
        printed = False
        while getattr(video_file, "state", None) and getattr(video_file.state, "name", None) == "PROCESSING":
            if not printed:
                print("[Gemini] File is processing...")
                printed = True
            video_file = genai.get_file(video_file.name)
        if getattr(video_file, "state", None) and getattr(video_file.state, "name", None) == "FAILED":
            raise RuntimeError("Gemini failed processing the file")
        print(f"[Gemini] Upload finished: uri={getattr(video_file, 'uri', getattr(video_file, 'name', str(path)))}")
        return video_file

    def generate_content(
            self, prompt: str,
            file_descriptor: Optional[Any] = None,
            *,
            response_mime_type: str = "application/json",
            schema: Optional[Dict] = None
    ) -> Any:
        inputs = [prompt]
        if file_descriptor is not None:
            inputs.append(file_descriptor)

        generation_config = genai.GenerationConfig(
            response_mime_type=response_mime_type,
            temperature=0.0,
            response_schema=schema
        )
        res = self._model.generate_content(inputs, generation_config=generation_config)
        return res
