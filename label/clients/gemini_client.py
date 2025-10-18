from __future__ import annotations
import os
import time
from google import genai
from google.genai import types
from typing import Any, Dict, Optional
from label.clients.prompt_client import PromptClient


class GeminiPromptClient(PromptClient):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not provided via constructor or environment"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def upload_file(self, path: str) -> Any:
        print(f"[Gemini] Uploading file: {path}")
        video_file = self.client.files.upload(file=path)

        printed = False
        while True:
            video_file = self.client.files.get(name=video_file.name)

            state_name = getattr(getattr(video_file, "state", None), "name", None)

            if state_name == "PROCESSING":
                if not printed:
                    print("[Gemini] File is processing...")
                    printed = True
                time.sleep(2)
            elif state_name == "FAILED":
                raise RuntimeError("Gemini failed processing the file")
            elif state_name == "ACTIVE":
                break
            else:
                break

        print(f"[Gemini] Upload finished: uri={getattr(video_file, 'uri', video_file.name)}")
        return video_file

    def generate_content(
        self,
        prompt: str,
        file_descriptor: Optional[Any] = None,
        *,
        response_mime_type: str = "application/json",
        schema: Optional[Dict] = None
    ) -> Any:
        inputs = [prompt]
        if file_descriptor is not None:
            inputs.append(file_descriptor)

        generation_config = types.GenerateContentConfig(
            response_mime_type=response_mime_type,
            temperature=0.0,
            response_schema=schema
        )

        res = self.client.models.generate_content(
            model=self.model_name,
            contents=inputs,
            config=generation_config
        )
        return res
