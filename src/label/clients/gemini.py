from typing import Optional, Any, Dict
import json
import os
import time
from label.clients.client import VLMClient, CAPTION_SCHEMA
from google import genai
from google.genai import types


class GeminiResponse:
    def __init__(self, response):
        self.response = response
        self._json = None

        # Extract token usage from response
        self.input_tokens = 0
        self.output_tokens = 0

        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            self.input_tokens = getattr(usage, 'prompt_token_count', 0)
            self.output_tokens = getattr(usage, 'candidates_token_count', 0)

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def json(self):
        if self._json is None:
            self._json = json.loads(self.text)
        return self._json


class GeminiClient(VLMClient):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        if genai is None:
            raise RuntimeError("google-genai not installed")

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def upload_file(self, path: str) -> Any:
        video_file = self.client.files.upload(file=path)

        while True:
            video_file = self.client.files.get(name=video_file.name)
            state = getattr(getattr(video_file, "state", None), "name", None)

            if state == "PROCESSING":
                time.sleep(2)
            elif state == "FAILED":
                raise RuntimeError("Gemini failed processing file")
            elif state == "ACTIVE":
                break
            else:
                break

        return video_file

    def generate(self, prompt: str, file_descriptor: Optional[Any] = None,
                 schema: Optional[Dict] = None) -> GeminiResponse:
        inputs = [prompt]
        if file_descriptor:
            inputs.append(file_descriptor)

        if "gemini-3" in self.model_name:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                response_schema=schema or CAPTION_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
            )
        else:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                response_schema=schema or CAPTION_SCHEMA,
            )

        res = self.client.models.generate_content(
            model=self.model_name,
            contents=inputs,
            config=config
        )

        response = GeminiResponse(res)

        # Track tokens
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens

        return response

    def get_token_stats(self) -> Dict[str, int]:
        """Get current token usage statistics."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }

    def reset_token_stats(self):
        """Reset token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def print_token_stats(self, prefix: str = ""):
        """Print token usage statistics."""
        stats = self.get_token_stats()
        print(f"\n{prefix}Token Usage:")
        print(f"  Input tokens:  {stats['input_tokens']:,}")
        print(f"  Output tokens: {stats['output_tokens']:,}")
        print(f"  Total tokens:  {stats['total_tokens']:,}")
