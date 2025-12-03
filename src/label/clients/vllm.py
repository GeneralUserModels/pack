import base64
import json
import re
from pathlib import Path
from typing import Optional, Dict

from openai import OpenAI
from label.clients.client import VLMClient, CAPTION_SCHEMA


class VLLMResponse:
    def __init__(self, text: str, completion, schema: Optional[Dict] = None):
        self.text = text
        self.completion = completion
        self.schema = schema
        self._json = None
        self._parse_json()

    def _parse_json(self):
        """Try to extract and parse JSON from the response."""
        try:
            self._json = json.loads(self.text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}|\[.*\]', self.text, re.DOTALL)
            if json_match:
                try:
                    self._json = json.loads(json_match.group())
                except json.JSONDecodeError:
                    self._json = None
            else:
                self._json = None

    @property
    def json(self):
        """Return parsed JSON or the entire response as text."""
        if self._json is not None:
            return self._json
        return self.text


class VLLMClient(VLMClient):
    def __init__(
        self, base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "default",
        max_tokens: int = 8192,
    ):

        if OpenAI is None:
            raise RuntimeError("openai package not installed")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens

        print(f"[VLLMClient] Connected to {base_url}")
        print(f"[VLLMClient] Model: {model_name}")

    def upload_file(self, path: str, session_id: str = None) -> Dict:
        """
        Register a file and return its descriptor with pre-encoded data.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(p, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        file_type = "video" if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} else "image"
        mime_type = "video/mp4" if file_type == "video" else "image/jpeg"
        data_url = f"data:{mime_type};base64,{b64_data}"

        return {
            "path": str(p.absolute()),
            "type": file_type,
            "data_url": data_url,
        }

    def generate(
        self,
        prompt: str,
        file_descriptor: Optional[Dict] = None,
        schema: Optional[Dict] = None
    ) -> VLLMResponse:
        """Generate a response for a single prompt."""

        if schema is None:
            schema = CAPTION_SCHEMA

        messages = self._build_messages(prompt, file_descriptor)

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

        if schema:
            request_params["extra_body"] = {"guided_json": schema}

        try:
            completion = self.client.chat.completions.create(**request_params)
            content = completion.choices[0].message.content
            return VLLMResponse(content, completion, schema)
        except Exception as e:
            print(f"[VLLMClient] Error during generation: {e}")
            raise

    def _build_messages(self, prompt: str, file_desc: Optional[Dict]) -> list:
        """Build messages in OpenAI chat format with optional media."""
        if not file_desc:
            return [{"role": "user", "content": prompt}]

        file_type = file_desc.get("type", "image")
        data_url = file_desc.get("data_url")

        if not data_url:
            file_path = file_desc.get("path")
            with open(file_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            mime_type = "video/mp4" if file_type == "video" else "image/jpeg"
            data_url = f"data:{mime_type};base64,{b64_data}"

        if file_type == "video":
            content = [
                {"type": "video_url", "video_url": {"url": data_url}},
                {"type": "text", "text": prompt}
            ]
        else:
            content = [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt}
            ]

        return [{"role": "user", "content": content}]
