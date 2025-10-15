from __future__ import annotations
from pathlib import Path
import json
import base64
from typing import Any, Dict, Optional, List, Union

from label.clients.prompt_client import PromptClient, TASK_SCHEMA

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class Qwen3VLPromptClient(PromptClient):
    """Qwen3-VL client using vLLM's OpenAI-compatible API for structured output."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
        max_tokens: int = 4096,
    ):
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI package not installed. Install with:\n"
                "pip install openai"
            )

        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens

        print(f"[Qwen3VL-OpenAI] Connecting to {base_url}")
        print(f"[Qwen3VL-OpenAI] Model: {model_name}")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def upload_file(self, path: str) -> Any:
        """
        Register a file and return its descriptor.
        For OpenAI API, we'll encode the file as base64.
        """
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"File not found: {path}")

        file_type = "video" if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} else "image"
        print(f"[Qwen3VL-OpenAI] File registered: {path} (type: {file_type})")

        return {
            "path": str(p.absolute()),
            "type": file_type
        }

    def _encode_file_to_base64_url(self, file_path: str, file_type: str) -> str:
        """Encode file to base64 data URL."""
        with open(file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        if file_type == "video":
            mime_type = "video/mp4"
        else:
            mime_type = "image/jpeg"

        return f"data:{mime_type};base64,{b64_data}"

    def _build_messages(self, prompt: str, file_descriptor: Optional[Dict]) -> List[Dict]:
        """Build messages in OpenAI chat format with optional media."""
        if not file_descriptor:
            return [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        file_path = file_descriptor.get("path")
        file_type = file_descriptor.get("type", "image")

        # Encode file as base64 data URL
        try:
            data_url = self._encode_file_to_base64_url(file_path, file_type)
            print(f"[Qwen3VL-OpenAI] Encoded {file_type} file ({len(data_url)} bytes)")
        except Exception as e:
            print(f"[Qwen3VL-OpenAI] Warning: Failed to encode file: {e}")
            # Fall back to text-only
            return [{"role": "user", "content": prompt}]

        # Build message with media content
        if file_type == "video":
            content = [
                {
                    "type": "video_url",
                    "video_url": {"url": data_url}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        else:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]

        return [
            {
                "role": "user",
                "content": content
            }
        ]

    def generate_content(
        self,
        prompt: Union[str, List[str]],
        file_descriptor: Optional[Union[Dict, List[Dict]]] = None,
        *,
        response_mime_type: str = "application/json",
        schema: Optional[Dict] = None,
    ) -> Union[Qwen3VLOpenAIResponse, List[Qwen3VLOpenAIResponse]]:
        if schema is None:
            schema = TASK_SCHEMA

        is_batch = isinstance(prompt, list)

        if is_batch:
            return self._generate_content_batch(
                prompt, file_descriptor, response_mime_type, schema
            )
        else:
            return self._generate_content_single(
                prompt, file_descriptor, response_mime_type, schema
            )

    def _generate_content_single(
        self,
        prompt: str,
        file_descriptor: Optional[Dict] = None,
        response_mime_type: str = "application/json",
        schema: Optional[Dict] = None,
    ) -> Qwen3VLOpenAIResponse:
        """Process a single prompt with optional file."""
        messages = self._build_messages(prompt, file_descriptor)

        print(f"[Qwen3VL-OpenAI] Processing {'video/image' if file_descriptor else 'text'} prompt...")

        try:
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            }

            # Add structured output constraint if schema provided
            if schema:
                request_params["extra_body"] = {"guided_json": schema}
                print("[Qwen3VL-OpenAI] Using guided JSON output")

            # Generate
            print("[Qwen3VL-OpenAI] Generating response...")
            completion = self.client.chat.completions.create(**request_params)

            # Extract response
            content = completion.choices[0].message.content

            print("[Qwen3VL-OpenAI] Response generated successfully")

            return Qwen3VLOpenAIResponse(
                text=content,
                completion=completion,
                schema=schema,
            )

        except Exception as e:
            print(f"[Qwen3VL-OpenAI] Error during generation: {e}")
            raise

    def _generate_content_batch(
        self,
        prompts: List[str],
        file_descriptors: Optional[List[Dict]] = None,
        response_mime_type: str = "application/json",
        schema: Optional[Dict] = None,
    ) -> List[Qwen3VLOpenAIResponse]:
        """Process multiple prompts sequentially (OpenAI API doesn't support true batching)."""
        if file_descriptors is None:
            file_descriptors = [None] * len(prompts)

        if len(prompts) != len(file_descriptors):
            raise ValueError("prompts and file_descriptors must have the same length")

        all_responses = []
        total_samples = len(prompts)

        print(f"[Qwen3VL-OpenAI] Processing {total_samples} samples sequentially...")

        for i, (prompt, file_desc) in enumerate(zip(prompts, file_descriptors)):
            print(f"[Qwen3VL-OpenAI] Sample {i + 1}/{total_samples}...")

            try:
                response = self._generate_content_single(
                    prompt, file_desc, response_mime_type, schema
                )
                all_responses.append(response)
                print(f"[Qwen3VL-OpenAI] Sample {i + 1} completed successfully")
            except Exception as e:
                print(f"[Qwen3VL-OpenAI] Error on sample {i + 1}: {e}")
                raise

        print(f"[Qwen3VL-OpenAI] All {total_samples} samples processed successfully")
        return all_responses


class Qwen3VLOpenAIResponse:
    """Wrapper for Qwen3-VL OpenAI API response with structured output support."""

    def __init__(self, text: str, completion: Any, schema: Optional[Dict] = None):
        """
        Initialize response wrapper.

        Args:
            text: The generated text response
            completion: The OpenAI completion object
            schema: JSON schema used for generation
        """
        self.text = text
        self.completion = completion
        self.schema = schema
        self._json = None
        self._parse_json()

    def _parse_json(self):
        """Try to extract and parse JSON from the response."""
        try:
            # Try to parse the entire response as JSON
            self._json = json.loads(self.text)
        except json.JSONDecodeError:
            # Try to find JSON within the text
            import re
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

    @property
    def token_ids(self) -> List[int]:
        """Return empty list (token IDs not available via OpenAI API)."""
        return []

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary for saving.

        Returns:
            Dictionary with json, usage stats, and raw text
        """
        usage = {}
        if hasattr(self.completion, "usage") and self.completion.usage:
            usage = {
                "prompt_tokens": self.completion.usage.prompt_tokens,
                "completion_tokens": self.completion.usage.completion_tokens,
                "total_tokens": self.completion.usage.total_tokens,
            }

        return {
            "json": self.json,
            "usage": usage,
            "raw_text": self.text,
        }
