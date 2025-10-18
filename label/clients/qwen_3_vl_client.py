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
    """Qwen3-VL client using vLLM's OpenAI-compatible API with true batch support."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
        max_tokens: int = 8192,
        enable_true_batching: bool = True,
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
        self.enable_true_batching = enable_true_batching

        print(f"[Qwen3VL-OpenAI] Connecting to {base_url}")
        print(f"[Qwen3VL-OpenAI] Model: {model_name}")
        print(f"[Qwen3VL-OpenAI] True batching: {enable_true_batching}")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def upload_file(self, path: str) -> Any:
        """
        Register a file and return its descriptor.
        For batch processing, we encode the file once and reuse it.
        """
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"File not found: {path}")

        file_type = "video" if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} else "image"

        # Encode file to base64 immediately for reuse
        with open(p, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        mime_type = "video/mp4" if file_type == "video" else "image/jpeg"
        data_url = f"data:{mime_type};base64,{b64_data}"

        print(f"[Qwen3VL-OpenAI] File registered: {path} (type: {file_type}, size: {len(b64_data)} bytes)")

        return {
            "path": str(p.absolute()),
            "type": file_type,
            "data_url": data_url,
        }

    def _build_messages(self, prompt: str, file_descriptor: Optional[Dict]) -> List[Dict]:
        """Build messages in OpenAI chat format with optional media."""
        if not file_descriptor:
            return [{"role": "user", "content": prompt}]

        file_type = file_descriptor.get("type", "image")
        data_url = file_descriptor.get("data_url")

        if not data_url:
            # Fallback if data_url not pre-computed
            file_path = file_descriptor.get("path")
            with open(file_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            mime_type = "video/mp4" if file_type == "video" else "image/jpeg"
            data_url = f"data:{mime_type};base64,{b64_data}"

        # Build message with media content
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

    def generate_content(
        self,
        prompt: Union[str, List[str]],
        file_descriptor: Optional[Union[Dict, List[Dict]]] = None,
        *,
        response_mime_type: str = "application/json",
        schema: Optional[Dict] = None,
    ) -> Union[Qwen3VLOpenAIResponse, List[Qwen3VLOpenAIResponse]]:
        """
        Generate content for single or multiple prompts.

        If enable_true_batching=True and input is a list, uses vLLM's batch API.
        Otherwise, processes sequentially.
        """
        if schema is None:
            schema = TASK_SCHEMA

        is_batch = isinstance(prompt, list)

        if is_batch and self.enable_true_batching:
            return self._generate_content_batch_true(
                prompt, file_descriptor, response_mime_type, schema
            )
        elif is_batch:
            return self._generate_content_batch_sequential(
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

        try:
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            }

            if schema:
                request_params["extra_body"] = {"guided_json": schema}

            completion = self.client.chat.completions.create(**request_params)
            content = completion.choices[0].message.content

            return Qwen3VLOpenAIResponse(
                text=content,
                completion=completion,
                schema=schema,
            )

        except Exception as e:
            print(f"[Qwen3VL-OpenAI] Error during generation: {e}")
            raise

    def _generate_content_batch_true(
        self,
        prompts: List[str],
        file_descriptors: Optional[List[Dict]] = None,
        response_mime_type: str = "application/json",
        schema: Optional[Dict] = None,
    ) -> List[Qwen3VLOpenAIResponse]:
        """
        Process multiple prompts using vLLM's batch API endpoint.

        Note: This requires vLLM to be configured with batching support.
        The batch endpoint allows processing multiple requests in a single call,
        which is more efficient than sequential processing.
        """
        if file_descriptors is None:
            file_descriptors = [None] * len(prompts)

        if len(prompts) != len(file_descriptors):
            raise ValueError("prompts and file_descriptors must have the same length")

        print(f"[Qwen3VL-OpenAI] Processing batch of {len(prompts)} samples...")

        # Build all messages
        all_messages = [
            self._build_messages(prompt, file_desc)
            for prompt, file_desc in zip(prompts, file_descriptors)
        ]

        # Prepare batch request
        request_params = {
            "model": self.model_name,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

        if schema:
            request_params["extra_body"] = {"guided_json": schema}

        try:
            from concurrent.futures import ThreadPoolExecutor

            responses = []
            with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                futures = []
                for messages in all_messages:
                    future = executor.submit(
                        self.client.chat.completions.create,
                        messages=messages,
                        **request_params
                    )
                    futures.append(future)

                # Collect results in order
                for future in futures:
                    completion = future.result()
                    content = completion.choices[0].message.content
                    responses.append(
                        Qwen3VLOpenAIResponse(
                            text=content,
                            completion=completion,
                            schema=schema,
                        )
                    )

            print(f"[Qwen3VL-OpenAI] Batch completed: {len(responses)} responses")
            return responses

        except Exception as e:
            print(f"[Qwen3VL-OpenAI] Batch processing failed: {e}")
            raise

    def _generate_content_batch_sequential(
        self,
        prompts: List[str],
        file_descriptors: Optional[List[Dict]] = None,
        response_mime_type: str = "application/json",
        schema: Optional[Dict] = None,
    ) -> List[Qwen3VLOpenAIResponse]:
        """Process multiple prompts sequentially (fallback method)."""
        if file_descriptors is None:
            file_descriptors = [None] * len(prompts)

        if len(prompts) != len(file_descriptors):
            raise ValueError("prompts and file_descriptors must have the same length")

        all_responses = []
        total_samples = len(prompts)

        print(f"[Qwen3VL-OpenAI] Processing {total_samples} samples sequentially...")

        for i, (prompt, file_desc) in enumerate(zip(prompts, file_descriptors)):
            try:
                response = self._generate_content_single(
                    prompt, file_desc, response_mime_type, schema
                )
                all_responses.append(response)
            except Exception as e:
                print(f"[Qwen3VL-OpenAI] Error on sample {i + 1}: {e}")
                raise

        print(f"[Qwen3VL-OpenAI] All {total_samples} samples processed")
        return all_responses


class Qwen3VLOpenAIResponse:
    """Wrapper for Qwen3-VL OpenAI API response with structured output support."""

    def __init__(self, text: str, completion: Any, schema: Optional[Dict] = None):
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
        """Convert response to dictionary for saving."""
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
