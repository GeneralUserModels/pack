import base64
import json
import re
from pathlib import Path
from typing import List, Union, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

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
        enable_batching: bool = True
    ):

        if OpenAI is None:
            raise RuntimeError("openai package not installed")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.enable_batching = enable_batching

        print(f"[VLLMClient] Connected to {base_url}")
        print(f"[VLLMClient] Model: {model_name}")
        print(f"[VLLMClient] Batching enabled: {enable_batching}")

    def upload_file(self, path: str) -> Dict:
        """
        Register a file and return its descriptor with pre-encoded data.
        For batch processing, we encode the file once and reuse it.
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

    def generate(self, prompt: Union[str, List[str]],
                 file_descriptor: Optional[Union[Dict, List[Dict]]] = None,
                 schema: Optional[Dict] = None) -> Union[VLLMResponse, List[VLLMResponse]]:

        if schema is None:
            schema = CAPTION_SCHEMA

        is_batch = isinstance(prompt, list)

        if is_batch and self.enable_batching:
            return self._generate_batch(prompt, file_descriptor, schema)
        elif is_batch:
            return self._generate_sequential(prompt, file_descriptor, schema)
        else:
            return self._generate_single(prompt, file_descriptor, schema)

    def _build_messages(self, prompt: str, file_desc: Optional[Dict]) -> List[Dict]:
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

    def _generate_single(self, prompt: str, file_desc: Optional[Dict],
                         schema: Optional[Dict]) -> VLLMResponse:
        """Process a single prompt with optional file."""
        messages = self._build_messages(prompt, file_desc)

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

    def _generate_batch(
        self, prompts: List[str],
        file_descs: Optional[List[Dict]],
        schema: Optional[Dict]
    ) -> List[VLLMResponse]:
        """
        Process multiple prompts in parallel using ThreadPoolExecutor.
        This is true batching - all requests are sent concurrently.
        """
        if file_descs is None:
            file_descs = [None] * len(prompts)

        if len(prompts) != len(file_descs):
            raise ValueError("prompts and file_descriptors must match in length")

        print(f"[VLLMClient] Processing batch of {len(prompts)} samples...")

        request_params = {
            "model": self.model_name,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

        if schema:
            request_params["extra_body"] = {"guided_json": schema}

        responses = []

        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = []

            for prompt, file_desc in zip(prompts, file_descs):
                messages = self._build_messages(prompt, file_desc)
                future = executor.submit(
                    self.client.chat.completions.create,
                    messages=messages,
                    **request_params
                )
                futures.append(future)

            for future in futures:
                try:
                    completion = future.result()
                    content = completion.choices[0].message.content
                    responses.append(VLLMResponse(content, completion, schema))
                except Exception as e:
                    print(f"[VLLMClient] Error in batch item: {e}")
                    responses.append(VLLMResponse(f'{{"error": "{str(e)}"}}', None, schema))

        print(f"[VLLMClient] Batch completed: {len(responses)} responses")
        return responses

    def _generate_sequential(
        self, prompts: List[str],
        file_descs: Optional[List[Dict]],
        schema: Optional[Dict]
    ) -> List[VLLMResponse]:
        """Process multiple prompts sequentially (fallback method)."""
        if file_descs is None:
            file_descs = [None] * len(prompts)

        if len(prompts) != len(file_descs):
            raise ValueError("prompts and file_descriptors must match in length")

        print(f"[VLLMClient] Processing {len(prompts)} samples sequentially...")

        responses = []
        for i, (prompt, file_desc) in enumerate(zip(prompts, file_descs)):
            try:
                response = self._generate_single(prompt, file_desc, schema)
                responses.append(response)
            except Exception as e:
                print(f"[VLLMClient] Error on sample {i + 1}: {e}")
                responses.append(VLLMResponse(f'{{"error": "{str(e)}"}}', None, schema))

        print(f"[VLLMClient] All {len(prompts)} samples processed")
        return responses
