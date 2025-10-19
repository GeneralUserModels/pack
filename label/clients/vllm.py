from typing import List, Union, Optional, Dict
import json
from pathlib import Path
import base64

from openai import OpenAI
from label.clients.client import VLMClient, CAPTION_SCHEMA


class VLLMResponse:
    def __init__(self, completion, schema: Optional[Dict] = None):
        self.completion = completion
        self.schema = schema
        self._json = None
        self._parse()

    def _parse(self):
        text = self.completion.choices[0].message.content
        try:
            self._json = json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
            if match:
                try:
                    self._json = json.loads(match.group())
                except json.JSONDecodeError:
                    self._json = None

    @property
    def text(self) -> str:
        return self.completion.choices[0].message.content

    @property
    def json(self):
        return self._json if self._json is not None else self.text


class VLLMClient(VLMClient):
    def __init__(
        self, base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "default",
        max_tokens: int = 8192,
        batch_parallel: bool = True
    ):

        if OpenAI is None:
            raise RuntimeError("openai package not installed")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_parallel = batch_parallel

    def upload_file(self, path: str) -> Dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(p, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        file_type = "video" if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} else "image"
        mime_type = "video/mp4" if file_type == "video" else "image/jpeg"

        return {
            "type": file_type,
            "data_url": f"data:{mime_type};base64,{b64_data}"
        }

    def generate(self, prompt: Union[str, List[str]],
                 file_descriptor: Optional[Union[Dict, List[Dict]]] = None,
                 schema: Optional[Dict] = None) -> Union[VLLMResponse, List[VLLMResponse]]:

        is_batch = isinstance(prompt, list)

        if is_batch:
            return self._generate_batch(prompt, file_descriptor, schema)
        else:
            return self._generate_single(prompt, file_descriptor, schema)

    def _build_messages(self, prompt: str, file_desc: Optional[Dict]) -> List[Dict]:
        if not file_desc:
            return [{"role": "user", "content": prompt}]

        content_type = "video_url" if file_desc["type"] == "video" else "image_url"
        url_key = "video_url" if file_desc["type"] == "video" else "image_url"

        content = [
            {"type": content_type, url_key: {"url": file_desc["data_url"]}},
            {"type": "text", "text": prompt}
        ]

        return [{"role": "user", "content": content}]

    def _generate_single(self, prompt: str, file_desc: Optional[Dict],
                         schema: Optional[Dict]) -> VLLMResponse:

        messages = self._build_messages(prompt, file_desc)

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

        if schema:
            params["extra_body"] = {"guided_json": schema or CAPTION_SCHEMA}

        completion = self.client.chat.completions.create(**params)
        return VLLMResponse(completion, schema)

    def _generate_batch(
        self, prompts: List[str],
        file_descs: Optional[List[Dict]],
        schema: Optional[Dict]
    ) -> List[VLLMResponse]:

        if file_descs is None:
            file_descs = [None] * len(prompts)

        if len(prompts) != len(file_descs):
            raise ValueError("prompts and file_descriptors must match in length")

        from concurrent.futures import ThreadPoolExecutor

        params = {
            "model": self.model_name,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

        if schema:
            params["extra_body"] = {"guided_json": schema or CAPTION_SCHEMA}

        responses = []

        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = []
            for prompt, file_desc in zip(prompts, file_descs):
                messages = self._build_messages(prompt, file_desc)
                future = executor.submit(
                    self.client.chat.completions.create,
                    messages=messages,
                    **params
                )
                futures.append(future)

            for future in futures:
                completion = future.result()
                responses.append(VLLMResponse(completion, schema))

        return responses
