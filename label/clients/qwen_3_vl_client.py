from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict, Optional

from label.clients.prompt_client import PromptClient


class Qwen3VLPromptClient(PromptClient):
    """Qwen3-VL client for video and image processing."""

    def __init__(self, model_path: str, device_map: str = "auto", dtype: str = "auto"):
        """
        Initialize Qwen3-VL client.

        Args:
            model_path: Path to the Qwen3-VL model
            device_map: Device mapping (default: "auto")
            dtype: Data type for model (default: "auto")
        """
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise RuntimeError(
                "Required packages not installed. Install with:\n"
                "pip install transformers qwen-vl-utils torch"
            )

        self.model_path = model_path
        self.device_map = device_map
        self.dtype = dtype

        print(f"[Qwen3VL] Loading model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map
        )
        self.process_vision_info = process_vision_info
        print("[Qwen3VL] Model loaded successfully")

    def upload_file(self, path: str) -> Any:
        """
        For Qwen3-VL, we return the file path as the descriptor.
        The actual processing happens in generate_content.
        """
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"File not found: {path}")

        file_type = "video" if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} else "image"
        print(f"[Qwen3VL] File registered: {path} (type: {file_type})")

        return {
            "path": str(p.absolute()),
            "type": file_type
        }

    def generate_content(
        self,
        prompt: str,
        file_descriptor: Optional[Any] = None,
        *,
        response_mime_type: str = "application/json",
        schema: Optional[Dict] = None
    ) -> Any:
        """
        Generate content using Qwen3-VL model.

        Args:
            prompt: The text prompt
            file_descriptor: Dict with "path" and "type" keys
            response_mime_type: Expected response format (currently ignored, always generates JSON)
            schema: Optional schema (currently ignored)

        Returns:
            Response object with text attribute containing the model output
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Add video or image if provided
        if file_descriptor:
            file_path = file_descriptor.get("path")
            file_type = file_descriptor.get("type", "image")

            if file_type == "video":
                messages[0]["content"].append({
                    "type": "video",
                    "video": f"file://{file_path}"
                })
            else:
                messages[0]["content"].append({
                    "type": "image",
                    "image": f"file://{file_path}"
                })

        print(f"[Qwen3VL] Processing {'video' if file_descriptor and file_descriptor.get('type') == 'video' else 'text'} prompt...")

        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision information
            images, videos, video_kwargs = self.process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)

            # Prepare inputs
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs
            )
            inputs = inputs.to(self.model.device)

            # Generate content
            print("[Qwen3VL] Generating response...")
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096)

            # Decode response
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            print("[Qwen3VL] Response generated successfully")

            # Try to extract JSON from response
            response_obj = Qwen3VLResponse(generated_text)
            return response_obj

        except Exception as e:
            print(f"[Qwen3VL] Error during generation: {e}")
            raise


class Qwen3VLResponse:
    """Wrapper for Qwen3-VL response."""

    def __init__(self, text: str):
        """
        Initialize response wrapper.

        Args:
            text: The raw text response from the model
        """
        self.text = text
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
            json_match = re.search(r'\{.*\}', self.text, re.DOTALL)
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
