import os
from typing import Optional
from label.clients.prompt_client import PromptClient


def create_client(
    client_type: str,
    model_name: str = "",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> PromptClient:
    """
    Create a VLM client.

    Args:
        client_type: 'gemini' or 'qwen3vl'
        model_name: Model identifier
        base_url: Base URL for API (Qwen3VL only)
        api_key: API key (Gemini only)
        **kwargs: Additional client-specific parameters

    Returns:
        PromptClient instance
    """
    if client_type == 'gemini':
        from label.clients.gemini_client import GeminiPromptClient

        if api_key is None:
            api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY not set')

        return GeminiPromptClient(
            api_key=api_key,
            model_name=model_name or 'gemini-2.5-flash'
        )

    elif client_type == 'qwen3vl':
        from label.clients.qwen_3_vl_client import Qwen3VLPromptClient

        return Qwen3VLPromptClient(
            base_url=base_url or "http://localhost:8000/v1",
            model_name=model_name or 'Qwen/Qwen3-VL-8B-Thinking-FP8',
            **kwargs
        )

    else:
        raise ValueError(f"Unknown client type: {client_type}")
