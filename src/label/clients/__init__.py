from label.clients.client import VLMClient, CAPTION_SCHEMA
from label.clients.gemini import GeminiClient, GeminiResponse
from label.clients.vllm import VLLMClient, VLLMResponse


def create_client(client_type: str, **kwargs) -> VLMClient:
    if client_type == 'gemini':
        return GeminiClient(**kwargs)
    elif client_type == 'vllm':
        return VLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


__all__ = [
    "VLMClient",
    "GeminiClient",
    "GeminiResponse",
    "VLLMClient",
    "VLLMResponse",
    "CAPTION_SCHEMA",
    "create_client",
]
