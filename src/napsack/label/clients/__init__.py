from napsack.label.clients.client import VLMClient, CAPTION_SCHEMA, IMAGE_CAPTION_SCHEMA, DENSE_CAPTION_SCHEMA, DENSE_IMAGE_CAPTION_SCHEMA
from napsack.label.clients.litellm import LiteLLMClient
from napsack.label.clients.bigquery import BigQueryClient, BigQueryResponse
from napsack.label.clients.tinfoil import TinfoilClient


def create_client(client_type: str, **kwargs) -> VLMClient:
    if client_type in ('gemini', 'litellm', 'vllm'):
        return LiteLLMClient(**kwargs)
    elif client_type == 'bigquery':
        return BigQueryClient(**kwargs)
    elif client_type == 'tinfoil':
        return TinfoilClient(**kwargs)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


__all__ = [
    "VLMClient",
    "LiteLLMClient",
    "BigQueryClient",
    "BigQueryResponse",
    "TinfoilClient",
    "CAPTION_SCHEMA",
    "IMAGE_CAPTION_SCHEMA",
    "DENSE_CAPTION_SCHEMA",
    "DENSE_IMAGE_CAPTION_SCHEMA",
    "create_client",
]
