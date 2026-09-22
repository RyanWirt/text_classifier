from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from text_classifier.config import Settings


def embed_texts(texts: Sequence[str], settings: Settings) -> list[list[float]]:
    if not texts:
        return []

    client = OpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )
    response = client.embeddings.create(
        model=settings.litellm_embedding_model,
        input=list(texts),
    )
    return [
        list(item["embedding"] if isinstance(item, dict) else item.embedding)
        for item in response.data
    ]
