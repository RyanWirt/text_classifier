from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from text_classifier.config import Settings
from text_classifier.embeddings import embed_texts
from text_classifier.loaders import TextChunk


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float
    metadata: dict[str, str | int]


class VectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_provider: Callable[[Sequence[str]], list[list[float]]],
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider

    @classmethod
    def from_settings(cls, settings: Settings) -> "VectorStore":
        client = QdrantClient(url=settings.qdrant_url)
        return cls(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding_provider=lambda texts: embed_texts(texts, settings),
        )

    def index_chunks(self, chunks: Sequence[TextChunk]) -> int:
        if not chunks:
            return 0

        vectors = self.embedding_provider([chunk.text for chunk in chunks])
        self._ensure_collection(len(vectors[0]))
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=vector,
                    payload={"text": chunk.text, "chunk_id": chunk.chunk_id, **chunk.metadata},
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )
        return len(chunks)

    def search(self, text: str, limit: int) -> list[SearchResult]:
        if not text.strip():
            return []
        if not self.client.collection_exists(self.collection_name):
            return []

        query_vector = self.embedding_provider([text])[0]
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [
            SearchResult(
                text=str(point.payload.get("text", "")),
                score=point.score,
                metadata={
                    key: value
                    for key, value in point.payload.items()
                    if key != "text"
                },
            )
            for point in response.points
        ]

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        except UnexpectedResponse:
            if not self.client.collection_exists(self.collection_name):
                raise
