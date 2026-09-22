from __future__ import annotations

from dataclasses import dataclass

from text_classifier.vector_store import SearchResult, VectorStore


@dataclass(frozen=True)
class ReviewResponse:
    response: str
    matches: list[SearchResult]


class ReviewService:
    def __init__(self, agent, vector_store: VectorStore, top_k: int) -> None:
        self.agent = agent
        self.vector_store = vector_store
        self.top_k = top_k

    def review(self, text: str) -> ReviewResponse:
        matches = self.vector_store.search(text, limit=self.top_k)
        context = self._format_context(matches)
        prompt = (
            "Reference context:\n"
            f"{context}\n\n"
            "Case description:\n"
            f"{text.strip()}"
        )
        result = self.agent.run_sync(prompt)
        return ReviewResponse(response=result.output, matches=matches)

    @staticmethod
    def _format_context(matches: list[SearchResult]) -> str:
        if not matches:
            return "No supporting reference context was retrieved."
        return "\n\n".join(
            f"Source: {match.metadata.get('source_path', 'unknown')} "
            f"(score={match.score:.3f})\n{match.text}"
            for match in matches
        )
