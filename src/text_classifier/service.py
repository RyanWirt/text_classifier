from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_classifier.vector_store import SearchResult, VectorStore

DEFAULT_REVIEW_PROMPT_TEMPLATE = """Review the case description using the reference context below.

Reference context:
{reference_context}

Case description:
{case_description}"""


@dataclass(frozen=True)
class ReviewResponse:
    response: str
    matches: list[SearchResult]


def load_review_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


class ReviewService:
    def __init__(
        self,
        agent,
        vector_store: VectorStore,
        top_k: int,
        prompt_template: str = DEFAULT_REVIEW_PROMPT_TEMPLATE,
    ) -> None:
        self.agent = agent
        self.vector_store = vector_store
        self.top_k = top_k
        self.prompt_template = prompt_template

    def review(self, text: str) -> ReviewResponse:
        matches = self.vector_store.search(text, limit=self.top_k)
        context = self._format_context(matches)
        prompt = self.prompt_template.format(
            reference_context=context,
            case_description=text.strip(),
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
