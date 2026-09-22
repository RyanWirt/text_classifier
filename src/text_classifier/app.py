from __future__ import annotations

from flask import Flask, jsonify, request
from pydantic import BaseModel, ValidationError
from werkzeug.exceptions import BadRequest

from text_classifier.agent import build_agent
from text_classifier.config import Settings, get_settings
from text_classifier.service import ReviewService
from text_classifier.vector_store import VectorStore


class ReviewRequest(BaseModel):
    text: str


def create_app(
    settings: Settings | None = None, review_service: ReviewService | None = None
) -> Flask:
    settings = settings or get_settings()
    review_service = review_service or ReviewService(
        agent=build_agent(settings),
        vector_store=VectorStore.from_settings(settings),
        top_k=settings.top_k,
    )
    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[dict[str, str], int]:
        return {"status": "ok"}, 200

    @app.post("/review")
    def review() -> tuple[object, int]:
        try:
            payload = ReviewRequest.model_validate(request.get_json(silent=False) or {})
        except BadRequest:
            return jsonify({"error": "Invalid JSON body"}), 400
        except ValidationError as exc:
            return jsonify({"error": "Invalid request body", "details": exc.errors()}), 400
        if not payload.text.strip():
            return jsonify({"error": "The 'text' field must not be empty"}), 400

        result = review_service.review(payload.text)
        return (
            jsonify(
                {
                    "response": result.response,
                    "matches": [
                        {
                            "score": match.score,
                            "text": match.text,
                            "metadata": match.metadata,
                        }
                        for match in result.matches
                    ],
                }
            ),
            200,
        )

    return app


def main() -> None:
    settings = get_settings()
    create_app(settings).run(host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    main()
