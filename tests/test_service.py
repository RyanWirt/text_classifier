from types import SimpleNamespace

from text_classifier.app import create_app
from text_classifier.service import ReviewResponse, ReviewService
from text_classifier.vector_store import SearchResult


class FakeAgent:
    def __init__(self):
        self.last_prompt = None

    def run_sync(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(output="clinical summary")


class FakeVectorStore:
    def search(self, text, limit):
        assert text == "patient text"
        assert limit == 2
        return [
            SearchResult(
                text="Relevant DSM-5 excerpt",
                score=0.91,
                metadata={"source_path": "data/reference.pdf"},
            )
        ]


def test_review_service_uses_retrieved_context():
    agent = FakeAgent()
    service = ReviewService(agent=agent, vector_store=FakeVectorStore(), top_k=2)

    result = service.review("patient text")

    assert result.response == "clinical summary"
    assert "Relevant DSM-5 excerpt" in agent.last_prompt
    assert "data/reference.pdf" in agent.last_prompt


def test_review_endpoint_accepts_json():
    class FakeReviewService:
        def review(self, text):
            assert text == "patient text"
            return ReviewResponse(
                response="ok",
                matches=[
                    SearchResult(
                        text="context",
                        score=0.8,
                        metadata={"source_path": "data/reference.pdf"},
                    )
                ],
            )

    app = create_app(review_service=FakeReviewService())
    client = app.test_client()

    response = client.post("/review", json={"text": "patient text"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["response"] == "ok"
    assert body["matches"] == [
        {
            "score": 0.8,
            "text": "context",
            "metadata": {"source_path": "data/reference.pdf"},
        }
    ]


def test_review_endpoint_rejects_blank_text():
    class NeverCalledReviewService:
        def review(self, text):
            raise AssertionError("review should not be called for blank input")

    app = create_app(review_service=NeverCalledReviewService())
    client = app.test_client()

    response = client.post("/review", json={"text": "  "})

    assert response.status_code == 400
    assert response.get_json()["error"] == "The 'text' field must not be empty"


def test_review_endpoint_rejects_invalid_json():
    class NeverCalledReviewService:
        def review(self, text):
            raise AssertionError("review should not be called for invalid json")

    app = create_app(review_service=NeverCalledReviewService())
    client = app.test_client()

    response = client.post(
        "/review",
        data='{"text": "missing end"',
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid JSON body"


def test_review_endpoint_rejects_missing_text():
    class NeverCalledReviewService:
        def review(self, text):
            raise AssertionError("review should not be called for invalid payloads")

    app = create_app(review_service=NeverCalledReviewService())
    client = app.test_client()

    response = client.post("/review", json={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid request body"
    assert response.get_json()["details"]


def test_review_endpoint_rejects_non_string_text():
    class NeverCalledReviewService:
        def review(self, text):
            raise AssertionError("review should not be called for invalid payloads")

    app = create_app(review_service=NeverCalledReviewService())
    client = app.test_client()

    response = client.post("/review", json={"text": {"value": "wrong type"}})

    assert response.status_code == 400
