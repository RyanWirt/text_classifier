from text_classifier import app as app_module
from text_classifier import ingest as ingest_module
from text_classifier.config import Settings
from text_classifier.loaders import SourceDocument


def build_settings(tmp_path):
    return Settings(
        app_host="127.0.0.1",
        app_port=5001,
        data_dir=tmp_path / "data",
        data_file_types=(".txt",),
        chunk_size=20,
        chunk_overlap=5,
        top_k=3,
        qdrant_url="http://localhost:6333",
        qdrant_collection="test_collection",
        litellm_base_url="http://localhost:4000",
        litellm_api_key="test-key",
        litellm_chat_model="openai/gpt-4.1-mini",
        litellm_embedding_model="openai/text-embedding-3-small",
        agent_prompt_file=tmp_path / "prompt.txt",
        review_prompt_file=tmp_path / "review_prompt.txt",
    )


def test_app_main_runs_with_loaded_settings(monkeypatch, tmp_path):
    settings = build_settings(tmp_path)
    calls = {}

    class FakeApp:
        def run(self, host, port):
            calls["host"] = host
            calls["port"] = port

    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "create_app", lambda current_settings: FakeApp())

    app_module.main()

    assert calls == {"host": "127.0.0.1", "port": 5001}


def test_ingest_main_indexes_loaded_documents(monkeypatch, tmp_path, capsys):
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir()
    source_path = settings.data_dir / "sample.txt"
    source_path.write_text("alpha beta gamma delta", encoding="utf-8")

    class FakeRegistry:
        def load(self, path):
            return [
                SourceDocument(
                    source_id=path.as_posix(),
                    text=path.read_text(encoding="utf-8"),
                    metadata={"source_path": path.as_posix(), "source_type": ".txt"},
                )
            ]

    class FakeVectorStore:
        def __init__(self):
            self.indexed = None

        def index_chunks(self, chunks):
            self.indexed = list(chunks)
            return len(self.indexed)

    fake_store = FakeVectorStore()

    class FakeVectorStoreFactory:
        @staticmethod
        def from_settings(current_settings):
            assert current_settings == settings
            return fake_store

    monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_module, "DocumentLoaderRegistry", lambda: FakeRegistry())
    monkeypatch.setattr(ingest_module, "VectorStore", FakeVectorStoreFactory)

    ingest_module.main()

    output = capsys.readouterr().out
    assert "Indexed 2 chunks from 1 documents" in output
    assert len(fake_store.indexed) == 2
