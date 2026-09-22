from text_classifier.loaders import (
    DocumentLoaderRegistry,
    SourceDocument,
    chunk_document,
    discover_documents,
)


def test_discover_documents_filters_supported_extensions(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (data_dir / "b.json").write_text('{"name":"beta"}', encoding="utf-8")
    (data_dir / "c.csv").write_text("ignore", encoding="utf-8")

    paths = discover_documents(data_dir, (".txt", ".json"))

    assert [path.name for path in paths] == ["a.txt", "b.json"]


def test_registry_loads_json_xml_and_text(tmp_path):
    registry = DocumentLoaderRegistry()
    json_path = tmp_path / "sample.json"
    xml_path = tmp_path / "sample.xml"
    txt_path = tmp_path / "sample.txt"
    json_path.write_text('{"symptom":"anxiety"}', encoding="utf-8")
    xml_path.write_text("<root><item>panic</item></root>", encoding="utf-8")
    txt_path.write_text("flat affect", encoding="utf-8")

    loaded = {
        path.suffix: registry.load(path)[0].text
        for path in [json_path, xml_path, txt_path]
    }

    assert '"symptom": "anxiety"' in loaded[".json"]
    assert loaded[".xml"] == "panic"
    assert loaded[".txt"] == "flat affect"


def test_chunk_document_preserves_source_metadata():
    document = SourceDocument(
        source_id="data/reference.txt",
        text="abcdef ghijkl mnopqr stuvwx yz",
        metadata={"source_path": "data/reference.txt", "source_type": ".txt"},
    )
    chunks = chunk_document(document, chunk_size=8, chunk_overlap=2)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "data/reference.txt::chunk-1"
    assert chunks[0].metadata["source_path"] == "data/reference.txt"
    assert chunks[0].metadata["chunk_index"] == 1


def test_chunk_text_rejects_overlap_not_smaller_than_chunk_size():
    document = SourceDocument(
        source_id="data/reference.txt",
        text="some example text",
        metadata={"source_path": "data/reference.txt", "source_type": ".txt"},
    )

    try:
        chunk_document(document, chunk_size=10, chunk_overlap=10)
    except ValueError as exc:
        assert "chunk_overlap must be smaller than chunk_size" in str(exc)
    else:
        raise AssertionError("Expected chunk_document to reject invalid overlap")


def test_chunk_document_prefers_word_boundaries():
    document = SourceDocument(
        source_id="data/reference.txt",
        text="alpha beta gamma delta epsilon zeta",
        metadata={"source_path": "data/reference.txt", "source_type": ".txt"},
    )

    chunks = chunk_document(document, chunk_size=12, chunk_overlap=4)

    assert all(" " in chunk.text or len(chunk.text.split()) == 1 for chunk in chunks)
    assert all(not chunk.text.endswith(("alp", "bet", "gam")) for chunk in chunks)


def test_chunk_document_rejects_single_token_longer_than_chunk_size():
    document = SourceDocument(
        source_id="data/reference.txt",
        text="supercalifragilisticexpialidocious",
        metadata={"source_path": "data/reference.txt", "source_type": ".txt"},
    )

    try:
        chunk_document(document, chunk_size=10, chunk_overlap=2)
    except ValueError as exc:
        assert "single token exceeds chunk_size" in str(exc)
    else:
        raise AssertionError("Expected chunk_document to reject oversized tokens")
