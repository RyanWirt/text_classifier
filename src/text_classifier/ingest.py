from __future__ import annotations

from text_classifier.config import get_settings
from text_classifier.loaders import DocumentLoaderRegistry, chunk_document, discover_documents
from text_classifier.vector_store import VectorStore


def main() -> None:
    settings = get_settings()
    registry = DocumentLoaderRegistry()
    vector_store = VectorStore.from_settings(settings)

    paths = discover_documents(settings.data_dir, settings.data_file_types)
    documents = [document for path in paths for document in registry.load(path)]
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    ]
    count = vector_store.index_chunks(chunks)
    print(
        f"Indexed {count} chunks from {len(documents)} documents in "
        f"{settings.data_dir.as_posix()}"
    )


if __name__ == "__main__":
    main()
