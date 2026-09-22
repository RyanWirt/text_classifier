from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pypdf import PdfReader


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


class Loader(Protocol):
    def load(self, path: Path) -> list[SourceDocument]: ...


class TextLoader:
    def load(self, path: Path) -> list[SourceDocument]:
        return [
            SourceDocument(
                source_id=path.as_posix(),
                text=path.read_text(encoding="utf-8"),
                metadata={"source_path": path.as_posix(), "source_type": path.suffix.lower()},
            )
        ]


class JsonLoader:
    def load(self, path: Path) -> list[SourceDocument]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            SourceDocument(
                source_id=path.as_posix(),
                text=json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
                metadata={"source_path": path.as_posix(), "source_type": ".json"},
            )
        ]


class XmlLoader:
    def load(self, path: Path) -> list[SourceDocument]:
        root = ET.parse(path).getroot()
        text = " ".join(part.strip() for part in root.itertext() if part.strip())
        return [
            SourceDocument(
                source_id=path.as_posix(),
                text=text,
                metadata={"source_path": path.as_posix(), "source_type": ".xml"},
            )
        ]


class PdfLoader:
    def load(self, path: Path) -> list[SourceDocument]:
        reader = PdfReader(path)
        documents: list[SourceDocument] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                SourceDocument(
                    source_id=f"{path.as_posix()}#page={index}",
                    text=text,
                    metadata={
                        "source_path": path.as_posix(),
                        "source_type": ".pdf",
                        "page": index,
                    },
                )
            )
        return documents


class DocumentLoaderRegistry:
    def __init__(self) -> None:
        text_loader = TextLoader()
        self._loaders: dict[str, Loader] = {
            ".txt": text_loader,
            ".md": text_loader,
            ".json": JsonLoader(),
            ".xml": XmlLoader(),
            ".pdf": PdfLoader(),
        }

    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaders))

    def load(self, path: Path) -> list[SourceDocument]:
        extension = path.suffix.lower()
        if extension not in self._loaders:
            raise ValueError(f"Unsupported document type: {extension or '<none>'}")
        return self._loaders[extension].load(path)


def discover_documents(data_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not data_dir.exists():
        return []
    normalized_extensions = {extension.lower() for extension in extensions}
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_document(
    document: SourceDocument, chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:
    chunks = chunk_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        TextChunk(
            chunk_id=f"{document.source_id}::chunk-{index}",
            text=chunk,
            metadata={**document.metadata, "chunk_index": index},
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
