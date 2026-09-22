# text_classifier

Starter workspace for experimenting with `pydantic-ai` agents that review DSM-V-style reference material and respond to JSON requests from a Flask app through a separate LiteLLM instance.

## What is included

- VS Code dev container support
- Local Qdrant vector database via Docker Compose
- Pluggable ingestion pipeline for `pdf`, `json`, `xml`, and `txt`
- Starter Flask API that accepts `{"text": "..."}` and retrieves supporting context before calling a `pydantic-ai` agent
- Prompt file at `prompts/dsm_v_review.txt` so you can iterate on behavior without changing code

## Project layout

- `data/` — drop source files here
- `prompts/dsm_v_review.txt` — agent system prompt
- `src/text_classifier/ingest.py` — document loading, chunking, vector upload entry point
- `src/text_classifier/app.py` — Flask API
- `docker-compose.yml` — Qdrant + devcontainer service

## Quick start

1. Copy `.env.example` to `.env`.
2. Update `LITELLM_BASE_URL`, `LITELLM_API_KEY`, `LITELLM_CHAT_MODEL`, and `LITELLM_EMBEDDING_MODEL` to match your separate LiteLLM instance. The starter defaults use LiteLLM-style model names such as `openai/gpt-4.1-mini`.
3. Open the repository in the dev container, or run `docker compose up -d qdrant` locally.
4. Install dependencies:

   ```bash
   python -m pip install -e .[dev]
   ```

5. Put your PDFs or other supported files into `data/`.
6. Build the vector index:

   ```bash
   make build-data
   ```

7. Start the API:

   ```bash
   make run
   ```

8. Send a review request:

   ```bash
   curl -X POST http://localhost:5000/review \
     -H "Content-Type: application/json" \
     -d '{"text":"Patient reports persistent low mood, insomnia, and loss of interest."}'
   ```

## Dev container

Open the repository in VS Code and choose **Reopen in Container**. The dev container starts:

- `app` — the Python workspace container
- `qdrant` — the local vector DB on `http://localhost:6333`

Inside the container, the workspace service talks to Qdrant using `http://qdrant:6333`. It can reach a separate LiteLLM instance through `http://host.docker.internal:4000` by default.

## Supported data types

Supported extensions are configured with `DATA_FILE_TYPES` in `.env`:

```env
DATA_FILE_TYPES=.pdf,.json,.xml,.txt
```

The loader registry lives in `src/text_classifier/loaders.py`.

- To enable only PDFs, set `DATA_FILE_TYPES=.pdf`
- To ingest XML and JSON, set `DATA_FILE_TYPES=.xml,.json`
- To add a new type later, add a loader class and register the new extension in `DocumentLoaderRegistry`

## Notes on the starter flow

- PDFs are split page-by-page, then chunked before upload
- JSON files are normalized into formatted text before embedding
- XML files are converted from element text into plain text before embedding
- The ingest step uses LiteLLM's OpenAI-compatible embeddings endpoint, and the API retrieves the top matching chunks from Qdrant before the agent responds

## Useful commands

```bash
make install
make build-data
make run
make test
```