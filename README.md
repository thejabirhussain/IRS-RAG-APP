# IRS RAG Monorepo

This repository hosts two isolated codebases:

- ai/  — Python backend (FastAPI) with crawler, ingestion, embeddings, vector DB, retriever, and RAG pipeline
- ui/  — React + Vite frontend (isolated from backend)

## Getting Started

### Backend (ai/)

- See `ai/README.md` for full backend docs
- Start services:
  - `make -C ai up`
  - API: http://localhost:8000
  - Qdrant: http://localhost:6333

### Frontend (ui/)

- See `ui/README.md` for frontend docs
- Dev server (after deps install): http://localhost:5173

## Docker Compose

The root `docker-compose.yml` orchestrates Qdrant and the API using the backend in `ai/`.






