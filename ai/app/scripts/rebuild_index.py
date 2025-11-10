"""Rebuild index CLI script."""

import logging

import typer

from app.core.config import settings
from app.core.logging import setup_logging
from app.ingestion.storage import StorageManager
from app.vector.embeddings import get_embedding_provider
from app.vector.qdrant_client import delete_collection, ensure_collection, get_client

setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    force: bool = typer.Option(False, help="Force reindex even if collection exists"),
    collection_name: str = typer.Option(settings.collection_name, help="Qdrant collection name"),
):
    """Rebuild vector index from stored chunks."""
    logger.info(f"Rebuilding index: collection={collection_name}, force={force}")

    client = get_client()
    storage = StorageManager()

    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        if force:
            logger.info(f"Deleting existing collection: {collection_name}")
            delete_collection(client, collection_name)
        else:
            logger.warning(f"Collection {collection_name} already exists. Use --force to rebuild.")
            return

    # Ensure collection
    embedding_provider = get_embedding_provider()
    ensure_collection(client, collection_name, embedding_provider.vector_size)

    # Load all chunks and re-embed
    # This is simplified - in production, you'd load from storage and re-embed
    logger.info("Index rebuild initiated. Re-ingest data to populate index.")
    logger.info("Run: python -m app.scripts.ingest")


if __name__ == "__main__":
    app()


