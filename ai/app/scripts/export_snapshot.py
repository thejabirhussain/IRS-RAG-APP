"""Export snapshot CLI script."""

import json
import logging
from datetime import datetime
from pathlib import Path

import typer

from app.core.config import settings
from app.core.logging import setup_logging
from app.vector.qdrant_client import get_client

setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    output_dir: str = typer.Option("snapshots", help="Output directory for snapshots"),
    collection_name: str = typer.Option(settings.collection_name, help="Qdrant collection name"),
):
    """Export Qdrant collection snapshot."""
    logger.info(f"Exporting snapshot: collection={collection_name}")

    client = get_client()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snapshot_file = output_path / f"{collection_name}_{timestamp}.jsonl"

    # Scroll through all points
    points = []
    offset = None
    limit = 100

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # Exclude vectors to save space
        )

        points.extend(result[0])

        if len(result[0]) < limit:
            break

        offset = result[1]

    # Write to JSONL
    with open(snapshot_file, "w") as f:
        for point in points:
            doc = {
                "id": point.id,
                "payload": point.payload,
            }
            f.write(json.dumps(doc) + "\n")

    logger.info(f"Exported {len(points)} points to {snapshot_file}")


if __name__ == "__main__":
    app()


