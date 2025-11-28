"""Vector retrieval with filtering and similarity cutoff."""

import logging
from typing import Any, Optional, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, QueryRequest

from app.core.config import settings

logger = logging.getLogger(__name__)


def retrieve(
    client: QdrantClient,
    collection: str,
    query_vec: Union[list[float], np.ndarray],
    top_k: int = 40,
    cutoff: float = 0.22,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Retrieve similar chunks from Qdrant."""
    try:
        # Build query filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if key == "content_type":
                    conditions.append(
                        FieldCondition(key="content_type", match=MatchValue(value=value))
                    )
                elif key == "last_modified":
                    # Handle date filtering if needed
                    pass
                # Add more filter conditions as needed
            if conditions:
                query_filter = Filter(must=conditions)

        # Convert to list if numpy array
        if isinstance(query_vec, np.ndarray):
            query_vector = query_vec.tolist()
        else:
            query_vector = query_vec

        # Use query_points instead of search (new API in 1.16+)
        hits = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
            score_threshold=cutoff,
        )

        # Convert to list of dicts
        results = []
        # In new API, hits might be a QueryResponse object
        points = hits.points if hasattr(hits, 'points') else hits
        
        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "id": point.id,
                    "score": point.score,
                    "url": payload.get("url", ""),
                    "title": payload.get("title", ""),
                    "section_heading": payload.get("section_heading"),
                    "text": payload.get("text", ""),
                    "char_start": payload.get("char_start", 0),
                    "char_end": payload.get("char_end", 0),
                    "content_type": payload.get("content_type", "html"),
                    "crawl_ts": payload.get("crawl_ts", ""),
                    "last_modified": payload.get("last_modified"),
                    "embedding_model": payload.get("embedding_model", ""),
                }
            )

        logger.info(f"Retrieved {len(results)} chunks (cutoff={cutoff})")
        return results

    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}", exc_info=True)
        return []


def retrieve_with_cutoff(
    client: QdrantClient,
    collection: str,
    query_vec: Union[list[float], np.ndarray],
    top_k: int = None,
    cutoff: float = None,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks with similarity cutoff applied."""
    if top_k is None:
        top_k = settings.top_k
    if cutoff is None:
        cutoff = settings.similarity_cutoff

    return retrieve(client, collection, query_vec, top_k=top_k, cutoff=cutoff, filters=filters)

    # First retrieve more candidates
    candidates = retrieve(client, collection, query_vec, top_k=top_k * 2, cutoff=0.0, filters=filters)

    # Apply cutoff
    filtered = [c for c in candidates if c["score"] >= cutoff]

    # Limit to top_k
    filtered = filtered[:top_k]

    return filtered
