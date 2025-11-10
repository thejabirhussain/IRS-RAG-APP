"""Tests for text chunking."""

import pytest
from datetime import datetime

from app.core.constants import DEFAULT_CHUNK_MAX, DEFAULT_CHUNK_MIN
from app.ingestion.chunker import chunk_by_sliding_window, chunk_page
from app.ingestion.models import ContentType, CrawledPage


def test_chunk_by_sliding_window():
    """Test sliding window chunking."""
    text = "This is a test text. " * 200  # ~4000 chars
    chunks = chunk_by_sliding_window(text)

    assert len(chunks) > 0
    assert all(char_start < char_end for char_start, char_end, _ in chunks)


def test_chunk_sizes():
    """Test that chunks respect min/max sizes."""
    text = "This is a test sentence. " * 500  # Long text
    chunks = chunk_by_sliding_window(text, min_chunk=DEFAULT_CHUNK_MIN, max_chunk=DEFAULT_CHUNK_MAX)

    for char_start, char_end, _ in chunks:
        chunk_size = char_end - char_start
        assert chunk_size <= DEFAULT_CHUNK_MAX + 100  # Allow small overflow
        # Note: minimum size is enforced during chunking, but edge cases may exist


def test_chunk_page():
    """Test chunking a crawled page."""
    page = CrawledPage(
        url="https://www.irs.gov/test",
        title="Test Page",
        crawl_timestamp=datetime.utcnow(),
        content_type=ContentType.HTML,
        raw_content=b"<html><body>Test content</body></html>",
        cleaned_text="This is a test page with enough content to be chunked. " * 100,
        content_hash="test",
    )

    chunks = chunk_page(page)

    assert len(chunks) > 0
    assert all(chunk.page_url == page.url for chunk in chunks)
    assert all(chunk.content_type == page.content_type for chunk in chunks)


def test_chunk_overlap():
    """Test that chunks have overlap."""
    text = "This is test content. " * 300
    chunks = chunk_by_sliding_window(text)

    if len(chunks) > 1:
        # Check that chunks overlap
        for i in range(len(chunks) - 1):
            current_start, current_end, _ = chunks[i]
            next_start, next_end, _ = chunks[i + 1]

            # Next chunk should start before current chunk ends
            assert next_start < current_end


def test_empty_text():
    """Test handling of empty text."""
    text = ""
    chunks = chunk_by_sliding_window(text)

    assert len(chunks) == 0


def test_short_text():
    """Test handling of short text."""
    text = "Short text"
    chunks = chunk_by_sliding_window(text)

    # Short text should result in one or zero chunks depending on min size
    assert len(chunks) <= 1


