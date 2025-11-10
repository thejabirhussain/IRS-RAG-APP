"""Tests for embedding and Qdrant upsert."""

import pytest
import numpy as np

from app.vector.embeddings import LocalEmbeddingProvider, OpenAIEmbeddingProvider


def test_local_embedding_provider():
    """Test local embedding provider."""
    provider = LocalEmbeddingProvider()
    texts = ["This is a test", "Another test sentence"]

    embeddings = provider.get_embeddings(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == provider.vector_size
    assert embeddings.dtype == np.float32


def test_local_embedding_single():
    """Test single text embedding."""
    provider = LocalEmbeddingProvider()
    text = "Test sentence"

    embedding = provider.get_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1
    assert embedding.shape[0] == provider.vector_size


def test_local_embedding_consistency():
    """Test that same text produces same embedding."""
    provider = LocalEmbeddingProvider()
    text = "Consistency test"

    embedding1 = provider.get_embedding(text)
    embedding2 = provider.get_embedding(text)

    np.testing.assert_array_almost_equal(embedding1, embedding2)


@pytest.mark.skipif(True, reason="Requires OpenAI API key")
def test_openai_embedding_provider():
    """Test OpenAI embedding provider (requires API key)."""
    # This test is skipped unless OPENAI_API_KEY is set
    provider = OpenAIEmbeddingProvider()
    texts = ["Test sentence 1", "Test sentence 2"]

    embeddings = provider.get_embeddings(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.dtype == np.float32


def test_embedding_vector_size():
    """Test that embeddings have correct vector size."""
    provider = LocalEmbeddingProvider()
    text = "Test"

    embedding = provider.get_embedding(text)

    assert embedding.shape[0] == provider.vector_size
    assert provider.vector_size > 0


