"""RAG pipeline orchestration."""

import logging
import time
import hashlib
from typing import Any, Optional
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.constants import NO_KB_MSG
from app.core.prompts import build_no_results_prompt, build_rag_prompt
from app.core.security import should_add_disclaimer
from app.core.utils import estimate_tokens
from app.generation.llm import get_llm_provider
from app.vector.embeddings import get_embedding_provider
from app.vector.qdrant_client import get_client
from app.vector.reranker import get_reranker
from app.vector.optimized_reranker import get_parallel_reranker
from app.vector.retriever import retrieve_with_cutoff

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for question answering with optimizations."""

    def __init__(self, use_parallel_reranker: bool = True, enable_caching: bool = True):
        """Initialize RAG pipeline with optimizations."""
        logger.info("Initializing RAG Pipeline with optimizations...")
        
        self.embedding_provider = get_embedding_provider()
        self.llm_provider = get_llm_provider()
        self.qdrant_client = get_client()
        self.collection_name = settings.collection_name
        self.enable_caching = enable_caching
        
        # Initialize reranker with parallel processing
        original_reranker = get_reranker()
        if use_parallel_reranker and original_reranker:
            self.reranker = get_parallel_reranker(
                original_reranker, 
                batch_size=8,
                max_workers=3
            )
            logger.info("✅ Using PARALLEL reranker")
        else:
            self.reranker = original_reranker
            logger.info("Using standard reranker")
        
        # Setup caching
        if self.enable_caching:
            self._response_cache = {}
            self._cache_ttl = 3600
            logger.info("✅ Response caching ENABLED (TTL: 1 hour)")
        
        # Warm up models (runs in background on first load)
        self._warmup_models()

    def _warmup_models(self):
        """Pre-warm models to avoid cold start on first query."""
        logger.info("🔥 Warming up models...")
        start = time.time()
        
        try:
            # Warm up embedding model
            _ = self.embedding_provider.get_embedding("test")
            logger.info(f"   ✅ Embedding model ready ({time.time() - start:.2f}s)")
            
            # Warm up LLM with minimal generation
            llm_start = time.time()
            self.llm_provider.generate(
                prompt="Say: OK",
                system_prompt="Respond briefly.",
                max_tokens=3
            )
            logger.info(f"   ✅ LLM ready ({time.time() - llm_start:.2f}s)")
            
            logger.info(f"🔥 Warmup complete: {time.time() - start:.2f}s total")
        except Exception as e:
            logger.warning(f"⚠️  Warmup failed: {e}")

    @lru_cache(maxsize=256)
    def _get_embedding_cached(self, query: str) -> tuple:
        """Get embedding with LRU cache."""
        embedding = self.embedding_provider.get_embedding(query)
        return tuple(embedding)

    def _get_cache_key(self, query: str, filters: Optional[dict] = None) -> str:
        """Generate cache key."""
        filter_str = str(sorted(filters.items())) if filters else ""
        return hashlib.md5(f"{query}|{filter_str}".encode()).hexdigest()

    def answer(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        cutoff: Optional[float] = None,
    ) -> dict[str, Any]:
        """Answer a query using optimized RAG pipeline."""
        total_start = time.time()
        logger.info(f"========== PIPELINE START: {query[:50]}... ==========")
        
        try:
            # Check response cache
            if self.enable_caching:
                cache_key = self._get_cache_key(query, filters)
                if cache_key in self._response_cache:
                    cached_response, cached_time = self._response_cache[cache_key]
                    if time.time() - cached_time < self._cache_ttl:
                        elapsed = time.time() - total_start
                        logger.info(f"💾 CACHE HIT! Returned in {elapsed:.3f}s")
                        logger.info(f"========== PIPELINE END (CACHED) ==========")
                        return cached_response
            
            # Step 1: Embed query with caching
            step_start = time.time()
            if self.enable_caching:
                query_embedding_tuple = self._get_embedding_cached(query)
                query_embedding = list(query_embedding_tuple)
            else:
                query_embedding = self.embedding_provider.get_embedding(query)
            embedding_time = time.time() - step_start
            logger.info(f"⏱️  [EMBEDDING] {embedding_time:.3f}s")

            # Step 2: Retrieve chunks
            step_start = time.time()
            top_k = top_k or settings.top_k
            top_n = top_n or settings.top_n
            cutoff = cutoff or settings.similarity_cutoff

            chunks = retrieve_with_cutoff(
                self.qdrant_client,
                self.collection_name,
                query_embedding,
                top_k=top_k,
                cutoff=cutoff,
                filters=filters,
            )
            retrieval_time = time.time() - step_start
            logger.info(f"⏱️  [RETRIEVAL] {retrieval_time:.3f}s - Found {len(chunks)} chunks")

            if not chunks:
                total_time = time.time() - total_start
                logger.warning(f"No chunks found")
                logger.info(f"========== PIPELINE END (NO RESULTS): {total_time:.3f}s ==========")
                return {
                    "answer_text": NO_KB_MSG,
                    "sources": [],
                    "confidence": "low",
                    "query_embedding_similarity": [],
                }

            # Step 3: Reranking (skip if < 10 chunks)
            step_start = time.time()
            original_chunk_count = len(chunks)
            if self.reranker and len(chunks) >= 10:
                chunks = self.reranker.rerank(query, chunks, top_n=top_n)
                reranking_time = time.time() - step_start
                logger.info(f"⏱️  [RERANKING] {reranking_time:.3f}s - {original_chunk_count} → {len(chunks)} chunks")  # Fixed            else:
                chunks = chunks[:top_n]
                reranking_time = 0

            # Step 4: Build prompt
            step_start = time.time()
            prompt = build_rag_prompt(chunks, query)
            prompt_time = time.time() - step_start
            prompt_tokens = estimate_tokens(prompt)
            logger.info(f"⏱️  [PROMPT BUILD] {prompt_time:.3f}s - {prompt_tokens} tokens")

            # Step 5: Generate answer
            step_start = time.time()
            system_prompt = "You are a factual assistant that answers only from the provided IRS.gov knowledge snippets."
            answer_text = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=200,  # Optimized
            )
            generation_time = time.time() - step_start
            response_tokens = len(answer_text.split())
            logger.info(f"⏱️  [LLM GENERATION] {generation_time:.3f}s - Generated ~{response_tokens} tokens")

            # Step 6: Post-processing
            step_start = time.time()
            if should_add_disclaimer(query):
                answer_text = f"{settings.legal_disclaimer}\n\n{answer_text}"
            
            sources = []
            similarities = []
            for chunk in chunks:
                sources.append({
                    "url": chunk.get("url", ""),
                    "title": chunk.get("title", ""),
                    "section": chunk.get("section_heading"),
                    "snippet": chunk.get("text", "")[:300],
                    "char_start": chunk.get("char_start", 0),
                    "char_end": chunk.get("char_end", 0),
                    "score": chunk.get("score", 0.0),
                })
                similarities.append(chunk.get("score", 0.0))

            avg_similarity = np.mean(similarities) if similarities else 0.0
            confidence = "high" if avg_similarity >= 0.8 else "medium" if avg_similarity >= 0.5 else "low"
            
            post_processing_time = time.time() - step_start
            logger.info(f"⏱️  [POST-PROCESSING] {post_processing_time:.3f}s")

            result = {
                "answer_text": answer_text,
                "sources": sources,
                "confidence": confidence,
                "query_embedding_similarity": similarities,
            }
            
            # Cache response
            if self.enable_caching:
                cache_key = self._get_cache_key(query, filters)
                if len(self._response_cache) >= 100:
                    oldest_key = next(iter(self._response_cache))
                    del self._response_cache[oldest_key]
                self._response_cache[cache_key] = (result, time.time())

            # Timing summary
            total_time = time.time() - total_start
            logger.info(f"📊 TIMING BREAKDOWN:")
            logger.info(f"   Embedding:       {embedding_time:6.3f}s ({embedding_time/total_time*100:5.1f}%)")
            logger.info(f"   Retrieval:       {retrieval_time:6.3f}s ({retrieval_time/total_time*100:5.1f}%)")
            logger.info(f"   Reranking:       {reranking_time:6.3f}s ({reranking_time/total_time*100:5.1f}%)")
            logger.info(f"   Prompt Build:    {prompt_time:6.3f}s ({prompt_time/total_time*100:5.1f}%)")
            logger.info(f"   LLM Generation:  {generation_time:6.3f}s ({generation_time/total_time*100:5.1f}%)")
            logger.info(f"   Post-processing: {post_processing_time:6.3f}s ({post_processing_time/total_time*100:5.1f}%)")
            logger.info(f"   ─────────────────────────────────────")
            logger.info(f"   TOTAL:           {total_time:6.3f}s (100.0%)")
            logger.info(f"========== PIPELINE END ==========")

            return result

        except Exception as e:
            total_time = time.time() - total_start
            logger.error(f"Error after {total_time:.3f}s: {e}", exc_info=True)
            logger.info(f"========== PIPELINE END (ERROR) ==========")
            return {
                "answer_text": NO_KB_MSG,
                "sources": [],
                "confidence": "low",
                "query_embedding_similarity": [],
            }


def get_rag_pipeline() -> RAGPipeline:
    """Get optimized RAG pipeline instance."""
    return RAGPipeline(use_parallel_reranker=True, enable_caching=True)
##