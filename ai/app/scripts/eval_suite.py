"""Evaluation suite with 20 IRS queries."""

import json
import logging
import time
from pathlib import Path

import typer

from app.core.config import settings
from app.core.logging import setup_logging
from app.generation.pipeline import RAGPipeline

setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer()

# 20 evaluation queries
EVAL_QUERIES = [
    {
        "query": "What is the 2024 standard deduction for single filers?",
        "expected_behavior": "Should return standard deduction amount with source citation",
    },
    {
        "query": "How do I get an IP PIN and where do I find it?",
        "expected_behavior": "Should explain IP PIN process with sources",
    },
    {
        "query": "Where do I mail Form 1120?",
        "expected_behavior": "Should provide mailing address with form citation",
    },
    {
        "query": "What is the penalty for late estimated tax payments?",
        "expected_behavior": "Should explain penalty with source citation",
    },
    {
        "query": "How to apply for an EIN?",
        "expected_behavior": "Should explain EIN application process",
    },
    {
        "query": "When are quarterly estimated payments due?",
        "expected_behavior": "Should provide due dates with sources",
    },
    {
        "query": "What are the limits for HSA contributions?",
        "expected_behavior": "Should provide contribution limits with sources",
    },
    {
        "query": "Instructions for Form W-9—who needs to provide it?",
        "expected_behavior": "Should explain W-9 requirements",
    },
    {
        "query": "Is Social Security taxable and how is it calculated?",
        "expected_behavior": "Should explain Social Security taxability",
    },
    {
        "query": "Child Tax Credit—eligibility rules",
        "expected_behavior": "Should explain eligibility criteria",
    },
    {
        "query": "How to request a tax transcript",
        "expected_behavior": "Should explain transcript request process",
    },
    {
        "query": "Direct Pay vs EFTPS—differences",
        "expected_behavior": "Should compare payment methods",
    },
    {
        "query": "Where to find 1099-K thresholds",
        "expected_behavior": "Should provide 1099-K threshold information",
    },
    {
        "query": "Head of Household—qualifying person rules",
        "expected_behavior": "Should explain HoH requirements",
    },
    {
        "query": "Form 4868—how to file an extension",
        "expected_behavior": "Should explain extension filing process",
    },
    {
        "query": "Mileage deduction rules for 2024",
        "expected_behavior": "Should provide mileage deduction rates",
    },
    {
        "query": "Where to report crypto sales",
        "expected_behavior": "Should explain crypto reporting requirements",
    },
    {
        "query": "What is Form 7203 used for?",
        "expected_behavior": "Should explain Form 7203 purpose",
    },
    {
        "query": "How to amend a return (1040-X)",
        "expected_behavior": "Should explain amendment process",
    },
    {
        "query": "What is the Saver's Credit and who qualifies?",
        "expected_behavior": "Should explain Saver's Credit eligibility",
    },
]


@app.command()
def main(
    output_file: str = typer.Option("eval_results.json", help="Output file for results"),
    verbose: bool = typer.Option(False, help="Verbose output"),
):
    """Run evaluation suite with 20 IRS queries."""
    logger.info("Starting evaluation suite")

    pipeline = RAGPipeline()
    results = []

    for i, eval_item in enumerate(EVAL_QUERIES, 1):
        query = eval_item["query"]
        logger.info(f"Evaluating query {i}/20: {query}")

        start_time = time.time()

        try:
            result = pipeline.answer(query)

            elapsed = time.time() - start_time

            eval_result = {
                "query": query,
                "expected_behavior": eval_item["expected_behavior"],
                "answer": result["answer_text"],
                "sources_count": len(result["sources"]),
                "confidence": result["confidence"],
                "similarity_scores": result.get("query_embedding_similarity", []),
                "latency_seconds": elapsed,
                "sources": [
                    {
                        "url": src.get("url", ""),
                        "title": src.get("title", ""),
                        "score": src.get("score", 0.0),
                    }
                    for src in result.get("sources", [])
                ],
            }

            results.append(eval_result)

            if verbose:
                print(f"\nQuery {i}: {query}")
                print(f"Answer: {result['answer_text'][:200]}...")
                print(f"Sources: {len(result['sources'])}")
                print(f"Confidence: {result['confidence']}")
                print(f"Latency: {elapsed:.2f}s\n")

        except Exception as e:
            logger.error(f"Error evaluating query {i}: {e}")
            results.append(
                {
                    "query": query,
                    "error": str(e),
                    "latency_seconds": time.time() - start_time,
                }
            )

    # Save results
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    total_queries = len(results)
    successful = sum(1 for r in results if "error" not in r)
    avg_latency = sum(r.get("latency_seconds", 0) for r in results) / total_queries if total_queries > 0 else 0
    avg_sources = sum(r.get("sources_count", 0) for r in results) / successful if successful > 0 else 0

    logger.info(f"Evaluation complete: {successful}/{total_queries} successful")
    logger.info(f"Average latency: {avg_latency:.2f}s")
    logger.info(f"Average sources per query: {avg_sources:.1f}")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    app()


