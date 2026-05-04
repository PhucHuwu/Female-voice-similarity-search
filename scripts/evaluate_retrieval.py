"""Evaluate retrieval performance on query_processed set."""
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.retrieval_evaluator import run_retrieval_evaluation


def evaluate(
    query_dir: str,
    metadata_db_path: str,
    top_k: int,
    output_dir: str,
) -> None:
    summary = run_retrieval_evaluation(
        query_dir=query_dir,
        metadata_db_path=metadata_db_path,
        top_k=top_k,
        output_dir=output_dir,
    )

    print("=" * 60)
    print("Retrieval Evaluation Complete")
    print("=" * 60)
    print(f"Query files: {summary['num_query_files']}")
    print(f"Top-k: {summary['top_k']}")
    print(f"Mean similarity (%): {summary['mean_similarity_percent']:.3f}")
    print(f"Hit@{top_k}: {summary['hit_rate_at_k']:.3f}")
    print(f"MRR: {summary['mean_mrr']:.3f}")
    print(f"Summary: {summary['outputs']['summary_json']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on query set")
    parser.add_argument("--query-dir", type=str, default="data/query_processed")
    parser.add_argument("--db", type=str, default="database/metadata.db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="reports/retrieval")
    args = parser.parse_args()

    evaluate(
        query_dir=args.query_dir,
        metadata_db_path=args.db,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
