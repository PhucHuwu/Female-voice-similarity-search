"""Evaluate retrieval performance on short+long query sets."""
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
    compare_to: str = None,
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

    if compare_to:
        import json
        from pathlib import Path as _Path

        cmp_path = _Path(compare_to)
        if cmp_path.exists():
            with open(cmp_path, "r", encoding="utf-8") as f:
                old = json.load(f)

            delta_hit = summary["hit_rate_at_k"] - old.get("hit_rate_at_k", 0.0)
            delta_mrr = summary["mean_mrr"] - old.get("mean_mrr", 0.0)
            delta_sim = summary["mean_similarity_percent"] - old.get("mean_similarity_percent", 0.0)

            print("-" * 60)
            print("Comparison vs baseline")
            print("-" * 60)
            print(f"Baseline: {compare_to}")
            print(f"Delta Hit@{top_k}: {delta_hit:+.4f}")
            print(f"Delta MRR: {delta_mrr:+.4f}")
            print(f"Delta Mean Similarity (%): {delta_sim:+.4f}")
        else:
            print(f"Compare file not found: {compare_to}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on query set")
    parser.add_argument("--query-dir", type=str, default="data/query_short,data/query_long")
    parser.add_argument("--db", type=str, default="database/metadata.db")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="reports/retrieval")
    parser.add_argument("--compare-to", type=str, default=None, help="Path to baseline retrieval_summary.json")
    args = parser.parse_args()

    evaluate(
        query_dir=args.query_dir,
        metadata_db_path=args.db,
        top_k=args.top_k,
        output_dir=args.output_dir,
        compare_to=args.compare_to,
    )


if __name__ == "__main__":
    main()
