"""Auto-tune retrieval transforms and select best config."""
import json
import subprocess
from pathlib import Path


def run_cmd(command: str) -> None:
    print(f"\n$ {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")


def read_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score(summary: dict) -> float:
    # prioritize retrieval quality: MRR primary, Hit@5 secondary
    return 0.7 * float(summary.get("mean_mrr", 0.0)) + 0.3 * float(summary.get("hit_rate_at_k", 0.0))


def main() -> None:
    configs = [
        {"name": "scaled_only", "use_pca": False, "pca_components": None},
        {"name": "pca_95", "use_pca": True, "pca_components": 0.95},
        {"name": "pca_98", "use_pca": True, "pca_components": 0.98},
        {"name": "pca_99", "use_pca": True, "pca_components": 0.99},
    ]

    results = []

    for cfg in configs:
        print("\n" + "=" * 70)
        print(f"Running config: {cfg['name']}")
        print("=" * 70)

        if cfg["use_pca"]:
            build_cmd = (
                "python scripts/build_database.py "
                "--use-pca "
                f"--pca-components {cfg['pca_components']}"
            )
        else:
            build_cmd = "python scripts/build_database.py"

        run_cmd(build_cmd)

        output_dir = f"reports/tuning/{cfg['name']}"
        eval_cmd = (
            "python scripts/evaluate_retrieval.py "
            f"--output-dir {output_dir}"
        )
        run_cmd(eval_cmd)

        summary_path = Path(output_dir) / "retrieval_summary.json"
        summary = read_summary(summary_path)
        cfg_result = {
            "config": cfg,
            "summary_path": str(summary_path),
            "hit_rate_at_5": float(summary.get("hit_rate_at_k", 0.0)),
            "mean_mrr": float(summary.get("mean_mrr", 0.0)),
            "mean_similarity_percent": float(summary.get("mean_similarity_percent", 0.0)),
            "score": score(summary),
        }
        results.append(cfg_result)

    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    best = results_sorted[0]

    out_dir = Path("reports/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = out_dir / "ranking.json"
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, indent=2, ensure_ascii=False)

    best_path = out_dir / "best_config.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    print(f"Best config: {best['config']['name']}")
    print(f"Hit@5: {best['hit_rate_at_5']:.4f}")
    print(f"MRR: {best['mean_mrr']:.4f}")
    print(f"Score: {best['score']:.4f}")
    print(f"Best summary: {best['summary_path']}")
    print(f"Ranking file: {ranking_path}")


if __name__ == "__main__":
    main()
