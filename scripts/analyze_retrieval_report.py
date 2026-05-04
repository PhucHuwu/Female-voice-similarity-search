"""Generate extended retrieval metrics, tables, and charts for report."""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_query_voice(query_file: str, current_voice: str) -> str:
    if isinstance(current_voice, str) and current_voice.strip():
        return current_voice.strip()

    name = Path(query_file).name
    m_short = re.match(r"^yt_(?P<voice>.+)_[A-Za-z0-9_-]{11}_chunk\d+\.wav$", name)
    if m_short:
        return m_short.group("voice")

    m_long = re.match(r"^yt_(?P<prefix>.+)_longq_d\d+p\d+s\.wav$", name)
    if m_long:
        prefix = m_long.group("prefix")
        if len(prefix) > 12 and prefix[-12] == "_":
            return prefix[:-12]
    return ""


def hit_and_mrr_at_k(df: pd.DataFrame, k: int) -> tuple[float, float]:
    per_query = []
    for q, g in df.groupby("query_file"):
        gk = g[g["rank"] <= k].sort_values("rank")
        hit_rows = gk[gk["same_voice_hit"] == True]
        if hit_rows.empty:
            per_query.append((q, False, 0.0))
        else:
            first_rank = int(hit_rows.iloc[0]["rank"])
            per_query.append((q, True, 1.0 / first_rank))
    tmp = pd.DataFrame(per_query, columns=["query_file", "hit", "mrr"])
    return float(tmp["hit"].mean()), float(tmp["mrr"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retrieval outputs for report")
    parser.add_argument("--input-dir", type=str, default="reports/retrieval")
    parser.add_argument("--output-dir", type=str, default="reports/retrieval/analysis")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    details_path = in_dir / "retrieval_details.csv"
    if not details_path.exists():
        raise FileNotFoundError(f"Missing file: {details_path}")

    df = pd.read_csv(details_path)
    df["query_voice_fixed"] = [
        infer_query_voice(q, v if isinstance(v, str) else "")
        for q, v in zip(df["query_file"], df["query_voice"])
    ]
    df["result_voice"] = df["result_voice"].fillna("")
    df["same_voice_hit_fixed"] = df["query_voice_fixed"] == df["result_voice"]

    df["query_type"] = np.where(
        df["query_file"].str.contains(r"query_long", regex=True), "long", "short"
    )

    # Base and fixed metrics
    hit5_base, mrr_base = hit_and_mrr_at_k(df, 5)
    dff = df.copy()
    dff["same_voice_hit"] = dff["same_voice_hit_fixed"]
    hit5_fixed, mrr_fixed = hit_and_mrr_at_k(dff, 5)

    # Metrics by query type
    rows = []
    for qtype in ["short", "long"]:
        sub = dff[dff["query_type"] == qtype]
        h5, m5 = hit_and_mrr_at_k(sub, 5)
        h1, m1 = hit_and_mrr_at_k(sub, 1)
        rows.append(
            {
                "query_type": qtype,
                "num_queries": int(sub["query_file"].nunique()),
                "hit_at_1": h1,
                "mrr_at_1": m1,
                "hit_at_5": h5,
                "mrr_at_5": m5,
                "mean_similarity_percent": float(sub["similarity_percent"].mean()),
            }
        )
    by_type = pd.DataFrame(rows)
    by_type.to_csv(out_dir / "metrics_by_query_type.csv", index=False)

    # Hit@K curve
    ks = [1, 2, 3, 4, 5]
    hitk_rows = []
    for k in ks:
        h, m = hit_and_mrr_at_k(dff, k)
        hitk_rows.append({"k": k, "hit_at_k": h, "mrr_at_k": m})
    hitk_df = pd.DataFrame(hitk_rows)
    hitk_df.to_csv(out_dir / "metrics_at_k.csv", index=False)

    # Mean similarity by rank
    by_rank = (
        dff.groupby("rank", as_index=False)["similarity_percent"]
        .mean()
        .rename(columns={"similarity_percent": "mean_similarity_percent"})
        .sort_values("rank")
    )
    by_rank.to_csv(out_dir / "mean_similarity_by_rank.csv", index=False)

    # Voice-level hit rate (fixed)
    top1 = dff[dff["rank"] == 1].copy()
    voice_hit = (
        top1.groupby("query_voice_fixed", as_index=False)["same_voice_hit_fixed"]
        .mean()
        .rename(columns={"query_voice_fixed": "query_voice", "same_voice_hit_fixed": "top1_hit_rate"})
        .sort_values("top1_hit_rate", ascending=False)
    )
    voice_hit.to_csv(out_dir / "voice_top1_hit_rate_fixed.csv", index=False)

    # Charts
    plt.figure(figsize=(7, 4))
    plt.plot(hitk_df["k"], hitk_df["hit_at_k"], marker="o", label="Hit@K")
    plt.plot(hitk_df["k"], hitk_df["mrr_at_k"], marker="s", label="MRR@K")
    plt.xticks(ks)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.title("Hit@K and MRR@K (Fixed labels)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hit_mrr_at_k.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(by_rank["rank"].astype(str), by_rank["mean_similarity_percent"])
    plt.xlabel("Rank")
    plt.ylabel("Mean Similarity (%)")
    plt.title("Mean similarity by rank")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_similarity_by_rank.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(by_type["query_type"], by_type["hit_at_5"])
    plt.ylim(0, 1)
    plt.xlabel("Query type")
    plt.ylabel("Hit@5")
    plt.title("Hit@5 by query type")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "hit5_by_query_type.png", dpi=150)
    plt.close()

    # Confusion matrix (top1, fixed label)
    cm = pd.crosstab(
        top1["query_voice_fixed"].replace("", "unknown"),
        top1["result_voice"].replace("", "unknown"),
        normalize="index",
    )
    cm.to_csv(out_dir / "confusion_matrix_top1_fixed_normalized.csv")

    # Limit to top frequent voices for readability
    counts = top1["query_voice_fixed"].value_counts()
    keep = counts.index[:15]
    cm_small = cm.loc[cm.index.isin(keep), cm.columns.isin(keep)]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_small.values, aspect="auto")
    plt.colorbar(label="Rate")
    plt.xticks(range(len(cm_small.columns)), cm_small.columns, rotation=90)
    plt.yticks(range(len(cm_small.index)), cm_small.index)
    plt.title("Top-1 confusion matrix (subset, normalized)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_top1_fixed_subset.png", dpi=160)
    plt.close()

    summary = {
        "num_queries": int(dff["query_file"].nunique()),
        "num_rows": int(dff.shape[0]),
        "base_metrics": {"hit_at_5": hit5_base, "mrr": mrr_base},
        "fixed_metrics": {"hit_at_5": hit5_fixed, "mrr": mrr_fixed},
        "note": "fixed_metrics infer missing query labels for short queries from filename pattern",
        "outputs": {
            "metrics_by_query_type_csv": str(out_dir / "metrics_by_query_type.csv"),
            "metrics_at_k_csv": str(out_dir / "metrics_at_k.csv"),
            "mean_similarity_by_rank_csv": str(out_dir / "mean_similarity_by_rank.csv"),
            "voice_top1_hit_rate_fixed_csv": str(out_dir / "voice_top1_hit_rate_fixed.csv"),
            "confusion_matrix_top1_fixed_normalized_csv": str(out_dir / "confusion_matrix_top1_fixed_normalized.csv"),
            "hit_mrr_at_k_png": str(out_dir / "hit_mrr_at_k.png"),
            "mean_similarity_by_rank_png": str(out_dir / "mean_similarity_by_rank.png"),
            "hit5_by_query_type_png": str(out_dir / "hit5_by_query_type.png"),
            "confusion_matrix_top1_fixed_subset_png": str(out_dir / "confusion_matrix_top1_fixed_subset.png"),
        },
    }

    with open(out_dir / "extended_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
