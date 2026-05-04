"""Retrieval evaluation utilities for query dataset."""
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.search.similarity_search import VoiceSimilaritySearch
from src.vector_database.metadata_db import parse_processed_filename


def extract_query_voice_from_name(file_path: str) -> str:
    """Extract voice key from query filename for both short and long queries."""
    parsed = parse_processed_filename(file_path)
    voice = parsed.get("voice")
    if voice:
        return voice

    stem = Path(file_path).stem
    marker = "_longq_d"
    if stem.startswith("yt_") and marker in stem:
        prefix = stem[3:stem.index(marker)]
        if len(prefix) > 12 and prefix[-12] == "_":
            return prefix[:-12]
    return ""


def run_retrieval_evaluation(
    query_dir: str = "data/query_short,data/query_long",
    metadata_db_path: str = "database/metadata.db",
    scaler_path: str = "database/scaler.pkl",
    pca_path: str = "database/pca.pkl",
    top_k: int = 5,
    output_dir: str = "reports/retrieval",
) -> Dict:
    """Run retrieval evaluation and save report files."""
    query_dirs = [q.strip() for q in query_dir.split(",") if q.strip()]
    query_files = []
    for qd in query_dirs:
        qp = Path(qd)
        query_files.extend(sorted(qp.glob("*.wav")))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not query_files:
        raise FileNotFoundError(f"No query files found in: {query_dirs}")

    search_system = VoiceSimilaritySearch(
        metadata_db_path=metadata_db_path,
        scaler_path=scaler_path,
        pca_path=pca_path,
    )

    rows = []
    all_scores = []
    rank_scores = {r: [] for r in range(1, top_k + 1)}

    for qf in query_files:
        query_voice = extract_query_voice_from_name(str(qf))
        results = search_system.search_similar(str(qf), top_k=top_k, preprocess=True)

        for rank, (file_path, sim_percent, cosine) in enumerate(results, start=1):
            all_scores.append(float(sim_percent))
            rank_scores[rank].append(float(sim_percent))

            meta = search_system.get_metadata(file_path) or {}
            candidate_voice = meta.get("voice_name")
            hit = bool(query_voice and candidate_voice and query_voice == candidate_voice)

            rows.append(
                {
                    "query_file": str(qf),
                    "query_voice": query_voice,
                    "rank": rank,
                    "result_file": file_path,
                    "result_voice": candidate_voice,
                    "similarity_percent": float(sim_percent),
                    "cosine_similarity": float(cosine),
                    "same_voice_hit": hit,
                }
            )

    details_df = pd.DataFrame(rows)
    details_csv = out_dir / "retrieval_details.csv"
    details_df.to_csv(details_csv, index=False)

    per_query = []
    for q, g in details_df.groupby("query_file", dropna=False):
        g = g.sort_values("rank")
        q_voice = g.iloc[0]["query_voice"] if not g.empty else None
        hit_rows = g[g["same_voice_hit"] == True]
        if hit_rows.empty:
            per_query.append({"query_file": q, "query_voice": q_voice, "hit_at_k": False, "mrr": 0.0})
        else:
            first_rank = int(hit_rows.iloc[0]["rank"])
            per_query.append(
                {
                    "query_file": q,
                    "query_voice": q_voice,
                    "hit_at_k": True,
                    "mrr": 1.0 / first_rank,
                }
            )

    per_query_df = pd.DataFrame(per_query)
    per_query_csv = out_dir / "retrieval_per_query.csv"
    per_query_df.to_csv(per_query_csv, index=False)

    by_voice = (
        per_query_df.groupby("query_voice", dropna=False)["hit_at_k"]
        .mean()
        .reset_index()
        .rename(columns={"hit_at_k": "voice_hit_rate_at_k"})
        .sort_values("voice_hit_rate_at_k", ascending=False)
    )
    by_voice_csv = out_dir / "retrieval_hit_rate_by_voice.csv"
    by_voice.to_csv(by_voice_csv, index=False)

    # Confusion matrix (query voice vs predicted top-1 voice)
    top1_df = details_df[details_df["rank"] == 1].copy()
    top1_df["query_voice"] = top1_df["query_voice"].fillna("unknown")
    top1_df["result_voice"] = top1_df["result_voice"].fillna("unknown")

    confusion_counts = pd.crosstab(
        top1_df["query_voice"],
        top1_df["result_voice"],
        rownames=["query_voice"],
        colnames=["predicted_top1_voice"],
    )
    confusion_counts_csv = out_dir / "confusion_matrix_counts.csv"
    confusion_counts.to_csv(confusion_counts_csv)

    confusion_normalized = pd.crosstab(
        top1_df["query_voice"],
        top1_df["result_voice"],
        rownames=["query_voice"],
        colnames=["predicted_top1_voice"],
        normalize="index",
    )
    confusion_normalized_csv = out_dir / "confusion_matrix_normalized.csv"
    confusion_normalized.to_csv(confusion_normalized_csv)

    summary = {
        "query_dirs": query_dirs,
        "num_query_files": len(query_files),
        "top_k": top_k,
        "mean_similarity_percent": float(np.mean(all_scores)) if all_scores else 0.0,
        "std_similarity_percent": float(np.std(all_scores)) if all_scores else 0.0,
        "min_similarity_percent": float(np.min(all_scores)) if all_scores else 0.0,
        "max_similarity_percent": float(np.max(all_scores)) if all_scores else 0.0,
        "p25_similarity_percent": float(np.percentile(all_scores, 25)) if all_scores else 0.0,
        "p50_similarity_percent": float(np.percentile(all_scores, 50)) if all_scores else 0.0,
        "p75_similarity_percent": float(np.percentile(all_scores, 75)) if all_scores else 0.0,
        "hit_rate_at_k": float(per_query_df["hit_at_k"].mean()) if not per_query_df.empty else 0.0,
        "mean_mrr": float(per_query_df["mrr"].mean()) if not per_query_df.empty else 0.0,
        "mean_similarity_by_rank": {
            f"rank_{rank}": (float(np.mean(vals)) if vals else 0.0)
            for rank, vals in rank_scores.items()
        },
        "outputs": {
            "details_csv": str(details_csv),
            "per_query_csv": str(per_query_csv),
            "hit_rate_by_voice_csv": str(by_voice_csv),
            "confusion_counts_csv": str(confusion_counts_csv),
            "confusion_normalized_csv": str(confusion_normalized_csv),
        },
    }

    summary_json = out_dir / "retrieval_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary["outputs"]["summary_json"] = str(summary_json)
    return summary
