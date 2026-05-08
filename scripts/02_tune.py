"""
02_tune.py — Step 2: UMAP/HDBSCAN parameter grid search

Usage:
    uv run python scripts/02_tune.py
    uv run python scripts/02_tune.py --config my_config.yaml
    uv run python scripts/02_tune.py --workers 4

Output: data/embeddings/<timestamp>_tuned_config.json
         reports/<timestamp>_tuning_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from itertools import product
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pipeline.config import load_config
from pipeline.embed import find_latest
from pipeline.metrics import calculate_coherence, calculate_diversity, composite_score
from pipeline.tokenize import get_tokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ── Global shared data (per-worker via initializer) ──────────────────────────
_G: dict = {}


def _worker_init(embeddings, docs, doc_word_sets, stopwords_list):
    _G["embeddings"] = embeddings
    _G["docs"] = docs
    _G["doc_word_sets"] = doc_word_sets
    _G["stopwords"] = stopwords_list


def _run_batched_trial(task: dict) -> list[dict]:
    """Run 1 UMAP → multiple HDBSCAN trials (reuses UMAP reduction)."""
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import silhouette_score

    embeddings = _G["embeddings"]
    docs = _G["docs"]
    doc_word_sets = _G["doc_word_sets"]
    stopwords = _G["stopwords"]

    umap_cfg = task["umap"]
    hdbscan_list = task["hdbscan_list"]
    target_range = task.get("target_range", (20, 50))

    results = []
    try:
        umap_model = UMAP(
            n_neighbors=umap_cfg["n_neighbors"],
            n_components=umap_cfg["n_components"],
            min_dist=umap_cfg["min_dist"],
            metric="cosine",
            random_state=42,
            n_jobs=1,
        )
        reduced = umap_model.fit_transform(embeddings)
    except Exception as e:
        print(f"[Tune] UMAP error: {e}")
        return []

    for hcfg in hdbscan_list:
        try:
            hdbscan_model = HDBSCAN(
                min_cluster_size=hcfg["min_cluster_size"],
                min_samples=hcfg["min_samples"],
                cluster_selection_method=hcfg["cluster_method"],
                metric="euclidean",
                prediction_data=True,
            )
            labels = hdbscan_model.fit_predict(reduced)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue

            noise_ratio = float((labels == -1).sum() / len(labels))
            non_noise = labels != -1
            sil = float(
                silhouette_score(reduced[non_noise], labels[non_noise])
                if non_noise.sum() > n_clusters
                else 0.0
            )

            vectorizer = CountVectorizer(
                tokenizer=lambda x: x.split(),
                token_pattern=None,
                stop_words=stopwords if stopwords else None,
                min_df=3,
                max_df=0.95,
            )
            topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer,
                embedding_model=None,
                calculate_probabilities=False,
                verbose=False,
            )
            topic_model.fit(docs, embeddings)
            topics_dict = topic_model.get_topics()

            div = calculate_diversity(topics_dict)
            coh = calculate_coherence(topics_dict, doc_word_sets)
            comp = composite_score(sil, noise_ratio, div, coh, n_clusters, target_range)

            results.append({
                **umap_cfg,
                **hcfg,
                "n_clusters": n_clusters,
                "noise_ratio": round(noise_ratio, 4),
                "silhouette": round(sil, 4),
                "diversity": round(div, 4),
                "coherence": round(coh, 4),
                "composite": round(comp, 4),
            })
        except Exception:
            continue

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UMAP/HDBSCAN parameter tuning")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, help="Number of parallel processes")
    parser.add_argument("--embed-dir", help="Override embedding directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tune_cfg = cfg["tuning"]
    tok_cfg = cfg["tokenizer"]

    if not tune_cfg.get("enabled", True):
        print("[Tune] Tuning disabled in config. Skipping.")
        return

    embed_dir = Path(args.embed_dir or tune_cfg.get("output_dir", "data/embeddings"))
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load embeddings + metadata
    embed_file = find_latest(embed_dir, "*_embeddings.npy")
    meta_file = find_latest(embed_dir, "*_metadata.csv")
    if not embed_file or not meta_file:
        raise FileNotFoundError(f"No embeddings in {embed_dir}. Run 01_embed first.")

    print(f"[Tune] Embeddings: {embed_file.name}")
    embeddings = np.load(embed_file)
    df = pd.read_csv(meta_file, encoding="utf-8-sig")

    text_col = cfg["data"].get("text_column", "text")
    for col in [text_col, "text", "abstract", "content", "본문"]:
        if col in df.columns:
            text_col = col
            break

    # 2. Tokenize documents
    print("[Tune] Tokenizing documents...")
    tokenizer = get_tokenizer(
        tokenizer_type=tok_cfg.get("type", "whitespace"),
        user_dict_path=tok_cfg.get("user_dict_path"),
        stopwords_path=tok_cfg.get("stopwords_path"),
        min_token_len=tok_cfg.get("min_token_len", 2),
    )
    raw_docs = df[text_col].fillna("").tolist()

    if hasattr(tokenizer, "tokenize_batch"):
        processed_docs = tokenizer.tokenize_batch(raw_docs)
    else:
        processed_docs = [" ".join(tokenizer.tokenize(t)) for t in tqdm(raw_docs, desc="Tokenizing")]

    doc_word_sets = [set(d.split()) for d in processed_docs if d.strip()]
    stopwords_list = list(tokenizer.stopwords) if hasattr(tokenizer, "stopwords") else []

    # 3. Build grid
    u_cfg = tune_cfg["umap"]
    h_cfg = tune_cfg["hdbscan"]
    target_range = tuple(tune_cfg.get("target_topics", [20, 50]))

    umap_combos = list(product(u_cfg["n_neighbors"], u_cfg["n_components"], u_cfg["min_dist"]))
    hdbscan_combos = list(product(
        h_cfg["min_cluster_size"],
        h_cfg["min_samples"],
        h_cfg.get("cluster_selection_method", ["eom"]),
    ))
    hdbscan_list = [
        {"min_cluster_size": h[0], "min_samples": h[1], "cluster_method": h[2]}
        for h in hdbscan_combos
    ]
    batched_tasks = [
        {
            "umap": {"n_neighbors": u[0], "n_components": u[1], "min_dist": u[2]},
            "hdbscan_list": hdbscan_list,
            "target_range": target_range,
        }
        for u in umap_combos
    ]

    n_workers = args.workers or tune_cfg.get("n_workers", 4)
    n_total = len(umap_combos) * len(hdbscan_combos)
    print(f"[Tune] Grid: {len(umap_combos)} UMAP × {len(hdbscan_combos)} HDBSCAN = {n_total} trials")
    print(f"[Tune] Workers: {n_workers}")

    # 4. Run parallel tuning
    all_results = []
    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(embeddings, processed_docs, doc_word_sets, stopwords_list),
    ) as pool:
        for batch_res in tqdm(
            pool.imap_unordered(_run_batched_trial, batched_tasks),
            total=len(batched_tasks),
            desc="Tuning",
        ):
            all_results.extend(batch_res)

    if not all_results:
        print("[Tune] All trials failed.")
        return

    # 5. Save results
    results_df = pd.DataFrame(all_results).sort_values("composite", ascending=False)
    best = results_df.iloc[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_out = {
        "timestamp": ts,
        "n_trials": len(all_results),
        "umap": {
            "n_neighbors": int(best["n_neighbors"]),
            "n_components": int(best["n_components"]),
            "min_dist": float(best["min_dist"]),
            "metric": "cosine",
        },
        "hdbscan": {
            "min_cluster_size": int(best["min_cluster_size"]),
            "min_samples": int(best["min_samples"]),
            "cluster_selection_method": best["cluster_method"],
            "metric": "euclidean",
        },
        "best_metrics": {
            "n_topics": int(best["n_clusters"]),
            "noise_ratio": float(best["noise_ratio"]),
            "silhouette": float(best["silhouette"]),
            "diversity": float(best["diversity"]),
            "coherence": float(best["coherence"]),
            "composite": float(best["composite"]),
        },
    }

    config_path = embed_dir / f"{ts}_tuned_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2, ensure_ascii=False)

    csv_path = report_dir / f"{ts}_tuning_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n[Done] Best config ({int(best['n_clusters'])} topics, composite={float(best['composite']):.4f}):")
    print(f"  UMAP:   n_neighbors={int(best['n_neighbors'])}, n_components={int(best['n_components'])}, min_dist={float(best['min_dist'])}")
    print(f"  HDBSCAN: min_cluster_size={int(best['min_cluster_size'])}, min_samples={int(best['min_samples'])}")
    print(f"\n  Config: {config_path}")
    print("Next step: uv run python scripts/03_model.py")


if __name__ == "__main__":
    main()
