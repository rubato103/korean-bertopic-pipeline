"""
02_reduce_cluster.py — Step 2: 차원축소(UMAP) · 군집(HDBSCAN) 파라미터 그리드 서치

BERTopic 표준 단계의 '차원축소 + 군집'에 대한 파라미터 튜닝(그리드 서치).
최적 UMAP/HDBSCAN 파라미터를 찾아 03_bertopic 단계에서 사용합니다.

Usage:
    uv run python scripts/02_reduce_cluster.py
    uv run python scripts/02_reduce_cluster.py --config my_config.yaml
    uv run python scripts/02_reduce_cluster.py --workers 4
    uv run python scripts/02_reduce_cluster.py --no-cuml   # force CPU even if cuML available

Output: data/embeddings/<timestamp>_tuned_config.json
         reports/<timestamp>_tuning_results.csv
"""
from __future__ import annotations

import argparse
import json
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
from pipeline.gpu import has_cuml, optimal_workers
from pipeline.metrics import calculate_coherence, calculate_diversity, composite_score
from pipeline.tokenize import get_tokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ── Global shared data (per-worker via initializer) ──────────────────────────
_G: dict = {}


def _worker_init(embeddings, docs, doc_word_sets, stopwords_list, use_cuml=False,
                 min_df=5, max_df=0.95):
    _G["embeddings"] = embeddings
    _G["docs"] = docs
    _G["doc_word_sets"] = doc_word_sets
    _G["stopwords"] = stopwords_list
    _G["use_cuml"] = use_cuml
    _G["min_df"] = min_df
    _G["max_df"] = max_df


def _run_batched_trial(task: dict) -> list[dict]:
    """Run 1 UMAP → multiple HDBSCAN trials (reuses UMAP reduction).

    Uses cuML (GPU) when _G["use_cuml"] is True, else umap-learn/hdbscan (CPU).
    """
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import silhouette_score

    use_cuml = _G.get("use_cuml", False)

    if use_cuml:
        import cuml  # type: ignore[import]
        cuml.set_global_output_type("numpy")
        from cuml.manifold import UMAP   # type: ignore[import]
        from cuml.cluster import HDBSCAN  # type: ignore[import]
        umap_extra = {"output_type": "numpy"}
        hdbscan_extra: dict = {}
    else:
        from umap import UMAP   # type: ignore[import]
        from hdbscan import HDBSCAN  # type: ignore[import]
        umap_extra = {"n_jobs": 1}
        hdbscan_extra = {"metric": "euclidean"}

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
            **umap_extra,
        )
        reduced = umap_model.fit_transform(embeddings)
        if not isinstance(reduced, np.ndarray):
            reduced = np.asarray(reduced)
    except Exception as e:
        print(f"[Tune] UMAP error: {e}")
        return []

    for hcfg in hdbscan_list:
        try:
            hdbscan_model = HDBSCAN(
                min_cluster_size=hcfg["min_cluster_size"],
                min_samples=hcfg["min_samples"],
                cluster_selection_method=hcfg["cluster_method"],
                prediction_data=True,
                **hdbscan_extra,
            )
            labels = hdbscan_model.fit_predict(reduced)
            if not isinstance(labels, np.ndarray):
                labels = np.asarray(labels)

            n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue

            noise_ratio = float((labels == -1).sum() / len(labels))
            non_noise = labels != -1
            sil = float(
                silhouette_score(reduced[non_noise], labels[non_noise])
                if non_noise.sum() > n_clusters
                else 0.0
            )

            # Docs are already tokenized + stopword-filtered by the tokenizer,
            # so we mirror 03_bertopic's vectorizer (same min_df/max_df, no
            # double stop_words) to keep tuning scores consistent with the
            # final model.
            vectorizer = CountVectorizer(
                tokenizer=lambda x: x.split(),
                token_pattern=None,
                min_df=_G.get("min_df", 5),
                max_df=_G.get("max_df", 0.95),
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
    parser.add_argument("--workers", type=int, help="Number of parallel processes (CPU mode only)")
    parser.add_argument("--embed-dir", help="Override embedding directory")
    parser.add_argument(
        "--no-cuml", action="store_true",
        help="Force CPU mode even if cuML (RAPIDS) is available",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tune_cfg = cfg["tuning"]
    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]
    min_df = model_cfg.get("min_df", 5)
    max_df = model_cfg.get("max_df", 0.95)

    if not tune_cfg.get("enabled", True):
        print("[Tune] Tuning disabled in config. Skipping.")
        return

    # cuML detection — config can override auto-detect, --no-cuml always wins
    cuml_cfg = cfg.get("cuml", {}) or {}
    cuml_enabled_cfg = cuml_cfg.get("enabled")  # None=auto, True=force on, False=force off
    if args.no_cuml:
        use_cuml = False
    elif cuml_enabled_cfg is False:
        use_cuml = False
    elif cuml_enabled_cfg is True:
        use_cuml = True
    else:
        use_cuml = has_cuml()

    if use_cuml:
        print("[Tune] cuML (RAPIDS GPU) 감지 — 단일 프로세스 GPU 실행")
        print("[Tune]   튜닝 속도: CPU 다중 프로세스 대비 약 10-20배 빠름")
        n_workers = 1
    else:
        cfg_workers = tune_cfg.get("n_workers")
        n_workers = args.workers or cfg_workers or optimal_workers()

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
        bareun=tok_cfg.get("bareun"),
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

    n_total = len(umap_combos) * len(hdbscan_combos)
    print(f"[Tune] Grid: {len(umap_combos)} UMAP × {len(hdbscan_combos)} HDBSCAN = {n_total} trials")

    # 4. Run tuning — sequential GPU (cuML) or parallel CPU
    all_results: list[dict] = []

    if use_cuml:
        print(f"[Tune] Mode: cuML GPU (single process)")
        _worker_init(embeddings, processed_docs, doc_word_sets, stopwords_list,
                     use_cuml=True, min_df=min_df, max_df=max_df)
        for task in tqdm(batched_tasks, desc="Tuning (cuML/GPU)"):
            all_results.extend(_run_batched_trial(task))
    else:
        print(f"[Tune] Mode: CPU (workers={n_workers}, RAM ~{n_workers * 2.5:.0f} GB)")
        with Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(embeddings, processed_docs, doc_word_sets, stopwords_list,
                      False, min_df, max_df),
        ) as pool:
            for batch_res in tqdm(
                pool.imap_unordered(_run_batched_trial, batched_tasks),
                total=len(batched_tasks),
                desc="Tuning (CPU)",
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
        "cuml_used": use_cuml,
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

    accel = "cuML/GPU" if use_cuml else f"CPU×{n_workers}"
    print(f"\n[Done] Best config ({int(best['n_clusters'])} topics, composite={float(best['composite']):.4f}) [{accel}]:")
    print(f"  UMAP:    n_neighbors={int(best['n_neighbors'])}, n_components={int(best['n_components'])}, min_dist={float(best['min_dist'])}")
    print(f"  HDBSCAN: min_cluster_size={int(best['min_cluster_size'])}, min_samples={int(best['min_samples'])}")
    print(f"\n  Config: {config_path}")
    print("Next step: uv run python scripts/03_bertopic.py")


if __name__ == "__main__":
    main()
