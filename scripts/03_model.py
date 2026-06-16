"""
03_model.py — Step 3: Run BERTopic model

Usage:
    uv run python scripts/03_model.py
    uv run python scripts/03_model.py --config my_config.yaml
    uv run python scripts/03_model.py --nr-topics 30

Output: data/model_results/<timestamp>_bertopic_model/
         data/model_results/<timestamp>_topic_info.csv
         data/model_results/<timestamp>_doc_topics.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from pipeline.config import load_config
from pipeline.embed import find_latest
from pipeline.gpu import has_cuml
from pipeline.tokenize import get_tokenizer


def _load_tuned_config(embed_dir: Path) -> dict | None:
    config_file = find_latest(embed_dir, "*_tuned_config.json")
    if config_file:
        print(f"[Model] Tuned config: {config_file.name}")
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Run BERTopic model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--nr-topics", type=int, help="Fix number of topics (overrides tuned config)")
    parser.add_argument("--embed-dir", help="Override embedding input directory")
    parser.add_argument("--output-dir", help="Override model output directory")
    parser.add_argument(
        "--no-cuml", action="store_true",
        help="Force CPU mode even if cuML (RAPIDS) is available",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    embed_cfg = cfg["embedding"]
    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]

    embed_dir = Path(args.embed_dir or embed_cfg.get("output_dir", "data/embeddings"))
    output_dir = Path(args.output_dir or model_cfg.get("output_dir", "data/model_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load embeddings + metadata
    embed_file = find_latest(embed_dir, "*_embeddings.npy")
    meta_file = find_latest(embed_dir, "*_metadata.csv")
    if not embed_file or not meta_file:
        raise FileNotFoundError(f"No embeddings in {embed_dir}. Run 01_embed first.")

    print(f"[Model] Loading: {embed_file.name}")
    embeddings = np.load(embed_file)
    df = pd.read_csv(meta_file, encoding="utf-8-sig")

    if len(embeddings) != len(df):
        raise ValueError(f"Shape mismatch: embeddings={len(embeddings)}, metadata={len(df)}")

    text_col = data_cfg.get("text_column", "text")
    for col in [text_col, "text", "abstract", "content", "본문"]:
        if col in df.columns:
            text_col = col
            break

    docs_raw = df[text_col].fillna("").tolist()
    print(f"[Model] Documents: {len(docs_raw):,}")

    # 2. Tokenize
    print("[Model] Tokenizing...")
    tokenizer = get_tokenizer(
        tokenizer_type=tok_cfg.get("type", "whitespace"),
        user_dict_path=tok_cfg.get("user_dict_path"),
        stopwords_path=tok_cfg.get("stopwords_path"),
        min_token_len=tok_cfg.get("min_token_len", 2),
        bareun=tok_cfg.get("bareun"),
    )

    if hasattr(tokenizer, "tokenize_batch"):
        docs_tokenized = tokenizer.tokenize_batch(docs_raw)
    else:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **k: x
        docs_tokenized = [" ".join(tokenizer.tokenize(t)) for t in tqdm(docs_raw, desc="Tokenizing")]

    # 3. Load parameters (tuned config > defaults)
    tuned = _load_tuned_config(embed_dir)
    DEFAULT_UMAP = {"n_neighbors": 15, "n_components": 10, "min_dist": 0.0, "metric": "cosine"}
    DEFAULT_HDBSCAN = {"min_cluster_size": 150, "min_samples": 15, "cluster_selection_method": "eom"}

    umap_params = tuned["umap"] if tuned else DEFAULT_UMAP
    hdbscan_params = tuned["hdbscan"] if tuned else DEFAULT_HDBSCAN

    # 4. Build BERTopic components — cuML GPU or CPU
    cuml_cfg = cfg.get("cuml", {}) or {}
    cuml_enabled_cfg = cuml_cfg.get("enabled")
    if args.no_cuml:
        use_cuml = False
    elif cuml_enabled_cfg is False:
        use_cuml = False
    elif cuml_enabled_cfg is True:
        use_cuml = True
    else:
        use_cuml = has_cuml()

    accel = "cuML/GPU" if use_cuml else "CPU"
    print(f"[Model] UMAP [{accel}]: n_neighbors={umap_params['n_neighbors']}, n_components={umap_params['n_components']}")
    print(f"[Model] HDBSCAN [{accel}]: min_cluster_size={hdbscan_params['min_cluster_size']}, min_samples={hdbscan_params.get('min_samples', 15)}")

    if use_cuml:
        import cuml  # type: ignore[import]
        cuml.set_global_output_type("numpy")
        from cuml.manifold import UMAP as _UMAP  # type: ignore[import]
        from cuml.cluster import HDBSCAN as _HDBSCAN  # type: ignore[import]
        umap_model = _UMAP(
            n_neighbors=umap_params["n_neighbors"],
            n_components=umap_params["n_components"],
            min_dist=umap_params["min_dist"],
            metric=umap_params.get("metric", "cosine"),
            random_state=42,
            output_type="numpy",
        )
        hdbscan_model = _HDBSCAN(
            min_cluster_size=hdbscan_params["min_cluster_size"],
            min_samples=hdbscan_params.get("min_samples", 15),
            cluster_selection_method=hdbscan_params.get("cluster_selection_method", "eom"),
            prediction_data=True,
        )
    else:
        from umap import UMAP as _UMAP  # type: ignore[import]
        from hdbscan import HDBSCAN as _HDBSCAN  # type: ignore[import]
        umap_model = _UMAP(
            n_neighbors=umap_params["n_neighbors"],
            n_components=umap_params["n_components"],
            min_dist=umap_params["min_dist"],
            metric=umap_params.get("metric", "cosine"),
            random_state=42,
            n_jobs=1,
        )
        hdbscan_model = _HDBSCAN(
            min_cluster_size=hdbscan_params["min_cluster_size"],
            min_samples=hdbscan_params.get("min_samples", 15),
            cluster_selection_method=hdbscan_params.get("cluster_selection_method", "eom"),
            metric="euclidean",
            prediction_data=True,
        )
    vectorizer_model = CountVectorizer(
        tokenizer=lambda x: x.split(),
        token_pattern=None,
        min_df=model_cfg.get("min_df", 5),
        max_df=model_cfg.get("max_df", 0.95),
    )

    # Representation models — a *chained list* refines the MAIN topic words
    # sequentially (config order matters: KeyBERT for relevance → MMR for
    # diversity). A dict, by contrast, only adds secondary "aspects" and leaves
    # the main representation as raw c-TF-IDF.
    repr_names = model_cfg.get("representation", ["KeyBERT", "MMR"])
    _repr_factory = {
        "KeyBERT": lambda: KeyBERTInspired(),
        "MMR": lambda: MaximalMarginalRelevance(
            diversity=model_cfg.get("mmr_diversity", 0.3)
        ),
    }
    repr_chain = [_repr_factory[n]() for n in repr_names if n in _repr_factory]
    if not repr_chain:
        representation_model = None
    elif len(repr_chain) == 1:
        representation_model = repr_chain[0]
    else:
        representation_model = repr_chain  # chained → updates Main representation
    print(f"[Model] Representation chain: {[n for n in repr_names if n in _repr_factory] or '(c-TF-IDF only)'}")

    embedding_model = SentenceTransformer(embed_cfg.get("model", "BAAI/bge-m3"))

    nr_topics = args.nr_topics or model_cfg.get("nr_topics")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )

    # 5. Progress handler
    class _ProgressHandler(logging.Handler):
        STAGES = [
            ("Dimensionality - Fitting",    10, "UMAP reducing..."),
            ("Dimensionality - Completed",  30, "UMAP done"),
            ("Cluster - Start",             35, "HDBSCAN clustering..."),
            ("Cluster - Completed",         50, "Clustering done"),
            ("Representation - Fine-tuning", 65, "Representation (slowest step)..."),
        ]

        def __init__(self):
            super().__init__()
            self.t0 = time.time()

        def emit(self, record):
            msg = record.getMessage()
            for key, pct, label in self.STAGES:
                if key in msg:
                    elapsed = time.time() - self.t0
                    m, s = divmod(int(elapsed), 60)
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    print(f"\r  [{bar}] {pct:3d}% | {label} ({m}:{s:02d})", end="", flush=True)
                    if "Completed" in key:
                        print()
                    break

    handler = _ProgressHandler()
    bert_logger = logging.getLogger("BERTopic")
    bert_logger.addHandler(handler)
    bert_logger.setLevel(logging.INFO)
    bert_logger.propagate = False

    print(f"\n[Model] Fitting BERTopic ({len(docs_tokenized):,} docs)...")
    t0 = time.time()
    topics, _ = topic_model.fit_transform(docs_tokenized, embeddings)
    elapsed = time.time() - t0

    bert_logger.removeHandler(handler)
    m, s = divmod(int(elapsed), 60)
    print(f"\r  [{'█' * 20}] 100% | Done ({m}:{s:02d})                    ")

    # 6. Save results
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    topic_info = topic_model.get_topic_info()
    n_topics = int((topic_info["Topic"] != -1).sum())
    n_outliers = int(topic_info.loc[topic_info["Topic"] == -1, "Count"].sum()) if -1 in topic_info["Topic"].values else 0

    print(f"\n[Model] Topics: {n_topics} | Outliers: {n_outliers:,} ({n_outliers/len(topics)*100:.1f}%)")
    print(topic_info.head(10).to_string())

    # Model
    if model_cfg.get("save_model", True):
        model_path = output_dir / f"{ts}_bertopic_model"
        topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
        print(f"\n[Model] Model: {model_path}")

    # Topic info CSV
    info_path = output_dir / f"{ts}_topic_info.csv"
    topic_info.to_csv(info_path, index=False, encoding="utf-8-sig")

    # Doc-topic CSV
    df["topic"] = topics
    doc_path = output_dir / f"{ts}_doc_topics.csv"
    df.to_csv(doc_path, index=False, encoding="utf-8-sig")

    # Run config JSON
    run_cfg = {
        "timestamp": ts,
        "n_documents": len(topics),
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "pct_outliers": round(n_outliers / len(topics) * 100, 1),
        "elapsed_seconds": round(elapsed, 1),
        "embedding_model": embed_cfg.get("model"),
        "tokenizer": tok_cfg.get("type"),
        "umap_params": umap_params,
        "hdbscan_params": hdbscan_params,
    }
    run_path = output_dir / f"{ts}_run_config.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print(f"[Model] Topic info: {info_path}")
    print(f"[Model] Doc-topics: {doc_path}")
    print("\n[Done] BERTopic modeling complete.")


if __name__ == "__main__":
    main()
