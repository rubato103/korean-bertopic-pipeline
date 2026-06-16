"""
01_embed.py — Step 1: 임베딩 (원문 SBERT 벡터 생성)

Usage:
    uv run python scripts/01_embed.py
    uv run python scripts/01_embed.py --config my_config.yaml
    uv run python scripts/01_embed.py --model BAAI/bge-m3 --batch-size 4

Output: data/embeddings/<timestamp>_embeddings.npy
         data/embeddings/<timestamp>_metadata.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from pipeline.config import load_config
from pipeline.embed import EmbeddingGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate SBERT embeddings")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", help="Override embedding model name")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--max-chars", type=int, help="Override max chars per document")
    parser.add_argument("--input", help="Override input CSV path")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    embed_cfg = cfg["embedding"]

    # CLI overrides
    if args.model:
        embed_cfg["model"] = args.model
    if args.batch_size:
        embed_cfg["batch_size"] = args.batch_size
    if args.max_chars:
        embed_cfg["max_chars"] = args.max_chars
    if args.input:
        data_cfg["input_path"] = args.input
    if args.output_dir:
        embed_cfg["output_dir"] = args.output_dir

    # 1. Load data
    input_path = Path(data_cfg["input_path"])
    print(f"[Step 1] Loading data: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"  Rows: {len(df):,} | Columns: {list(df.columns)}")

    # 2. Filter (optional verify column)
    if data_cfg.get("verify_column") and data_cfg["verify_column"] in df.columns:
        before = len(df)
        valid = data_cfg.get("verify_values") or []
        df = df[df[data_cfg["verify_column"]].isin(valid)].copy()
        print(f"  Filtered ({data_cfg['verify_column']}): {before:,} → {len(df):,}")

    # 3. Extract text
    text_col = data_cfg.get("text_column", "text")
    for col in [text_col, "text", "abstract", "content", "본문"]:
        if col in df.columns:
            text_col = col
            break
    else:
        raise ValueError(f"Text column not found. Available: {list(df.columns)}")

    before = len(df)
    df = df[df[text_col].notna() & (df[text_col].str.len() > 0)].copy()
    if before != len(df):
        print(f"  Removed empty texts: {before:,} → {len(df):,}")

    # Keep original text for BERTopic c-TF-IDF, truncate only for embedding
    max_chars = embed_cfg.get("max_chars", 2500)
    df["text_for_embed"] = df[text_col].str.slice(0, max_chars)

    # 4. Generate embeddings
    gen = EmbeddingGenerator(
        model_name=embed_cfg.get("model", "BAAI/bge-m3"),
        max_chars=max_chars,
        batch_size=embed_cfg.get("batch_size", 4),
        normalize=embed_cfg.get("normalize", True),
    )
    docs = df["text_for_embed"].tolist()
    embeddings = gen.encode(docs)

    # 5. Save
    output_dir = embed_cfg.get("output_dir", "data/embeddings")
    paths = gen.save(embeddings, df, output_dir)

    print(f"\n[Done] Embeddings saved to: {paths['embeddings']}")
    print("Next step: uv run python scripts/02_reduce_cluster.py  (or skip to 03_bertopic.py)")


if __name__ == "__main__":
    main()
