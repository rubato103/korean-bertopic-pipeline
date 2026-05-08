"""Embedding generation utilities — wraps sentence-transformers."""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate and save SBERT embeddings for a document corpus.

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Recommended options:
        - "BAAI/bge-m3"                      (SOTA multilingual, 1024d, GPU recommended)
        - "jhgan/ko-sroberta-multitask"       (lightweight Korean, 768d, CPU-friendly)
    max_chars:
        Truncate documents to this many characters before encoding.
        bge-m3 ≈ 2500 chars (8192 tokens), ko-sroberta ≈ 1000 chars (512 tokens).
    batch_size:
        Sentences per batch. Lower for large GPU models to avoid OOM.
    normalize:
        L2-normalize embeddings (recommended for cosine similarity).
    device:
        "cuda", "cpu", or None (auto-detect).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        max_chars: int = 2500,
        batch_size: int = 4,
        normalize: bool = True,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_chars = max_chars
        self.batch_size = batch_size
        self.normalize = normalize

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        print(f"[Embed] Model: {model_name}")
        print(f"[Embed] Device: {device} | batch_size: {batch_size} | max_chars: {max_chars}")
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts → (N, D) float32 array."""
        truncated = [t[:self.max_chars] if isinstance(t, str) else "" for t in texts]
        t0 = time.time()
        embeddings = self.model.encode(
            truncated,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        elapsed = time.time() - t0
        n = len(texts)
        print(f"[Embed] Done: {n:,} docs in {elapsed:.1f}s ({n/elapsed:.0f} docs/s)")
        print(f"[Embed] Shape: {embeddings.shape}")
        return embeddings

    def save(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        output_dir: str | Path,
        timestamp: str | None = None,
    ) -> dict[str, Path]:
        """Save embeddings (.npy) + metadata (.csv) + info (.json)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = timestamp or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        paths: dict[str, Path] = {}

        embed_path = output_dir / f"{ts}_embeddings.npy"
        np.save(embed_path, embeddings)
        paths["embeddings"] = embed_path

        meta_path = output_dir / f"{ts}_metadata.csv"
        metadata.to_csv(meta_path, index=False, encoding="utf-8-sig")
        paths["metadata"] = meta_path

        info = {
            "timestamp": ts,
            "model_name": self.model_name,
            "embedding_dim": int(embeddings.shape[1]),
            "n_documents": int(embeddings.shape[0]),
            "max_chars": self.max_chars,
            "batch_size": self.batch_size,
            "device": self.device,
            "normalize_embeddings": self.normalize,
        }
        info_path = output_dir / f"{ts}_embedding_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        paths["info"] = info_path

        for name, p in paths.items():
            size_mb = p.stat().st_size / 1024**2
            print(f"[Embed] Saved {name}: {p.name} ({size_mb:.1f} MB)")

        return paths


def find_latest(directory: str | Path, pattern: str) -> Path | None:
    """Return the most-recently-modified file matching pattern, or None."""
    matches = list(Path(directory).glob(pattern))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def load_embeddings(embed_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load the latest embeddings + metadata from a directory."""
    embed_dir = Path(embed_dir)
    embed_file = find_latest(embed_dir, "*_embeddings.npy")
    meta_file = find_latest(embed_dir, "*_metadata.csv")

    if embed_file is None or meta_file is None:
        raise FileNotFoundError(
            f"No embedding files found in {embed_dir}. Run 01_embed first."
        )

    print(f"[Embed] Loading: {embed_file.name}")
    embeddings = np.load(embed_file)
    metadata = pd.read_csv(meta_file, encoding="utf-8-sig")

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Shape mismatch: embeddings={len(embeddings)}, metadata={len(metadata)}"
        )

    print(f"[Embed] Loaded: {embeddings.shape}, {len(metadata):,} docs")
    return embeddings, metadata
