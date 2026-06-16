"""Configuration loader — reads config.yaml and provides typed access."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULTS: dict[str, Any] = {
    "data": {
        "input_path": "data/sample/sample_texts.csv",
        "text_column": "text",
        "id_column": None,
        "date_column": None,
        "verify_column": None,
        "verify_values": None,
    },
    "embedding": {
        "model": "BAAI/bge-m3",
        "max_chars": 2500,
        "batch_size": 4,
        "normalize": True,
        "output_dir": "data/embeddings",
    },
    "tokenizer": {
        "type": "whitespace",
        "user_dict_path": None,
        "stopwords_path": None,
        "min_token_len": 2,
        "bareun": {
            "host": "localhost",
            "port": 5656,
            "apikey": None,  # None → BAREUN_API_KEY env var
            "custom_dict_names": [],          # DictManager로 튜닝한 도메인명
            "batch_size": 50,                 # bareun-pipeline 배치 크기
            "max_workers": 8,                 # 병렬 요청 수
            "combine_consecutive_nominals": True,  # 연속 NNG/NNP 결합
        },
    },
    "tuning": {
        "enabled": True,
        "target_topics": [20, 50],
        "n_workers": 4,
        "output_dir": "data/embeddings",
        "umap": {
            "n_neighbors": [10, 15, 20],
            "n_components": [5, 10, 15],
            "min_dist": [0.0, 0.05],
        },
        "hdbscan": {
            "min_cluster_size": [100, 150, 200, 250, 300],
            "min_samples": [10, 15, 20],
            "cluster_selection_method": ["eom"],
        },
    },
    "model": {
        "nr_topics": None,
        "min_df": 5,
        "max_df": 0.95,
        "representation": ["KeyBERT", "MMR"],
        "mmr_diversity": 0.3,
        "output_dir": "data/model_results",
        "save_model": True,
    },
    "cuml": {
        "enabled": None,  # None=auto-detect, True=force GPU, False=force CPU
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, merging with defaults.

    If path is None, look for config.yaml in the current directory.
    """
    cfg = _DEFAULTS.copy()

    if path is None:
        candidates = ["config.yaml", "config.yml"]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break

    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)

    return cfg


def resolve_path(cfg: dict, key: str, base_dir: Path | None = None) -> Path | None:
    """Resolve a config path relative to base_dir (project root)."""
    raw = cfg.get(key)
    if raw is None:
        return None
    p = Path(raw)
    if base_dir and not p.is_absolute():
        return base_dir / p
    return p
