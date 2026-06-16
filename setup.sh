#!/usr/bin/env bash
# ============================================================
# 비-Docker 로컬 원클릭 셋업 (uv 기반)
#   ./setup.sh
# Docker 없이 로컬에 환경을 구성합니다.
# 단, cuML(GPU UMAP/HDBSCAN)은 conda/WSL2가 필요해 여기서 설치하지 않습니다.
# GPU 차원축소/군집까지 원클릭으로 원하면 `make build` (Docker) 사용을 권장합니다.
# ============================================================
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv가 필요합니다 → https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "[setup] 의존성 설치 (korean=Kiwi, bareun=Bareun client)..."
uv sync --extra korean --extra bareun

echo "[setup] PyTorch (CUDA 12.8 / sm_120) 설치 시도..."
if uv pip install torch --index-url https://download.pytorch.org/whl/cu128; then
  echo "[setup] torch(GPU) 설치 완료"
else
  echo "[setup] ⚠ torch GPU 설치 실패 — CPU torch는 'uv sync --extra gpu'로 설치하세요"
fi

if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
  echo "[setup] config.yaml 생성됨 (config.example.yaml 복사) — 입력 경로/모델 수정하세요"
fi
mkdir -p data/embeddings data/model_results reports

cat <<'EOF'

[setup] 완료 ✓
다음 단계:
  uv run python scripts/01_embed.py
  uv run python scripts/02_tune.py
  uv run python scripts/03_model.py

GPU 차원축소/군집(cuML)까지 원클릭 → Docker 권장:
  make build && make pipeline
EOF
