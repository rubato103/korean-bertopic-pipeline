# ============================================================
# 단일 GPU 이미지: 임베딩(PyTorch CUDA) + UMAP/HDBSCAN(RAPIDS cuML)
# ------------------------------------------------------------
# RAPIDS 베이스 이미지에 cuML이 이미 포함되어 있고, 여기에 torch(cu128)와
# BERTopic 스택을 얹어 임베딩과 차원축소/군집을 한 컨테이너에서 처리합니다.
#
# RTX 5060 Ti(sm_120) → RAPIDS 25.04 + CUDA 12.8 필요
# 실행에는 nvidia-container-toolkit + `--gpus all` (WSL2 지원)
#
# 형태소 분석기(Bareun)는 별도 서버 컨테이너로 동작 → docker-compose 참고
# ============================================================
ARG RAPIDS_VERSION=25.04
ARG CUDA_VERSION=12.8
ARG PYTHON_VERSION=3.12
FROM rapidsai/base:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py${PYTHON_VERSION}

USER root
WORKDIR /workspace

# 1) PyTorch (CUDA 12.8 / sm_120) — 임베딩 GPU 가속
#    먼저 설치해 BERTopic/sentence-transformers가 별도 torch를 끌어오지 않게 함
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu128

# 2) BERTopic 스택 + Bareun 클라이언트
#    numpy/pandas/scikit-learn은 RAPIDS 베이스에 포함되어 있어 재설치하지 않음
#    (cuML과의 버전 충돌 방지). bertopic이 umap-learn/hdbscan(CPU 폴백)을 끌어옴.
RUN pip install --no-cache-dir \
        "bertopic>=0.16" \
        "sentence-transformers>=3.0" \
        "pyyaml>=6.0" \
        "tqdm>=4.65" \
        "bareun-pipeline>=0.1"

# 3) 프로젝트 코드 복사
COPY . .

# HuggingFace 캐시 위치 (compose에서 볼륨으로 마운트)
ENV HF_HOME=/workspace/.cache/huggingface

# 기본은 셸 — 실제 단계는 compose / Makefile에서 지정
CMD ["bash"]
