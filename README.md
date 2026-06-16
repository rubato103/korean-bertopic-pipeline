# Korean BERTopic Pipeline

한국어 텍스트에 최적화된 BERTopic 토픽 모델링 파이프라인  
**RTX 5060 Ti 로컬 GPU · RAPIDS cuML GPU 가속 지원**

---

## 개요

뉴스, 논문, SNS 등 대규모 텍스트 데이터에서 토픽을 자동으로 추출하는 파이프라인입니다.

| 단계 | 스크립트 | 역할 |
|------|----------|------|
| 1. 임베딩 | `scripts/01_embed.py` | SBERT 벡터 생성 |
| 2. 튜닝 | `scripts/02_tune.py` | UMAP/HDBSCAN 파라미터 최적화 |
| 3. 모델링 | `scripts/03_model.py` | BERTopic 실행 |

---

## 원클릭 실행 (Docker · 권장)

임베딩(PyTorch CUDA)과 차원축소/군집(RAPIDS cuML)을 **단일 GPU 이미지**로 묶어
클론 후 한두 명령으로 전체 파이프라인이 동작합니다.

```bash
git clone https://github.com/rubato103/korean-bertopic-pipeline.git
cd korean-bertopic-pipeline

make build          # 이미지 빌드 (+ config.yaml/.env 자동 생성)
make gpu-check      # GPU + cuML 감지 확인
make pipeline       # 01 임베딩 → 02 튜닝 → 03 모델 전체 실행
```

| 명령 | 동작 |
|------|------|
| `make build` | RAPIDS 25.04 + torch(cu128) + BERTopic 단일 이미지 빌드 |
| `make embed` / `tune` / `model` | 단계별 실행 |
| `make pipeline` | 1→2→3 전체 실행 |
| `make up-bareun` | Bareun 형태소 서버 컨테이너까지 기동 후 전체 실행 |
| `make gpu-check` | GPU/cuML 가용성 점검 |
| `make shell` | 컨테이너 셸 진입 |

**전제**: NVIDIA 드라이버 + `nvidia-container-toolkit` (WSL2 지원). RTX 5060 Ti(sm_120)는
RAPIDS 25.04 / CUDA 12.8 기준입니다. `docker-compose.yml`의 버전 인자로 조정 가능합니다.

> Docker 없이 로컬(uv)로 구성하려면 `./setup.sh` — 단 cuML(GPU UMAP/HDBSCAN)은
> conda/WSL2가 필요하므로 GPU 차원축소까지 원클릭으로 원하면 Docker를 권장합니다.

---

## 빠른 시작 (수동 설치)

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/rubato103/korean-bertopic-pipeline.git
cd korean-bertopic-pipeline

# 가상환경 + 의존성 설치 (uv — uv.lock으로 버전 고정)
uv sync

# 한국어 형태소 분석기 (선택): korean=Kiwi, bareun=Bareun 클라이언트
uv sync --extra korean
uv sync --extra bareun

# PyTorch GPU 가속 (선택 — 임베딩 단계)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> 패키지 관리는 **uv** 단일 도구로 합니다. 런타임 의존성은 `[project]`,
> 개발 의존성은 `[tool.uv]`, 선택 기능은 `optional-dependencies`(korean/bareun/gpu)로
> 정의되며 `uv.lock`에 고정됩니다.

### 2. 설정

```bash
cp config.example.yaml config.yaml
# config.yaml 편집: data.input_path, embedding.model 등
```

**최소 설정**:
```yaml
data:
  input_path: "data/my_texts.csv"   # 분석할 CSV
  text_column: "text"               # 텍스트 컬럼명

embedding:
  model: "BAAI/bge-m3"             # 임베딩 모델
  batch_size: null                  # null = VRAM 자동 감지
```

### 3. 실행

```bash
# Step 1: 임베딩 생성 (~2시간 / 44K 문서, RTX 5060 Ti)
uv run python scripts/01_embed.py

# Step 2: 파라미터 튜닝 (~30분 CPU / ~5분 cuML GPU, 선택)
uv run python scripts/02_tune.py

# Step 3: BERTopic 모델링 (~10분)
uv run python scripts/03_model.py
```

---

## GPU 가속 구성

### 임베딩 (PyTorch CUDA)

SBERT 임베딩 생성 단계에서 GPU를 사용합니다. `pipeline/gpu.py`가 VRAM을 자동 감지하여 배치 크기를 조정합니다.

```
RTX 5060 Ti (16GB) 기준:
  Ollama 미실행 시 → bge-m3: batch_size ≈ 24
  Ollama gemma4:26b 실행 중 → bge-m3: batch_size ≈ 4 (자동 조정)
```

### UMAP / HDBSCAN — RAPIDS cuML (WSL2 필수)

cuML이 설치된 경우 UMAP과 HDBSCAN을 GPU에서 실행합니다. CPU 대비 속도 비교:

| 단계 | CPU (umap-learn / hdbscan) | cuML GPU |
|------|---------------------------|----------|
| UMAP (44K docs, 1024d→10d) | ~30분 | ~30초 |
| 튜닝 그리드 전체 | ~3시간 | ~5분 |
| BERTopic 전체 | ~40분 | ~8분 |

**cuML 설치 (WSL2 환경)**:
```bash
# RTX 5060 Ti (sm_120): RAPIDS 25.04+ 필요
conda install -c rapidsai cuml=25.04 python=3.12 cuda-version=12.8

# pip nightly (대안)
pip install cuml-cu12 \
  --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple
```

**자동 감지 동작 방식**: cuML이 설치되어 있으면 `import cuml` 성공 여부를 확인하여 자동 활성화합니다.

```yaml
# config.yaml
cuml:
  enabled: null   # null=자동 감지 / true=강제 활성화 / false=CPU 강제
```

```bash
# CLI로 CPU 강제 실행
uv run python scripts/02_tune.py --no-cuml
uv run python scripts/03_model.py --no-cuml
```

**cuML 튜닝 모드 특이사항**: cuML UMAP/HDBSCAN은 단일 GPU에서 내부적으로 병렬 처리되므로, cuML 감지 시 멀티프로세스 Pool 대신 단일 프로세스로 자동 전환됩니다.

---

## 임베딩 모델 선택

| 모델 | 언어 | 차원 | 속도 | 권장 환경 |
|------|------|------|------|-----------|
| `BAAI/bge-m3` | 다국어 | 1024d | 느림 | GPU 필수 |
| `jhgan/ko-sroberta-multitask` | 한국어 | 768d | 빠름 | CPU 가능 |
| `sentence-transformers/all-MiniLM-L6-v2` | 영어 | 384d | 매우 빠름 | CPU 가능 |

---

## 토크나이저 설정

```yaml
tokenizer:
  type: "bareun"        # 한국어: Bareun 형태소 분석 (별도 gRPC 서버)
  # type: "kiwi"        # 한국어: Kiwi 형태소 분석 (in-process, pip install kiwipiepy)
  # type: "whitespace"  # 범용: 공백 분리 (기본값)
  user_dict_path: "data/dictionaries/user_dict.txt"  # 사용자 사전 (kiwi)
  stopwords_path: "data/dictionaries/stopwords.txt"  # 불용어
  bareun:
    host: "bareun"      # docker-compose 서비스명 (로컬은 "localhost")
    port: 5656
    apikey: null        # null이면 BAREUN_API_KEY 환경변수 사용
    domain: null        # 사용자 사전 도메인 (선택)
```

### Bareun (한국어 형태소 — 별도 컨테이너)

Bareun은 in-process 라이브러리인 Kiwi와 달리 **gRPC 서버**로 동작합니다. 파이프라인은
`bareunpy` 클라이언트로 접속하므로 서버를 **별도 컨테이너(사이드카)** 로 띄웁니다.

```bash
# .env 에 BAREUN_API_KEY 입력 (https://bareun.ai 에서 발급)
# config.yaml: tokenizer.type: "bareun", bareun.host: "bareun"
make up-bareun        # bareun 서버 + 파이프라인 함께 실행
```

> Bareun 서버 이미지는 라이선스/레지스트리 정책에 따라 `docker-compose.yml`의
> `bareun.image` 태그를 발급받은 값으로 맞춰 주세요. API key는 `.env`(`BAREUN_API_KEY`)
> 또는 `config.yaml`의 `tokenizer.bareun.apikey`로 주입됩니다.

**Kiwi 사용자 사전 형식** (`user_dict.txt`):
```
# 한 줄에 하나씩, 형태소 태그 선택
청소년정책연구원	NNP
디지털리터러시	NNG
```

**불용어 파일 형식** (`stopwords.txt`):
```
# # 으로 시작하는 줄은 무시
것
수
등
관련
```

---

## 디렉토리 구조

```
korean-bertopic-pipeline/
├── config.example.yaml        ← 설정 템플릿
├── pyproject.toml
├── Dockerfile                 ← 단일 GPU 이미지 (임베딩 + cuML)
├── docker-compose.yml         ← pipeline + bareun 오케스트레이션
├── Makefile                   ← 원클릭 명령 (build/pipeline/up-bareun…)
├── setup.sh                   ← 비-Docker 로컬 셋업 (uv)
├── .env.example               ← BAREUN_API_KEY 등 환경변수 템플릿
├── pipeline/                  ← 재사용 가능한 핵심 모듈
│   ├── config.py              ·· YAML 설정 로더
│   ├── embed.py               ·· EmbeddingGenerator (VRAM 자동 감지)
│   ├── gpu.py                 ·· GPU/cuML 감지, 배치 크기 계산, UMAP/HDBSCAN 팩토리
│   ├── tokenize.py            ·· Bareun / Kiwi / Whitespace 토크나이저
│   └── metrics.py             ·· 토픽 품질 지표 (Coherence, Diversity)
├── scripts/                   ← CLI 실행 스크립트
│   ├── 01_embed.py            ·· 임베딩 생성
│   ├── 02_tune.py             ·· 파라미터 튜닝 (CPU 멀티프로세스 / cuML GPU 자동 선택)
│   └── 03_model.py            ·· BERTopic 실행 (CPU / cuML GPU 자동 선택)
└── data/
    └── sample/
        └── sample_texts.csv   ← 샘플 데이터 (20건)
```

---

## 파이프라인 모듈 직접 사용

```python
from pipeline import load_config, EmbeddingGenerator, get_tokenizer
from pipeline import has_cuml, make_umap, make_hdbscan

# 설정 로드
cfg = load_config("config.yaml")

# 임베딩 생성 (VRAM 자동 감지)
gen = EmbeddingGenerator(
    model_name="BAAI/bge-m3",
    batch_size=None,   # None = 자동 감지
)
embeddings = gen.encode(texts)
gen.save(embeddings, metadata_df, "data/embeddings/")

# UMAP / HDBSCAN — cuML 설치 시 자동으로 GPU 사용
print(f"cuML available: {has_cuml()}")
umap_model = make_umap(n_neighbors=15, n_components=10)
hdbscan_model = make_hdbscan(min_cluster_size=150)

# 한국어 토크나이저
# Bareun(권장): get_tokenizer("bareun", bareun={"host": "localhost", "port": 5656})
tokenizer = get_tokenizer(
    tokenizer_type="kiwi",
    user_dict_path="data/dictionaries/user_dict.txt",
    stopwords_path="data/dictionaries/stopwords.txt",
)
tokens = tokenizer.tokenize("청소년 정책 지원 방안 마련")
```

---

## 출력 파일

```
data/embeddings/
  20240801_120000_embeddings.npy       ← 임베딩 벡터 (N × D)
  20240801_120000_metadata.csv         ← 원본 데이터
  20240801_120000_embedding_info.json  ← 모델 정보 + GPU 스냅샷
  20240801_130000_tuned_config.json    ← 최적 파라미터 (cuml_used 포함)

data/model_results/
  20240801_140000_bertopic_model/      ← BERTopic 모델 파일
  20240801_140000_topic_info.csv       ← 토픽 목록 + 대표 단어
  20240801_140000_doc_topics.csv       ← 문서별 토픽 할당
  20240801_140000_run_config.json      ← 실행 설정 (재현성)
```

---

## 요구사항

- Python 3.12+ · 패키지 관리: **uv** (`uv.lock` 고정)
- GPU (임베딩): NVIDIA CUDA 12+ — bge-m3 사용 시 VRAM 8GB 이상 권장
- GPU (cuML): WSL2 + RAPIDS 25.04+ — RTX 5060 Ti(sm_120) 이상
- 한국어 형태소(선택): **Bareun**(별도 gRPC 서버, 권장) 또는 **Kiwi**(`--extra korean`)
- RAM: 16GB 이상 (튜닝 시 32GB 권장)

---

## 라이선스

MIT License
