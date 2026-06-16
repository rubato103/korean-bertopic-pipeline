# Korean BERTopic Pipeline

한국어/다국어 텍스트를 위한 **BERTopic 토픽 모델링 파이프라인**.
config 기반 · GPU 가속(임베딩 PyTorch CUDA + 차원축소/군집 RAPIDS cuML) · uv 패키지 관리.

---

## 개요

뉴스·논문·SNS 등 대규모 텍스트에서 토픽을 자동 추출합니다. 3단계 CLI 스크립트로 구성됩니다.

| 단계 | 스크립트 | 역할 | 가속 |
|------|----------|------|------|
| 0. 사전 튜닝 (선택, 한국어) | `scripts/00_dict.py` | Bareun 사용자 사전 반복 등록·테스트 | Bareun 서버 |
| 1. 임베딩 | `scripts/01_embed.py` | **원문** SBERT 벡터 생성 | PyTorch CUDA (VRAM 자동 배치) |
| 2. 튜닝 (선택) | `scripts/02_tune.py` | UMAP/HDBSCAN 그리드 서치 | CPU 멀티프로세스 / cuML GPU |
| 3. 모델링 | `scripts/03_model.py` | BERTopic 실행 + 표현 정제 | cuML GPU / CPU 자동 |

각 단계는 타임스탬프 산출물을 남기고, 다음 단계가 디렉토리에서 최신 파일을 자동으로 찾습니다.

> **데이터 흐름**: 임베딩은 항상 **원문**으로 수행됩니다(토큰화와 독립). 형태소 분석(Bareun/Kiwi)은
> 토픽 단어(c-TF-IDF) 표현에만 쓰이며, 한국어는 0단계에서 사용자 사전을 반복 튜닝해 품질을 높입니다.
> 즉 *(사전 튜닝) → 원문 임베딩 → UMAP 차원축소 → HDBSCAN 군집 → 토픽 표현* 순서입니다.

---

## 구현 · 검증 상태

이 저장소가 실제로 무엇을 제공하고, 어디까지 검증되었는지 명확히 적습니다.

| 항목 | 상태 |
|------|------|
| 핵심 모듈(config·embed·gpu·tokenize·metrics) | ✅ 구현 + 단위 동작 검증 |
| 3단계 스크립트 / cuML 자동감지 / Bareun·Kiwi 토크나이저 | ✅ 구현 |
| uv 패키지 관리 + `uv.lock` 고정 | ✅ 구현 (`uv lock` 해석 성공, 86 패키지) |
| 원클릭 오케스트레이션 (`make init`, `docker compose config`) | ✅ 동작 검증 |
| GPU 이미지 **실빌드**(`docker compose build`) · GPU 런타임 | ⚠️ **실제 GPU 호스트에서 최초 1회 검증 필요** |
| Bareun 서버 컨테이너 이미지 태그 | ⚠️ **placeholder** — 발급받은 실제 태그로 교체 필요 |

> 즉 코드·구성·원클릭 흐름은 구현·검증되었고, GPU 이미지의 빌드/실행만 실제 하드웨어(예: RTX 5060 Ti + `nvidia-container-toolkit`)에서 첫 검증이 필요합니다.

---

## 빠른 시작 ① — Docker 원클릭 (권장)

임베딩(PyTorch CUDA)과 차원축소/군집(RAPIDS cuML)을 **단일 GPU 이미지**로 묶었습니다.
형태소 분석기 Bareun은 gRPC 서버이므로 **별도 컨테이너(사이드카)** 로 띄웁니다.

```bash
git clone https://github.com/rubato103/korean-bertopic-pipeline.git
cd korean-bertopic-pipeline

make build          # config.yaml/.env 자동 생성 후 단일 GPU 이미지 빌드
make gpu-check      # GPU + cuML 감지 확인
make pipeline       # 01 임베딩 → 02 튜닝 → 03 모델 전체 실행
```

### Make 타깃

| 명령 | 동작 |
|------|------|
| `make init` | `config.yaml`·`.env`·산출물 디렉토리 생성 (다른 타깃의 선행 조건) |
| `make build` | RAPIDS 25.04(cuML 포함) + torch(cu128) + BERTopic 단일 이미지 빌드 |
| `make embed` / `make tune` / `make model` | 단계별 실행 |
| `make pipeline` | 1 → 2 → 3 전체 실행 |
| `make up-bareun` | Bareun 서버 기동 후 전체 파이프라인 실행 (`tokenizer.type: bareun`) |
| `make gpu-check` | GPU/cuML 가용성 점검 |
| `make shell` | 컨테이너 셸 진입 |
| `make down` / `make clean` | 컨테이너 종료 / 볼륨까지 삭제 |

**전제**: NVIDIA 드라이버 + `nvidia-container-toolkit`(WSL2 지원). RTX 5060 Ti(sm_120)는
RAPIDS 25.04 / CUDA 12.8 기준이며, 버전은 `docker-compose.yml`의 build args
(`RAPIDS_VERSION`, `CUDA_VERSION`, `PYTHON_VERSION`)로 조정합니다.

> 데이터·설정·산출물은 호스트와 바인드 마운트됩니다: `./config.yaml`, `./data`, `./reports`.
> HuggingFace 모델 캐시는 named volume(`hf-cache`)에 보존됩니다.

---

## 빠른 시작 ② — 로컬 (uv, Docker 없이)

```bash
git clone https://github.com/rubato103/korean-bertopic-pipeline.git
cd korean-bertopic-pipeline

# 의존성 설치 (uv.lock으로 버전 고정)
uv sync

# 한국어 형태소 분석기 (선택): korean=Kiwi, bareun=Bareun 클라이언트
uv sync --extra korean      # Kiwi (in-process)
uv sync --extra bareun      # Bareun (gRPC 클라이언트 — 별도 서버 필요)

# PyTorch GPU 가속 (선택 — 임베딩 단계, CUDA 12.8)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 설정 파일 생성 후 실행
cp config.example.yaml config.yaml
uv run python scripts/01_embed.py
uv run python scripts/02_tune.py     # 선택
uv run python scripts/03_model.py
```

또는 한 번에: `./setup.sh` (uv sync + torch 설치 + config 생성).

> **cuML(GPU UMAP/HDBSCAN)** 은 conda/WSL2가 필요해 `uv`/pip로 간단히 설치되지 않습니다.
> 로컬에서 cuML 미설치 시 자동으로 CPU(umap-learn/hdbscan)로 폴백합니다.
> GPU 차원축소까지 원클릭으로 원하면 **Docker 경로**를 권장합니다.

---

## 패키지 관리 (uv)

단일 도구 **uv**로 관리하며, 모든 의존성은 `uv.lock`에 고정됩니다.

| 그룹 | 정의 위치 | 내용 |
|------|-----------|------|
| 런타임 | `[project].dependencies` | bertopic, sentence-transformers, umap-learn, hdbscan, pandas, numpy, scikit-learn, pyyaml, tqdm |
| 선택 | `[project.optional-dependencies]` | `korean`(kiwipiepy≥0.21) · `bareun`(bareunpy≥1.6) · `gpu`(torch≥2.7) · `cuml`(수동 설치 안내) |
| 개발 | `[tool.uv].dev-dependencies` | pytest, ruff |

```bash
uv lock        # 잠금 갱신
uv sync        # 잠금 기준 설치
```

> `cuml`·`torch`는 CUDA 휠/RAPIDS 특성상 PyPI 표준 해석으로 설치되지 않아 별도 안내(위)와 Docker로 처리합니다.
> `bareun` extra(`bareun-pipeline`)를 처음 쓸 때는 네트워크가 되는 호스트에서 `uv lock`을 한 번 실행해
> 잠금에 반영하세요.

---

## 설정 (`config.yaml`)

`config.example.yaml`을 복사해 사용합니다. 미지정 항목은 내부 기본값과 병합됩니다.

```yaml
data:
  input_path: "data/sample/sample_texts.csv"   # 분석 CSV
  text_column: "text"                            # 텍스트 컬럼
  date_column: null                              # 시계열용(선택)
  verify_column: null                            # 품질 필터(선택)

embedding:
  model: "BAAI/bge-m3"     # 임베딩 모델
  max_chars: 2500          # 임베딩 입력 절단(원문은 보존)
  batch_size: null         # null = VRAM 자동 감지
  normalize: true

tokenizer:
  type: "whitespace"       # bareun | kiwi | whitespace | none

tuning:
  enabled: true
  target_topics: [20, 50]
  n_workers: null          # null = CPU 자동 (cuML 시 1로 전환)

model:
  nr_topics: null
  min_df: 5
  max_df: 0.95
  representation: ["KeyBERT", "MMR"]

cuml:
  enabled: null            # null=자동감지 / true=강제 GPU / false=강제 CPU
```

### 스크립트 CLI 옵션

```bash
# 01_embed: 임베딩 생성
scripts/01_embed.py [--config] [--model] [--batch-size] [--max-chars] [--input] [--output-dir]

# 02_tune: 파라미터 튜닝
scripts/02_tune.py  [--config] [--workers] [--embed-dir] [--no-cuml]

# 03_model: BERTopic 실행
scripts/03_model.py [--config] [--nr-topics] [--embed-dir] [--output-dir] [--no-cuml]
```

---

## GPU 가속

### 임베딩 (PyTorch CUDA)

`pipeline/gpu.py`가 `nvidia-smi`로 VRAM을 감지해 배치 크기를 자동 조정하고,
Ollama 등 다른 프로세스의 VRAM 점유도 고려합니다. GPU가 없으면 CPU로 폴백합니다.

```
RTX 5060 Ti (16GB) 참고치:
  여유 VRAM 충분    → bge-m3 batch_size ≈ 24
  Ollama가 VRAM 점유 → bge-m3 batch_size ≈ 4 (자동 축소)
```

### UMAP / HDBSCAN — RAPIDS cuML

`import cuml` 성공 여부로 자동 활성화됩니다(설정/CLI로 강제 가능). cuML 사용 시
튜닝은 멀티프로세스 대신 단일 프로세스 GPU로 전환됩니다.

```bash
# CPU 강제 실행
uv run python scripts/02_tune.py  --no-cuml
uv run python scripts/03_model.py --no-cuml
```

```bash
# cuML 설치 (WSL2 / sm_120 → RAPIDS 25.04+)
conda install -c rapidsai cuml=25.04 python=3.12 cuda-version=12.8
# pip nightly 대안
pip install cuml-cu12 --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple
```

> 아래 속도 표는 환경 의존 **참고치**(보장값 아님). 실제 수치는 데이터·하드웨어에 따라 달라집니다.

| 단계 | CPU | cuML GPU |
|------|-----|----------|
| UMAP (44K docs, 1024d→10d) | ~30분 | ~30초 |
| 튜닝 그리드 전체 | ~3시간 | ~5분 |
| BERTopic 전체 | ~40분 | ~8분 |

---

## 토크나이저

| type | 설명 | 설치 |
|------|------|------|
| `bareun` | 한국어 형태소 — **bareun-pipeline**(배치 클라이언트) + 별도 서버 | `uv sync --extra bareun` + Bareun 서버 |
| `kiwi` | 한국어 형태소 — in-process | `uv sync --extra korean` |
| `whitespace` | 공백 분리(범용, 기본값) | 기본 포함 |
| `none` | 토크나이저 미적용 | 기본 포함 |

Kiwi는 명사(NNG/NNP/SL) 추출 + 접두/접미사 결합(예: 신+청소년+법) + 복수 `들` 제거를
in-process로 처리합니다. Bareun은 아래 `bareun-pipeline`이 동일/상위 규칙(연속 NNG/NNP 결합 포함)을
수행하며, 두 경우 모두 불용어·최소 길이 필터를 적용합니다.

### Bareun — `bareun-pipeline` 백엔드 (별도 컨테이너)

형태소 분석은 [`bareun-pipeline`](https://github.com/rubato103/bareun-pipeline)
패키지(`BareunPipeline`, 배치/병렬 httpx 클라이언트)로 수행하고, Bareun 서버는
별도 컨테이너로 띄웁니다. 사용자 사전은 `DictManager`로 **반복 튜닝**합니다.

```yaml
tokenizer:
  type: "bareun"
  bareun:
    host: "bareun"          # docker-compose 서비스명 (로컬은 "localhost")
    port: 5656
    apikey: null            # null이면 BAREUN_API_KEY 환경변수 사용
    custom_dict_names: []   # 00_dict.py로 등록·튜닝한 도메인 이름
    batch_size: 50
    max_workers: 8
    combine_consecutive_nominals: true   # 연속 NNG/NNP 결합
```

**사용자 사전 반복 튜닝** (`scripts/00_dict.py` = `DictManager` 래퍼):

```bash
# .env 에 BAREUN_API_KEY 입력 (발급: https://bareun.ai)
make dict ARGS="register --domain youth --np 청소년참여위원회 --cp 학교폭력예방"
make dict ARGS="test --domain youth --text '청소년참여위원회 회의'"   # 결과 확인 → 보완 → 재등록
make dict ARGS="list"
# 확정 후 config의 tokenizer.bareun.custom_dict_names: ["youth"] 추가

make up-bareun         # bareun 서버 + 파이프라인 전체 실행
```

> ⚠️ `docker-compose.yml`의 `bareun.image`는 **placeholder**입니다. 발급받은 실제 태그로 교체하세요.
> GPU 서버는 참조 레포 `examples/docker_gpu`(ONNX Runtime) 기준이며, **sm_120(RTX 5060 Ti)은
> TensorRT EP 권장**입니다(`BAREUN_ORT_PROVIDER`, compose에 GPU 예약 포함).

**Kiwi 사용자 사전** (`user_dict.txt`) / **불용어** (`stopwords.txt`):
```
# user_dict.txt — 한 줄에 하나, 태그 선택
청소년정책연구원	NNP
디지털리터러시	NNG

# stopwords.txt — # 주석 무시
것
등
관련
```

---

## 토픽 표현(representation)

`model.representation` 리스트가 **순서대로 체인 적용**되어 주 토픽 단어를 정제합니다.
빈 리스트면 c-TF-IDF 원형을 사용합니다.

```yaml
model:
  representation: ["KeyBERT", "MMR"]   # 관련성 → 다양성 (체인)
```

| 항목 | 역할 | 근거 |
|------|------|------|
| `KeyBERT` | 임베딩 기반 관련성 높은 키워드 | KeyBERTInspired (BERTopic) |
| `MMR` | 다양성 확보(중복 억제) 재정렬 | Carbonell & Goldstein, SIGIR 1998 |
| `LLM` | vLLM 로컬 모델로 **자연어 토픽 라벨** 생성 (선택) | TopicGPT(NAACL 2024), arXiv:2502.18469 |

### LLM 표현 — vLLM 로컬 모델 (선택, 표준 패턴)

별도 커스텀 없이 **BERTopic 기본 `OpenAI` 표현 모델을 vLLM의 OpenAI 호환
엔드포인트(`base_url`)에 연결**합니다. `representation`에 `"LLM"`을 추가할 때만
동작하며, 끄면 `openai` 패키지도 필요 없습니다.

```bash
uv sync --extra llm     # openai 클라이언트
# vLLM 서버 예: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

```yaml
model:
  representation: ["KeyBERT", "MMR", "LLM"]   # LLM은 체인 마지막 권장
  llm:
    base_url: "http://localhost:8000/v1"
    model: "Qwen/Qwen2.5-7B-Instruct"   # vLLM이 서빙 중인 모델명 (필수)
    api_key: null        # null → OPENAI_API_KEY 또는 "EMPTY"(무인증)
    temperature: 0.0
    nr_docs: 4
    prompt: null         # null → 내장 한국어 프롬프트(환각 완화 지시 포함)
```

> 내장 프롬프트는 `[KEYWORDS]`·`[DOCUMENTS]`만 근거로 라벨을 생성하도록 지시해
> **환각**을 줄입니다(참고: 토픽 입도·환각 한계, arXiv:2405.00611). 결과 라벨은 검증 권장.

---

## 임베딩 모델 선택

| 모델 | 언어 | 차원 | 속도 | 권장 환경 |
|------|------|------|------|-----------|
| `BAAI/bge-m3` | 다국어 | 1024d | 느림 | GPU 권장 |
| `jhgan/ko-sroberta-multitask` | 한국어 | 768d | 빠름 | CPU 가능 |
| `sentence-transformers/all-MiniLM-L6-v2` | 영어 | 384d | 매우 빠름 | CPU 가능 |

---

## 출력 파일

```
data/embeddings/
  <ts>_embeddings.npy        ← 임베딩 벡터 (N × D)
  <ts>_metadata.csv          ← 원본 데이터(원문 보존)
  <ts>_embedding_info.json   ← 모델 정보 + GPU 스냅샷
  <ts>_tuned_config.json     ← 최적 파라미터 (cuml_used 포함)

data/model_results/
  <ts>_bertopic_model/       ← BERTopic 모델 (safetensors)
  <ts>_topic_info.csv        ← 토픽 목록 + 대표 단어
  <ts>_doc_topics.csv        ← 문서별 토픽 할당
  <ts>_run_config.json       ← 실행 설정 (재현성)

reports/
  <ts>_tuning_results.csv    ← 튜닝 그리드 전체 결과
```

---

## 디렉토리 구조

```
korean-bertopic-pipeline/
├── config.example.yaml     ← 설정 템플릿
├── pyproject.toml · uv.lock ← uv 패키지 관리(잠금 고정)
├── Dockerfile              ← 단일 GPU 이미지 (임베딩 + cuML)
├── docker-compose.yml      ← pipeline + bareun(사이드카) 오케스트레이션
├── Makefile                ← 원클릭 명령 (build/pipeline/up-bareun…)
├── setup.sh                ← 비-Docker 로컬 셋업 (uv)
├── .env.example            ← BAREUN_API_KEY 등 환경변수 템플릿
├── pipeline/               ← 재사용 핵심 모듈
│   ├── config.py           ·· YAML 설정 로더(기본값 병합)
│   ├── embed.py            ·· EmbeddingGenerator (VRAM 자동 감지)
│   ├── gpu.py              ·· GPU/cuML 감지, 배치 계산, UMAP/HDBSCAN 팩토리
│   ├── tokenize.py         ·· Bareun / Kiwi / Whitespace 토크나이저
│   └── metrics.py          ·· 토픽 품질 지표 (Coherence, Diversity)
├── scripts/                ← CLI 실행 스크립트 (00_dict / 01 / 02 / 03)
└── data/sample/            ← 샘플 데이터 (20건)
```

---

## 모듈 직접 사용

```python
from pipeline import load_config, EmbeddingGenerator, get_tokenizer
from pipeline import has_cuml, make_umap, make_hdbscan

cfg = load_config("config.yaml")

# 임베딩 (VRAM 자동 감지)
gen = EmbeddingGenerator(model_name="BAAI/bge-m3", batch_size=None)
embeddings = gen.encode(texts)

# UMAP/HDBSCAN — cuML 설치 시 자동 GPU
print("cuML:", has_cuml())
umap_model = make_umap(n_neighbors=15, n_components=10)
hdbscan_model = make_hdbscan(min_cluster_size=150)

# 토크나이저
tok = get_tokenizer("kiwi", stopwords_path="data/dictionaries/stopwords.txt")
# Bareun: get_tokenizer("bareun", bareun={"host": "localhost", "port": 5656})
print(tok.tokenize("청소년 정책 지원 방안 마련"))
```

공개 API: `load_config`, `EmbeddingGenerator`, `get_tokenizer`,
`get_gpu_status`, `print_gpu_summary`, `has_cuml`, `make_umap`, `make_hdbscan`,
`optimal_workers`, `calculate_coherence`, `calculate_diversity`.

---

## 요구사항

- Python **3.12+** · 패키지 관리 **uv** (`uv.lock` 고정)
- GPU(임베딩): NVIDIA CUDA 12.x — bge-m3 사용 시 VRAM 8GB+ 권장 (없으면 CPU 폴백)
- GPU(cuML): WSL2 + RAPIDS 25.04+ — RTX 5060 Ti(sm_120) 이상 (Docker 경로 권장)
- 한국어 형태소(선택): **Bareun**(별도 gRPC 서버) 또는 **Kiwi**(`--extra korean`)
- RAM: 16GB+ (튜닝 시 32GB 권장)

---

## 라이선스

MIT License
