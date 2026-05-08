# Korean BERTopic Pipeline

한국어 텍스트에 최적화된 BERTopic 토픽 모델링 파이프라인  
**RTX 5060 Ti 로컬 GPU + Google Colab 호환**

---

## 개요

뉴스, 논문, SNS 등 대규모 텍스트 데이터에서 토픽을 자동으로 추출하는 파이프라인입니다.

| 단계 | 스크립트 / 노트북 | 역할 |
|------|------------------|------|
| 1. 임베딩 | `scripts/01_embed.py` · `notebooks/01_embed.ipynb` | SBERT 벡터 생성 |
| 2. 튜닝 | `scripts/02_tune.py` | UMAP/HDBSCAN 파라미터 최적화 |
| 3. 모델링 | `scripts/03_model.py` · `notebooks/02_bertopic.ipynb` | BERTopic 실행 |

---

## 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/korean-bertopic-pipeline.git
cd korean-bertopic-pipeline

# 가상환경 + 의존성 설치 (uv 권장)
uv sync

# 한국어 형태소 분석기 (선택)
uv sync --extra korean

# GPU 가속 (선택)
uv sync --extra gpu
```

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
  batch_size: 4                     # GPU 메모리에 맞게 조정
```

### 3. 실행

```bash
# Step 1: 임베딩 생성 (~2시간 / 44K 문서, RTX 5060 Ti)
uv run python scripts/01_embed.py

# Step 2: 파라미터 튜닝 (~30분, 선택)
uv run python scripts/02_tune.py

# Step 3: BERTopic 모델링 (~10분)
uv run python scripts/03_model.py
```

또는 **Jupyter 노트북**으로:
```bash
jupyter lab notebooks/
```

---

## 임베딩 모델 선택

| 모델 | 언어 | 차원 | 속도 | 권장 환경 |
|------|------|------|------|-----------|
| `BAAI/bge-m3` | 다국어 | 1024d | 느림 | GPU 필수 |
| `jhgan/ko-sroberta-multitask` | 한국어 | 768d | 빠름 | CPU 가능 |
| `all-MiniLM-L6-v2` | 영어 | 384d | 매우 빠름 | CPU 가능 |

---

## 토크나이저 설정

```yaml
tokenizer:
  type: "kiwi"          # 한국어: 형태소 분석 (pip install kiwipiepy)
  # type: "whitespace"  # 범용: 공백 분리 (기본값)
  user_dict_path: "data/dictionaries/user_dict.txt"  # 사용자 사전
  stopwords_path: "data/dictionaries/stopwords.txt"  # 불용어
```

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
├── pipeline/                  ← 재사용 가능한 핵심 모듈
│   ├── config.py              ·· YAML 설정 로더
│   ├── embed.py               ·· EmbeddingGenerator 클래스
│   ├── tokenize.py            ·· Kiwi / Whitespace 토크나이저
│   └── metrics.py             ·· 토픽 품질 지표 (Coherence, Diversity)
├── scripts/                   ← CLI 실행 스크립트
│   ├── 01_embed.py
│   ├── 02_tune.py
│   └── 03_model.py
├── notebooks/                 ← Jupyter 인터랙티브 워크플로
│   ├── 01_embed.ipynb         ·· Colab/로컬 GPU 임베딩
│   └── 02_bertopic.ipynb      ·· 튜닝 + 모델링
└── data/
    └── sample/
        └── sample_texts.csv   ← 샘플 데이터 (20건)
```

---

## 파이프라인 모듈 직접 사용

```python
from pipeline import load_config, EmbeddingGenerator, get_tokenizer

# 설정 로드
cfg = load_config("config.yaml")

# 임베딩 생성
gen = EmbeddingGenerator(
    model_name="BAAI/bge-m3",
    max_chars=2500,
    batch_size=4,
)
embeddings = gen.encode(texts)
gen.save(embeddings, metadata_df, "data/embeddings/")

# 한국어 토크나이저
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
  20240801_120000_metadata.csv         ← 원본 데이터 + text_for_embed 컬럼
  20240801_120000_embedding_info.json  ← 모델 정보
  20240801_130000_tuned_config.json    ← 최적 파라미터 (튜닝 완료 시)

data/model_results/
  20240801_140000_bertopic_model/      ← BERTopic 모델 파일
  20240801_140000_topic_info.csv       ← 토픽 목록 + 대표 단어
  20240801_140000_doc_topics.csv       ← 문서별 토픽 할당
  20240801_140000_run_config.json      ← 실행 설정 (재현성)
```

---

## Google Colab 사용

```python
# Colab 셀에서 실행
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = '/content/drive/MyDrive/korean-bertopic-pipeline'

import sys
sys.path.insert(0, PROJECT_DIR)

from pipeline import EmbeddingGenerator
# ... 이후 동일하게 사용
```

---

## 요구사항

- Python 3.12+
- GPU: NVIDIA (CUDA 12+) — bge-m3 사용 시 VRAM 8GB 이상 권장
- RAM: 16GB 이상 (튜닝 시 32GB 권장)

---

## 라이선스

MIT License
