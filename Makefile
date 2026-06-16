# ============================================================
# Korean BERTopic Pipeline — 원클릭 명령
#   make build            이미지 빌드 (config.yaml 자동 생성)
#   make pipeline         01 임베딩 → 02 튜닝 → 03 모델 전체 실행
#   make embed/tune/model 단계별 실행
#   make up-bareun        Bareun 형태소 서버까지 기동
#   make gpu-check        GPU/cuML 감지 확인
#   make shell            컨테이너 셸 진입
# ============================================================
.PHONY: init build pipeline embed tune model shell gpu-check up-bareun down clean

DC ?= docker compose
RUN = $(DC) run --rm pipeline

# config.yaml + 산출물 디렉토리 준비 (compose 바인드 마운트 전 필수)
init:
	@test -f config.yaml || (cp config.example.yaml config.yaml && echo "[init] config.yaml 생성됨 (config.example.yaml 복사)")
	@mkdir -p data/embeddings data/model_results reports
	@test -f .env || (cp .env.example .env && echo "[init] .env 생성됨 — Bareun 사용 시 BAREUN_API_KEY 입력")

build: init
	$(DC) build

embed: init
	$(RUN) python scripts/01_embed.py

tune: init
	$(RUN) python scripts/02_tune.py

model: init
	$(RUN) python scripts/03_model.py

pipeline: embed tune model

# Bareun 서버를 띄운 뒤 전체 파이프라인 실행 (tokenizer.type: "bareun")
up-bareun: init
	$(DC) --profile bareun up -d bareun
	$(DC) --profile bareun run --rm pipeline python scripts/01_embed.py
	$(DC) --profile bareun run --rm pipeline python scripts/02_tune.py
	$(DC) --profile bareun run --rm pipeline python scripts/03_model.py

gpu-check: init
	$(RUN) python -c "from pipeline.gpu import get_gpu_status, has_cuml, print_gpu_summary; print_gpu_summary(get_gpu_status()); print('cuML available:', has_cuml())"

shell: init
	$(RUN) bash

down:
	$(DC) --profile bareun down

clean:
	$(DC) --profile bareun down -v
