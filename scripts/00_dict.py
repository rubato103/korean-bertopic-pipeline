"""
00_dict.py — Bareun 사용자 사전 반복 튜닝 (register / list / test)

bareun_pipeline.DictManager 래퍼. 형태소 분석 품질을 "등록 → 테스트 → 재등록"
루프로 개선합니다. 확정된 도메인명을 config.yaml의
tokenizer.bareun.custom_dict_names 에 넣으면 01~03 파이프라인에 반영됩니다.

참조: https://github.com/rubato103/bareun-pipeline

Usage:
    # 등록된 도메인 목록
    uv run python scripts/00_dict.py list

    # 도메인 등록 (인라인)
    uv run python scripts/00_dict.py register --domain youth \
        --np 청소년참여위원회 --cp 학교폭력예방 --cp-caret "학교^폭력"

    # 도메인 등록 (파일: 한 줄에 하나)
    uv run python scripts/00_dict.py register --domain youth \
        --np-file data/dictionaries/np.txt --cp-file data/dictionaries/cp.txt

    # 단일 문장 테스트 (사전 적용 전/후 비교)
    uv run python scripts/00_dict.py test --domain youth --text "청소년참여위원회 회의"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import load_config


def _read_lines(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _make_manager(cfg: dict):
    try:
        from bareun_pipeline import DictManager
    except ImportError as e:
        raise ImportError(
            "bareun-pipeline이 필요합니다. 설치: uv sync --extra bareun"
        ) from e

    import os

    b = (cfg.get("tokenizer", {}) or {}).get("bareun", {}) or {}
    host = b.get("host", "localhost")
    port = b.get("port", 5656)
    apikey = b.get("apikey") or os.environ.get("BAREUN_API_KEY")
    if not apikey:
        raise ValueError(
            "Bareun API key 필요: config tokenizer.bareun.apikey 또는 BAREUN_API_KEY 환경변수"
        )
    url = host if str(host).startswith("http") else f"http://{host}:{port}"
    print(f"[Dict] DictManager: {url}")
    return DictManager(host=url, api_key=apikey)


def main():
    parser = argparse.ArgumentParser(description="Bareun 사용자 사전 관리 (DictManager)")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="등록된 도메인 목록")

    p_reg = sub.add_parser("register", help="도메인 등록/갱신")
    p_reg.add_argument("--domain", required=True, help="도메인 이름")
    p_reg.add_argument("--np", action="append", default=[], help="고유명사 (반복 지정 가능)")
    p_reg.add_argument("--cp", action="append", default=[], help="복합명사 (반복 지정 가능)")
    p_reg.add_argument("--cp-caret", action="append", default=[], help="분리 지정 (예: 학교^폭력)")
    p_reg.add_argument("--np-file", help="고유명사 파일 (한 줄에 하나)")
    p_reg.add_argument("--cp-file", help="복합명사 파일 (한 줄에 하나)")
    p_reg.add_argument("--cp-caret-file", help="분리 지정 파일")

    p_test = sub.add_parser("test", help="단일 문장 테스트")
    p_test.add_argument("--domain", required=True)
    p_test.add_argument("--text", required=True)

    args = parser.parse_args()
    cfg = load_config(args.config)
    dm = _make_manager(cfg)

    if args.command == "list":
        domains = dm.list_domains()
        print(f"[Dict] 등록 도메인 ({len(domains)}):")
        for d in domains:
            print(f"  - {d}")

    elif args.command == "register":
        np_set = list(dict.fromkeys(args.np + _read_lines(args.np_file)))
        cp_set = list(dict.fromkeys(args.cp + _read_lines(args.cp_file)))
        cp_caret_set = list(dict.fromkeys(args.cp_caret + _read_lines(args.cp_caret_file)))
        print(f"[Dict] register '{args.domain}': "
              f"np={len(np_set)}, cp={len(cp_set)}, cp_caret={len(cp_caret_set)}")
        dm.register(
            domain=args.domain,
            np_set=np_set,
            cp_set=cp_set,
            cp_caret_set=cp_caret_set,
        )
        print(f"[Dict] 완료 → config tokenizer.bareun.custom_dict_names 에 "
              f"\"{args.domain}\" 추가하세요")

    elif args.command == "test":
        result = dm.test(domain=args.domain, text=args.text)
        print(f"[Dict] test '{args.domain}': {args.text}")
        print(result)


if __name__ == "__main__":
    main()
