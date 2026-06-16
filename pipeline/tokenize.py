"""Tokenizer interface + implementations.

Supported tokenizers:
  - "whitespace"  : split on whitespace (language-agnostic)
  - "kiwi"        : Korean morpheme analysis via kiwipiepy (optional install)
  - "none"        : return text as-is (BERTopic uses its own tokenization)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


def _extract_nouns(
    pairs: list[tuple[str, str]],
    stopwords: set[str],
    min_len: int,
) -> list[str]:
    """Extract nouns from (morpheme, POS-tag) pairs.

    Shared by Kiwi and Bareun — both emit the standard Sejong tagset
    (NNG/NNP nouns, SL foreign, XPN prefix, XSN noun-deriving suffix).
    Handles XPN + noun + XSN combination (e.g. 신+청소년+법) and strips
    the plural suffix 들.
    """
    nouns: list[str] = []
    i, n = 0, len(pairs)

    while i < n:
        form, tag = pairs[i]

        # XPN + NNG/NNP + XSN  (e.g., 신+청소년+법)
        if tag == "XPN" and i + 2 < n:
            f1, t1 = pairs[i + 1]
            f2, t2 = pairs[i + 2]
            if t1 in ("NNG", "NNP") and t2 == "XSN":
                combined = form + f1 if f2 == "들" else form + f1 + f2
                nouns.append(combined)
                i += 3
                continue

        # XPN + NNG/NNP  (e.g., 신+청소년)
        if tag == "XPN" and i + 1 < n:
            f1, t1 = pairs[i + 1]
            if t1 in ("NNG", "NNP"):
                nouns.append(form + f1)
                i += 2
                continue

        # NNG/NNP + XSN  (e.g., 청소년+들 → 청소년)
        if tag in ("NNG", "NNP") and i + 1 < n:
            f1, t1 = pairs[i + 1]
            if t1 == "XSN":
                combined = form if f1 == "들" else form + f1
                nouns.append(combined)
                i += 2
                continue

        if tag in ("NNG", "NNP", "SL"):
            nouns.append(form)
        elif tag == "XPN":
            nouns.append(form)
        elif tag == "XSN" and form != "들":
            nouns.append(form)

        i += 1

    result: list[str] = []
    for noun in nouns:
        noun = re.sub(r"들$", "", noun).replace(" ", "")
        if len(noun) >= min_len and noun not in stopwords:
            result.append(noun)
    return result


class WhitespaceTokenizer:
    """Simple whitespace tokenizer — language-agnostic fallback."""

    def __init__(self, stopwords: set[str] | None = None, min_len: int = 2):
        self.stopwords = stopwords or set()
        self.min_len = min_len

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = text.split()
        return [t for t in tokens if len(t) >= self.min_len and t not in self.stopwords]


class KiwiTokenizer:
    """Korean morpheme tokenizer using kiwipiepy.

    Extracts nouns (NNG, NNP), handles XPN prefix + XSN suffix combination.
    Requires: pip install kiwipiepy
    """

    def __init__(
        self,
        user_dict_path: str | Path | None = None,
        stopwords: set[str] | None = None,
        min_len: int = 2,
        model_type: str = "cong-global",
    ):
        try:
            from kiwipiepy import Kiwi
        except ImportError as e:
            raise ImportError(
                "kiwipiepy is required for KiwiTokenizer. "
                "Install with: pip install kiwipiepy"
            ) from e

        self.kiwi = Kiwi(model_type=model_type)
        self.stopwords = stopwords or set()
        self.min_len = min_len

        if user_dict_path and Path(user_dict_path).exists():
            self.kiwi.load_user_dictionary(str(user_dict_path))
            print(f"[Tokenizer] User dict loaded: {Path(user_dict_path).name}")

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        pairs = [(t.form, t.tag) for t in self.kiwi.tokenize(text)]
        return _extract_nouns(pairs, self.stopwords, self.min_len)

    def tokenize_batch(self, texts: list[str], show_progress: bool = True) -> list[str]:
        """Tokenize a list of documents, returning joined token strings."""
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Tokenizing") if show_progress else texts
        except ImportError:
            iterator = texts
        return [" ".join(self.tokenize(t)) for t in iterator]


class BareunTokenizer:
    """Korean morpheme tokenizer using the Bareun gRPC server (bareunpy client).

    Unlike Kiwi (in-process), Bareun runs as a **separate server** (default
    port 5656). This client connects over gRPC, so the server must be reachable
    — see the ``bareun`` service in docker-compose.yml. Requires an API key
    from https://bareun.ai (set ``apikey`` or the ``BAREUN_API_KEY`` env var).

    Requires: pip install bareunpy
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5656,
        apikey: str | None = None,
        stopwords: set[str] | None = None,
        min_len: int = 2,
        domain: str | None = None,
    ):
        try:
            from bareunpy import Tagger
        except ImportError as e:
            raise ImportError(
                "bareunpy is required for BareunTokenizer. "
                "Install with: pip install bareunpy"
            ) from e

        if not apikey:
            apikey = os.environ.get("BAREUN_API_KEY")
        if not apikey:
            raise ValueError(
                "Bareun API key required. Set tokenizer.bareun.apikey in config "
                "or the BAREUN_API_KEY environment variable."
            )

        self.tagger = Tagger(apikey, host, port)
        if domain:
            # Custom user-dictionary domain (optional, best-effort).
            try:
                self.tagger.set_domain(domain)
            except Exception:
                print(f"[Tokenizer] ⚠  Bareun domain '{domain}' not set (ignored)")
        self.stopwords = stopwords or set()
        self.min_len = min_len
        print(f"[Tokenizer] Bareun connected: {host}:{port}")

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        # bareunpy Tagger.pos(text) → list[(morpheme, POS-tag)]
        pairs = [(m, t) for m, t in self.tagger.pos(text)]
        return _extract_nouns(pairs, self.stopwords, self.min_len)

    def tokenize_batch(self, texts: list[str], show_progress: bool = True) -> list[str]:
        """Tokenize a list of documents, returning joined token strings."""
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Tokenizing(Bareun)") if show_progress else texts
        except ImportError:
            iterator = texts
        return [" ".join(self.tokenize(t)) for t in iterator]


def load_stopwords(path: str | Path | None) -> set[str]:
    """Load stopwords from a text file (one word per line, # comments ignored)."""
    if path is None or not Path(path).exists():
        return set()
    words: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line)
    print(f"[Tokenizer] Stopwords loaded: {len(words)} from {Path(path).name}")
    return words


def get_tokenizer(
    tokenizer_type: str = "whitespace",
    user_dict_path: str | Path | None = None,
    stopwords_path: str | Path | None = None,
    min_token_len: int = 2,
    bareun: dict | None = None,
) -> Tokenizer:
    """Factory function — returns the appropriate tokenizer.

    Parameters
    ----------
    tokenizer_type : "bareun" | "kiwi" | "whitespace" | "none"
    user_dict_path : path to Kiwi user dictionary (kiwi only)
    stopwords_path : path to stopwords file (one word per line)
    min_token_len  : minimum token length to keep
    bareun         : Bareun server settings dict (host/port/apikey/domain)
    """
    stopwords = load_stopwords(stopwords_path)

    if tokenizer_type == "bareun":
        b = bareun or {}
        return BareunTokenizer(
            host=b.get("host", "localhost"),
            port=b.get("port", 5656),
            apikey=b.get("apikey"),
            domain=b.get("domain"),
            stopwords=stopwords,
            min_len=min_token_len,
        )
    elif tokenizer_type == "kiwi":
        return KiwiTokenizer(
            user_dict_path=user_dict_path,
            stopwords=stopwords,
            min_len=min_token_len,
        )
    elif tokenizer_type == "whitespace":
        return WhitespaceTokenizer(stopwords=stopwords, min_len=min_token_len)
    elif tokenizer_type == "none":
        return WhitespaceTokenizer(min_len=0)  # passthrough
    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type!r}. "
            "Use 'bareun', 'kiwi', 'whitespace', or 'none'."
        )
