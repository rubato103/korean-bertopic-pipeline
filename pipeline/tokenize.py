"""Tokenizer interface + implementations.

Supported tokenizers:
  - "whitespace"  : split on whitespace (language-agnostic)
  - "kiwi"        : Korean morpheme analysis via kiwipiepy (optional install)
  - "none"        : return text as-is (BERTopic uses its own tokenization)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


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

        tokens = self.kiwi.tokenize(text)
        nouns: list[str] = []
        i, n = 0, len(tokens)

        while i < n:
            form = tokens[i].form
            tag = tokens[i].tag

            # XPN + NNG/NNP + XSN  (e.g., 신+청소년+법)
            if tag == "XPN" and i + 2 < n:
                t1, t2 = tokens[i + 1], tokens[i + 2]
                if t1.tag in ("NNG", "NNP") and t2.tag == "XSN":
                    combined = form + t1.form if t2.form == "들" else form + t1.form + t2.form
                    nouns.append(combined)
                    i += 3
                    continue

            # XPN + NNG/NNP  (e.g., 신+청소년)
            if tag == "XPN" and i + 1 < n:
                t1 = tokens[i + 1]
                if t1.tag in ("NNG", "NNP"):
                    nouns.append(form + t1.form)
                    i += 2
                    continue

            # NNG/NNP + XSN  (e.g., 청소년+들 → 청소년)
            if tag in ("NNG", "NNP") and i + 1 < n:
                t1 = tokens[i + 1]
                if t1.tag == "XSN":
                    combined = form if t1.form == "들" else form + t1.form
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
            if len(noun) >= self.min_len and noun not in self.stopwords:
                result.append(noun)
        return result

    def tokenize_batch(self, texts: list[str], show_progress: bool = True) -> list[str]:
        """Tokenize a list of documents, returning joined token strings."""
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Tokenizing") if show_progress else texts
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
) -> Tokenizer:
    """Factory function — returns the appropriate tokenizer.

    Parameters
    ----------
    tokenizer_type : "kiwi" | "whitespace" | "none"
    user_dict_path : path to Kiwi user dictionary (kiwi only)
    stopwords_path : path to stopwords file (one word per line)
    min_token_len  : minimum token length to keep
    """
    stopwords = load_stopwords(stopwords_path)

    if tokenizer_type == "kiwi":
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
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}. Use 'kiwi', 'whitespace', or 'none'.")
