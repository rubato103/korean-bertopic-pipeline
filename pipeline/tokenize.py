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
    """Korean morpheme tokenizer backed by the ``bareun-pipeline`` package.

    Wraps ``bareun_pipeline.BareunPipeline`` — a batched/parallel httpx client
    for a Bareun server (separate container, default port 5656). Noun
    extraction, prefix/suffix and consecutive-nominal combination are handled
    by bareun-pipeline itself; this wrapper only applies stopword / min-length
    filtering on top.

    Custom dictionaries are tuned separately via ``scripts/00_dict.py``
    (``bareun_pipeline.DictManager``) and referenced here by
    ``custom_dict_names``. Requires an API key (``apikey`` or ``BAREUN_API_KEY``).

    Reference: https://github.com/rubato103/bareun-pipeline
    Requires: pip install bareun-pipeline
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5656,
        apikey: str | None = None,
        stopwords: set[str] | None = None,
        min_len: int = 2,
        custom_dict_names: list[str] | None = None,
        batch_size: int = 50,
        max_workers: int = 8,
        combine_consecutive_nominals: bool = True,
        post_combine_pairs: set[str] | None = None,
    ):
        try:
            from bareun_pipeline import BareunPipeline
        except ImportError as e:
            raise ImportError(
                "bareun-pipeline is required for BareunTokenizer. "
                "Install with: pip install bareun-pipeline"
            ) from e

        if not apikey:
            apikey = os.environ.get("BAREUN_API_KEY")
        if not apikey:
            raise ValueError(
                "Bareun API key required. Set tokenizer.bareun.apikey in config "
                "or the BAREUN_API_KEY environment variable."
            )

        # bareun-pipeline expects a full URL host (http://host:port).
        url = host if str(host).startswith("http") else f"http://{host}:{port}"
        self.pipeline = BareunPipeline(
            host=url, api_key=apikey, batch_size=batch_size, max_workers=max_workers,
        )
        self.custom_dict_names = list(custom_dict_names) if custom_dict_names else []
        self.combine_consecutive_nominals = combine_consecutive_nominals
        self.post_combine_pairs = set(post_combine_pairs) if post_combine_pairs else set()
        self.stopwords = stopwords or set()
        self.min_len = min_len
        print(f"[Tokenizer] Bareun(pipeline) connected: {url} "
              f"| dicts={self.custom_dict_names or '-'}")

    def _filter(self, noun_list: list[str]) -> list[str]:
        return [w for w in noun_list if len(w) >= self.min_len and w not in self.stopwords]

    def _run(self, texts: list[str]):
        kwargs: dict = {
            "custom_dict_names": self.custom_dict_names,
            "combine_consecutive_nominals": self.combine_consecutive_nominals,
        }
        if self.post_combine_pairs:
            kwargs["post_combine_pairs"] = self.post_combine_pairs
        return self.pipeline.run(texts, **kwargs)

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        batch = self._run([text])
        return self._filter(batch[0].noun_list)

    def tokenize_batch(self, texts: list[str], show_progress: bool = True) -> list[str]:
        """Tokenize a list of documents (batched/parallel inside bareun-pipeline)."""
        batch = self._run(list(texts))
        return [" ".join(self._filter(nl)) for nl in batch.noun_lists()]


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
            stopwords=stopwords,
            min_len=min_token_len,
            custom_dict_names=b.get("custom_dict_names"),
            batch_size=b.get("batch_size", 50),
            max_workers=b.get("max_workers", 8),
            combine_consecutive_nominals=b.get("combine_consecutive_nominals", True),
            post_combine_pairs=b.get("post_combine_pairs"),
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
