"""Topic quality metrics: coherence, diversity, silhouette."""
from __future__ import annotations

import numpy as np


def calculate_diversity(topics_dict: dict, top_n: int = 10) -> float:
    """Proportion of unique words across all topic top-N word lists."""
    if not topics_dict or len(topics_dict) <= 1:
        return 0.0

    all_words: list[str] = []
    unique_words: set[str] = set()

    for topic_id, words in topics_dict.items():
        if topic_id == -1:
            continue
        top = [w[0] for w in words[:top_n]]
        all_words.extend(top)
        unique_words.update(top)

    return len(unique_words) / len(all_words) if all_words else 0.0


def calculate_coherence(
    topics_dict: dict,
    doc_word_sets: list[set[str]],
    top_n: int = 10,
) -> float:
    """PMI-based topic coherence (fast approximation).

    Parameters
    ----------
    topics_dict   : {topic_id: [(word, score), ...]} from BERTopic.get_topics()
    doc_word_sets : list of per-document word sets (pre-built for speed)
    top_n         : number of top words per topic to evaluate
    """
    if not topics_dict or not doc_word_sets:
        return 0.0

    n_docs = len(doc_word_sets)
    all_topic_words: set[str] = set()

    for topic_id, words in topics_dict.items():
        if topic_id == -1:
            continue
        for w, _ in words[:top_n]:
            all_topic_words.add(w)

    word_counts = {w: sum(1 for dw in doc_word_sets if w in dw) for w in all_topic_words}

    coherence_scores: list[float] = []
    for topic_id, words in topics_dict.items():
        if topic_id == -1:
            continue
        top_words = [w[0] for w in words[:top_n]]
        pair_scores: list[float] = []
        for i, w1 in enumerate(top_words):
            d1 = word_counts.get(w1, 0)
            if d1 == 0:
                continue
            for w2 in top_words[i + 1:]:
                d2 = word_counts.get(w2, 0)
                if d2 == 0:
                    continue
                d12 = sum(1 for dw in doc_word_sets if w1 in dw and w2 in dw)
                pmi = np.log((d12 + 1) / (d1 * d2 / n_docs + 1))
                pair_scores.append(pmi)
        if pair_scores:
            coherence_scores.append(float(np.mean(pair_scores)))

    return float(np.mean(coherence_scores)) if coherence_scores else 0.0


def composite_score(
    silhouette: float,
    noise_ratio: float,
    diversity: float,
    coherence: float,
    n_topics: int,
    target_range: tuple[int, int] = (20, 50),
) -> float:
    """Weighted composite score for parameter tuning.

    Weights: silhouette(0.25) + coverage(0.20) + diversity(0.25)
             + coherence(0.15) + topic_range(0.15)
    """
    lo, hi = target_range
    if lo <= n_topics <= hi:
        range_score = 1.0
    elif (lo * 0.5) <= n_topics <= (hi * 1.5):
        range_score = 0.7
    else:
        range_score = 0.3

    return (
        0.25 * max(silhouette, 0)
        + 0.20 * (1 - noise_ratio)
        + 0.25 * diversity
        + 0.15 * min(coherence / 5.0, 1.0)
        + 0.15 * range_score
    )
