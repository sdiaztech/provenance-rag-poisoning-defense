"""Retrieval and the systems' ranking rules.

Original study: every system embeds the question and the passages with all-MiniLM-L6-v2 and ranks passages by cosine
similarity. The baseline gives the model the top 5. The provenance-aware system takes the top 10, moves verified
passages ahead of unverified ones (keeping the similarity order within each group) and gives the model the first 5.
An extra version re-ranks the top 50 instead, to show how the size of the candidate pool limits the defense.

Follow-up study: every system ranks with HybridRetriever, which fuses the embedding ranking with BM25 keyword search.
The reserved-slot defense searches the verified passages separately and always gives the model the top 3 of them,
so the number of poisoned passages in the knowledge base no longer decides whether verified passages are seen.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .corpus import Passage
from .manifest import HashManifest

EMBEDDING_MODEL = "all-minilm:l6-v2"  # all-MiniLM-L6-v2, served by Ollama
CONTEXT_SIZE = 5  # passages given to the model
CANDIDATES = 10  # passages the provenance-aware system re-ranks
WIDE_CANDIDATES = 50  # passages the extra, top-50 version re-ranks
RESERVED_SLOTS = 3  # context slots the reserved-slot defense keeps for verified passages (follow-up)
RRF_K = 60  # reciprocal rank fusion constant (Cormack et al., 2009)
BM25_K1, BM25_B = 1.2, 0.75  # standard BM25 parameters (Robertson & Zaragoza, 2009)
TOKEN = re.compile(r"[a-z0-9]+")

Embed = Callable[[Sequence[str]], Sequence[Sequence[float]]]
Ranking = Sequence[tuple[Passage, float]]  # passages with their scores, best first


class Retriever:
    """Cosine-similarity search. Vectors are cached by text, so each passage is embedded once, however many
    knowledge bases contain it."""

    def __init__(self, embed: Embed) -> None:
        self.embed = embed
        self._vectors: dict[str, np.ndarray] = {}

    def prepare(self, texts: Iterable[str]) -> int:
        """Embeds the texts not seen yet; returns how many were embedded."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._vectors))
        if missing:
            vectors = np.asarray(self.embed(missing), dtype=float)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            self._vectors.update(zip(missing, vectors))
        return len(missing)

    def rank(self, question: str, passages: Sequence[Passage]) -> Ranking:
        """All passages, most similar first (ties broken by doc_id so the order is reproducible)."""
        self.prepare([question, *(p.text for p in passages)])
        scores = np.stack([self._vectors[p.text] for p in passages]) @ self._vectors[question]
        order = sorted(range(len(passages)), key=lambda i: (-scores[i], passages[i].doc_id))
        return [(passages[i], float(scores[i])) for i in order]


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HybridRetriever(Retriever):
    """Embedding search fused with BM25 keyword search by reciprocal rank fusion (follow-up study).

    Keyword search ranks exact names and identifiers (a technique name, a CVE number) that a small embedding model
    can miss, especially in long passages that it truncates. Each passage scores 1 / (60 + its embedding rank), plus
    1 / (60 + its keyword rank) if it contains any question word. BM25 statistics come from the passages searched.
    """

    def __init__(self, embed: Embed) -> None:
        super().__init__(embed)
        self._counts: dict[str, Counter] = {}

    def dense_rank(self, question: str, passages: Sequence[Passage]) -> Ranking:
        return super().rank(question, passages)

    def keyword_scores(self, question: str, passages: Sequence[Passage]) -> list[float]:
        """BM25 score of every passage for the question's words."""
        terms = set(tokenize(question))
        docs = [self._tokens(p.text) for p in passages]
        lengths = [sum(d.values()) for d in docs]
        n, average = len(docs), sum(lengths) / max(len(docs), 1)
        idf = {}
        for t in terms:
            df = sum(1 for d in docs if t in d)
            idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        scores = []
        for d, length in zip(docs, lengths):
            norm = BM25_K1 * (1 - BM25_B + BM25_B * length / average)
            scores.append(sum(idf[t] * d[t] * (BM25_K1 + 1) / (d[t] + norm) for t in terms if t in d))
        return scores

    def rank(self, question: str, passages: Sequence[Passage]) -> Ranking:
        """All passages, best fused score first (ties broken by doc_id)."""
        dense = {p.doc_id: r for r, (p, _) in enumerate(self.dense_rank(question, passages), start=1)}
        keyword = self.keyword_scores(question, passages)
        order = sorted(range(len(passages)), key=lambda i: (-keyword[i], passages[i].doc_id))
        keyword_rank = {passages[i].doc_id: r for r, i in enumerate(order, start=1) if keyword[i] > 0}
        fused = {p.doc_id: 1 / (RRF_K + dense[p.doc_id])
                 + (1 / (RRF_K + keyword_rank[p.doc_id]) if p.doc_id in keyword_rank else 0.0) for p in passages}
        ranked = sorted(passages, key=lambda p: (-fused[p.doc_id], p.doc_id))
        return [(p, fused[p.doc_id]) for p in ranked]

    def _tokens(self, text: str) -> Counter:
        if text not in self._counts:
            self._counts[text] = Counter(tokenize(text))
        return self._counts[text]


def baseline_context(ranked: Ranking, k: int = CONTEXT_SIZE) -> list[Passage]:
    return [p for p, _ in ranked[:k]]


def provenance_context(ranked: Ranking, manifest: HashManifest, k: int = CONTEXT_SIZE,
                       candidates: int = CANDIDATES) -> list[Passage]:
    """Verified-first ranking. sorted() is stable, so similarity order is kept within each group."""
    top = [p for p, _ in ranked[:candidates]]
    return sorted(top, key=lambda p: not manifest.is_verified(p.text))[:k]


def reserved_context(ranked: Ranking, verified_ranked: Ranking,
                     k: int = CONTEXT_SIZE, reserved: int = RESERVED_SLOTS) -> list[Passage]:
    """Reserved slots (follow-up): the top `reserved` passages from a separate search of the verified passages, then
    the best-ranked remaining passages from the whole knowledge base. Unverified passages can still fill the open
    slots, so a correct passage missing from the manifest can reach the model when no attack crowds it out."""
    kept = [p for p, _ in verified_ranked[:reserved]]
    ids = {p.doc_id for p in kept}
    return kept + [p for p, _ in ranked if p.doc_id not in ids][:k - len(kept)]
