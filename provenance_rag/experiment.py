"""Runs every system on every question in every knowledge base, plus a closed-book run, and records the answers.

Each trial is one question answered by one system in one condition: S0, S1 or S2 with 1, 5 or 10 poisoned passages
per question, or closed-book (no passages at all, to show what the model knows from memory).

Attacks are targeted by default: each question is answered from the legitimate passages plus its own poisoned
passages. With attack="all-at-once", every question's poisoned passages share one knowledge base (a stress test).
Which systems run in each study is set in studies.py.

Tiered answering asks the model twice at most: first with the top 5 verified passages alone, allowing it to reply NOT
FOUND, and only then with the regular top 5, verified or not; an answer from that fallback is flagged as unverified.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from .corpus import Passage
from .manifest import HashManifest
from .prompts import VERIFIED_ONLY_INSTRUCTION, build_closed_book_prompt, build_prompt
from .questions import Question, QuestionSet
from .retrieval import (CANDIDATES, CONTEXT_SIZE, RESERVED_SLOTS, WIDE_CANDIDATES, HybridRetriever, Ranking,
                        Retriever, baseline_context, provenance_context, reserved_context)
from .scenarios import KnowledgeBase, is_poison_for
from .scoring import not_found, score

BASELINE = "baseline"
PROVENANCE = "provenance-aware"  # verified passages first among the top 10 (the original design)
PROVENANCE_WIDE = "provenance-aware, top 50"  # the same among the top 50 (original study, extra analysis)
RESERVED = "provenance-aware, reserved slots"  # 3 of the 5 slots kept for verified passages (the follow-up's design)
TIERED = "tiered answering"  # verified passages alone first; the regular top 5 only if they lack the answer
CLOSED_BOOK = "closed-book"  # no passages: what the model answers from memory
# The reserved-slot defense with 1 to 5 of the 5 slots kept for verified passages (follow-up trade-off analysis)
SLOT_SYSTEMS = {k: RESERVED if k == RESERVED_SLOTS else f"reserved, {k} verified slot{'s' if k > 1 else ''}"
                for k in range(1, CONTEXT_SIZE + 1)}
RESERVED_K = {name: k for k, name in SLOT_SYSTEMS.items()}  # verified slots of each reserved-slot system
TRADEOFF_SYSTEMS = tuple(name for name in RESERVED_K if name != RESERVED)
# Top passages each system's diagnostics describe (and that the provenance-aware systems re-rank)
POOL = {BASELINE: CANDIDATES, PROVENANCE: CANDIDATES, PROVENANCE_WIDE: WIDE_CANDIDATES, TIERED: CANDIDATES,
        **dict.fromkeys(RESERVED_K, CANDIDATES)}
VERIFIED_SEARCH = {*RESERVED_K, TIERED}  # systems that also search the verified passages on their own
ATTACKS = ("targeted", "all-at-once")


class ReplyCache:
    """Model replies keyed by model and prompt, saved as they arrive. A re-run reuses them, and an interrupted run
    resumes where it stopped. Any change to a prompt or the model gives a new key, so stale replies are never reused.
    Prompts themselves are not saved, because they contain poisoned passages."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._replies: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._replies[record["key"]] = record["reply"]

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()[:24]

    def get(self, model: str, prompt: str) -> str | None:
        return self._replies.get(self.key(model, prompt))

    def put(self, model: str, prompt: str, reply: str) -> None:
        key = self.key(model, prompt)
        self._replies[key] = reply
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": model, "reply": reply}) + "\n")


def _shown(passages: list[Passage], q: Question) -> dict:
    """Which passages the model is shown, how many are poisoned, and whether the answer passage is among them."""
    return {"context": " ".join(p.doc_id for p in passages),
            "poisoned_in_context": sum(p.label == "poisoned" for p in passages),
            "gold_in_context": any(p.doc_id == q.gold_doc for p in passages)}


def rank_of(doc_id: str, ranked: Ranking) -> int | None:
    """The passage's position in a ranking (1 = first), or None if it is not there."""
    ids = [p.doc_id for p, _ in ranked]
    return ids.index(doc_id) + 1 if doc_id in ids else None


class Experiment:
    def __init__(self, question_set: QuestionSet, bases: list[KnowledgeBase], manifest: HashManifest,
                 retriever: Retriever, generate: Callable[[str], str], model: str, cache: ReplyCache, *,
                 systems: Iterable[str], attack: str = "targeted") -> None:
        if attack not in ATTACKS:
            raise ValueError(f"unknown attack {attack!r}; expected one of {ATTACKS}")
        self.question_set, self.bases, self.manifest = question_set, bases, manifest
        self.retriever, self.generate, self.model, self.cache = retriever, generate, model, cache
        self.systems, self.attack = tuple(systems), attack
        self.new_calls = 0

    def plan(self, splits: Iterable[str]) -> list[dict]:
        """Every trial with its prompt, before any model call."""
        splits = set(splits)
        questions = [q for q in self.question_set.questions if q.split in splits]
        self.retriever.prepare([q.text for q in questions] + [p.text for kb in self.bases for p in kb.passages])
        trials = []
        for q in questions:
            about = {"split": q.split, "qid": q.qid, "kind": q.kind, "gold_verified": q.gold_verified,
                     "reference": q.reference, "target": q.target, "attack": self.attack}
            trials.append({**about, "condition": CLOSED_BOOK, "poisons_per_question": 0, "system": CLOSED_BOOK,
                           "context": "", "poisoned_in_context": 0, "gold_in_context": False,
                           "prompt": build_closed_book_prompt(q.text)})
            for kb in self.bases:
                trials += self._trials(q, kb, {**about, "condition": kb.condition,
                                               "poisons_per_question": kb.n_poisons})
        return trials

    def _trials(self, q: Question, kb: KnowledgeBase, about: dict) -> list[dict]:
        """One trial per system for this question in this knowledge base."""
        passages = kb.passages_for(q.qid) if self.attack == "targeted" else kb.passages
        ranked = self.retriever.rank(q.text, passages)
        gold_rank = rank_of(q.gold_doc, ranked)
        verified_ranked, extra = [], {}
        if any(s in VERIFIED_SEARCH for s in self.systems):
            verified_ranked = self.retriever.rank(q.text, [p for p in passages if self.manifest.is_verified(p.text)])
            extra["verified_gold_rank"] = rank_of(q.gold_doc, verified_ranked)
        trials = []
        for system in self.systems:
            context = self._context(system, ranked, verified_ranked)
            trial = {**about, "system": system, "gold_rank": gold_rank, **self._candidates(q, ranked, POOL[system]),
                     **extra, **_shown(context, q), "prompt": self._prompt(system, q, context)}
            if system == TIERED:  # the second try, used only if the first finds no answer
                fallback = baseline_context(ranked)
                trial["fallback"] = {**_shown(fallback, q), "prompt": build_prompt(q.text, fallback, self.manifest)}
            trials.append(trial)
        return trials

    def _context(self, system: str, ranked: Ranking, verified_ranked: Ranking) -> list[Passage]:
        """The passages this system gives the model."""
        if system == BASELINE:
            return baseline_context(ranked)
        if system in RESERVED_K:
            return reserved_context(ranked, verified_ranked, reserved=RESERVED_K[system])
        if system == TIERED:
            return [p for p, _ in verified_ranked[:CONTEXT_SIZE]]
        return provenance_context(ranked, self.manifest, candidates=POOL[system])

    def _prompt(self, system: str, q: Question, context: list[Passage]) -> str:
        if system == BASELINE:
            return build_prompt(q.text, context)
        if system == TIERED:
            return build_prompt(q.text, context, self.manifest, VERIFIED_ONLY_INSTRUCTION)
        return build_prompt(q.text, context, self.manifest)

    def _candidates(self, q: Question, ranked: Ranking, n: int) -> dict:
        """What the top n passages hold, to explain failures (n: the passages the system re-ranks; 10 for the
        baseline, which takes its top 5 as they are)."""
        top = [p for p, _ in ranked[:n]]
        own = sum(is_poison_for(p, q.qid) for p in top)
        return {"candidates": n, "verified_in_candidates": sum(self.manifest.is_verified(p.text) for p in top),
                "own_poisons_in_candidates": own,
                "other_poisons_in_candidates": sum(p.label == "poisoned" for p in top) - own}

    def run(self, splits: Iterable[str], progress: Callable = lambda items, **_: items) -> pd.DataFrame:
        """Answers every trial (reusing cached replies) and scores it. Prompts are dropped from the result."""
        trials = self.plan(splits)
        self.new_calls = 0
        for trial in progress(trials, description=f"Answering with {self.model}"):
            reply = self._reply(trial.pop("prompt"))
            fallback = trial.pop("fallback", None)
            if fallback is not None:  # tiered answering
                trial["verified_reply"] = " ".join(reply.split())
                trial["fell_back"] = not_found(reply, trial["kind"])
                trial["fallback_context"] = fallback["context"] if trial["fell_back"] else ""
                if trial["fell_back"]:  # the passages shown now include the fallback's
                    reply = self._reply(fallback["prompt"])
                    trial["poisoned_in_context"] += fallback["poisoned_in_context"]
                    trial["gold_in_context"] = trial["gold_in_context"] or fallback["gold_in_context"]
            trial.update(model=self.model, reply=" ".join(reply.split()),
                         **score(reply, trial["kind"], trial["reference"], trial["target"]))
        return pd.DataFrame(trials)

    def _reply(self, prompt: str) -> str:
        """The model's reply, from the cache if it was asked this before."""
        reply = self.cache.get(self.model, prompt)
        if reply is None:
            reply = self.generate(prompt)
            self.cache.put(self.model, prompt, reply)
            self.new_calls += 1
        return reply


def retrieval_comparison(question_set: QuestionSet, kb: KnowledgeBase, manifest: HashManifest,
                         retriever: HybridRetriever, splits: Iterable[str]) -> pd.DataFrame:
    """Where each question's answer passage ranks with embedding search alone and with hybrid search, in the
    knowledge base without poisoning, and (for verified answer passages) in the separate search of verified passages.
    Needs embeddings but no model answers."""
    splits = set(splits)
    verified = [p for p in kb.passages if manifest.is_verified(p.text)]
    return pd.DataFrame([
        {"split": q.split, "qid": q.qid, "kind": q.kind, "gold_verified": q.gold_verified,
         "embedding_rank": rank_of(q.gold_doc, retriever.dense_rank(q.text, kb.passages)),
         "hybrid_rank": rank_of(q.gold_doc, retriever.rank(q.text, kb.passages)),
         "hybrid_rank_among_verified": rank_of(q.gold_doc, retriever.rank(q.text, verified)) if q.gold_verified
         else None}
        for q in question_set.questions if q.split in splits])
