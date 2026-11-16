"""Knowledge bases for the three scenarios.

S0  no poisoning: the 150 legitimate passages only
S1  plus n poisoned passages per question, whose claimed source is "Unverified"
S2  the same poisoned passages, relabeled to claim MITRE ATT&CK or CISA KEV as their source

Attacks are targeted: each question is answered from the 150 legitimate passages plus only its own poisoned passages
(passages_for). The all-at-once setting, where every question's poisoned passages share one knowledge base, is kept
as a stress test (passages).

Every passage keeps the source it claims, which the baseline sees. Whether a passage is verified comes only from the
hash manifest, never from that label, so S2 changes what the baseline sees but not what the defense verifies.
"""
from __future__ import annotations

import pandas as pd

from .corpus import Corpus, Passage
from .manifest import HashManifest
from .poisons import PoisonSet
from .questions import QuestionSet

SCENARIOS = {"S0": "No poisoning",
             "S1": "Poisoned passages labeled unverified",
             "S2": "Poisoned passages claim MITRE ATT&CK or CISA KEV"}
POISONED_SCENARIOS = ("S1", "S2")
POISON_COUNTS = (1, 5, 10)  # poisoned passages per question
CONDITIONS = ("S0", *(f"{scenario}-{n}" for scenario in POISONED_SCENARIOS for n in POISON_COUNTS))
UNVERIFIED_LABEL = "Unverified"
FORGED_LABELS = {"attack": "MITRE ATT&CK", "kev": "CISA KEV"}


def is_poison_for(passage: Passage, qid: str) -> bool:
    """Whether the passage is one of the poisoned passages written for this question."""
    return passage.doc_id.startswith(f"POI-{qid}-")


class KnowledgeBase:
    """The passages a system retrieves from in one scenario."""

    def __init__(self, corpus: Corpus, question_set: QuestionSet, poisons: PoisonSet | None = None,
                 scenario: str = "S0", n_poisons: int = 0) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; expected one of {list(SCENARIOS)}")
        if scenario == "S0" and n_poisons:
            raise ValueError("S0 has no poisoned passages")
        if scenario != "S0" and (poisons is None or n_poisons < 1):
            raise ValueError(f"{scenario} needs poisoned passages and n_poisons of at least 1")
        self.corpus, self.question_set, self.poisons = corpus, question_set, poisons
        self.scenario, self.n_poisons = scenario, n_poisons
        self._passages: list[Passage] | None = None

    @property
    def condition(self) -> str:
        """S0, or the scenario with its poisons per question, e.g. S1-5."""
        return self.scenario if self.scenario == "S0" else f"{self.scenario}-{self.n_poisons}"

    @property
    def passages(self) -> list[Passage]:
        if self._passages is None:
            self._passages = list(self.corpus.passages) + self._poisoned_passages()
        return self._passages

    def passages_for(self, qid: str) -> list[Passage]:
        """The legitimate passages plus this question's own poisoned passages (a targeted attack)."""
        return [p for p in self.passages if p.label != "poisoned" or is_poison_for(p, qid)]

    def to_frame(self, manifest: HashManifest) -> pd.DataFrame:
        return pd.DataFrame([{"doc_id": p.doc_id, "claimed_source": p.source, "verified": manifest.is_verified(p.text),
                              "label": p.label, "text": p.text} for p in self.passages])

    def composition(self, manifest: HashManifest) -> dict:
        frame = self.to_frame(manifest)
        poisoned = frame.label == "poisoned"
        return {"scenario": self.scenario, "poisons per question": self.n_poisons,
                "passages per question": int((~poisoned).sum()) + self.n_poisons,
                "verified": int(frame.verified.sum()),
                "legitimate, unverified": int((~poisoned & ~frame.verified).sum()),
                "poisoned, all questions": int(poisoned.sum()),
                "poisoned but verified": int((poisoned & frame.verified).sum())}  # must be 0

    def _poisoned_passages(self) -> list[Passage]:
        if self.scenario == "S0":
            return []
        passages = []
        for q in self.question_set.questions:
            gold = self.corpus.by_id(q.gold_doc)
            source = UNVERIFIED_LABEL if self.scenario == "S1" else FORGED_LABELS[q.kind]
            for poison in self.poisons.for_question(q.qid, self.n_poisons):
                assert poison.target == q.target, f"{q.qid}: poisoned passages are stale; regenerate them"
                passages.append(Passage(doc_id=f"POI-{q.qid}-{poison.rank:02d}", source=source, text=poison.text,
                                        key=gold.key, name=gold.name, label="poisoned"))
        return passages
