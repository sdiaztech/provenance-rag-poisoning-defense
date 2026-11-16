"""Trusted corpus: 100 MITRE ATT&CK technique passages and 50 CISA KEV passages (original study).

ExpandedCorpus is the follow-up's larger knowledge base: the same 150 passages plus every other CISA KEV entry and
every other top-level ATT&CK technique except a few held back as false answers.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data_sources import AttackTechniques, KevCatalog


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Passage:
    doc_id: str
    source: str  # claimed source label
    text: str
    key: str  # technique ID or CVE ID
    name: str
    due_date: str | None = None
    date_added: str | None = None
    label: str = "trusted"  # ground truth, used only for scoring

    @property
    def sha256(self) -> str:
        return sha256(self.text)

    @property
    def words(self) -> int:
        return len(self.text.split())


class TextCleaner:
    """Removes citation markers, links and markup, and caps passages at a word limit."""

    def __init__(self, max_words: int = 250) -> None:
        self.max_words = max_words

    def clean(self, text: str) -> str:
        text = re.sub(r"\(Citation:[^)]*\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)", r"\1", text)  # [text](url) -> text
        text = re.sub(r"<[^>]+>", "", text).replace("`", "")
        return " ".join(text.split())

    def truncate(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.max_words:
            return text
        cut = " ".join(words[: self.max_words])
        last_period = cut.rfind(".")
        return cut[: last_period + 1] if last_period > 0.6 * len(cut) else cut

    def __call__(self, text: str) -> str:
        return self.truncate(self.clean(text))


class Corpus:
    """The trusted passages, sampled with a fixed seed so every run gives the same corpus."""

    def __init__(self, techniques: AttackTechniques, kev: KevCatalog, n_attack: int = 100, n_kev: int = 50,
                 seed: int = 760, cleaner: TextCleaner | None = None) -> None:
        self.techniques = techniques
        self.kev = kev
        self.n_attack = n_attack
        self.n_kev = n_kev
        self.seed = seed
        self.cleaner = cleaner or TextCleaner()
        self._passages: list[Passage] | None = None

    @property
    def eligible_techniques(self) -> pd.DataFrame:
        """Top-level techniques with a description (sub-technique IDs look like T1055.001)."""
        table = self.techniques.table
        top_level = table["ID"].astype(str).str.fullmatch(r"T\d{4}")
        has_text = table["description"].fillna("").str.strip() != ""
        return table[top_level & has_text].sort_values("ID").reset_index(drop=True)

    @property
    def passages(self) -> list[Passage]:
        if self._passages is None:
            self._passages = self._build()
            assert len({p.sha256 for p in self._passages}) == len(self._passages), "duplicate passages"
        return self._passages

    def _build(self) -> list[Passage]:
        return self._attack_passages() + self._kev_passages()

    def by_source(self, source: str) -> list[Passage]:
        return [p for p in self.passages if p.source == source]

    def by_id(self, doc_id: str) -> Passage:
        return next(p for p in self.passages if p.doc_id == doc_id)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"doc_id": p.doc_id, "source": p.source, "words": p.words,
                              "sha256": p.sha256[:12] + "…", "text": p.text} for p in self.passages])

    def save(self, path: Path) -> Path:
        pd.DataFrame([{**asdict(p), "sha256": p.sha256} for p in self.passages]).to_csv(path, index=False)
        return path

    def _attack_passages(self) -> list[Passage]:
        rows = self.eligible_techniques.to_dict("records")
        picked = random.Random(self.seed).sample(rows, self.n_attack)
        return [self._technique_passage(r) for r in picked]

    def _kev_passages(self) -> list[Passage]:
        picked = random.Random(self.seed + 2).sample(self._kev_rows(), self.n_kev)
        return [self._kev_passage(v) for v in picked]

    def _kev_rows(self) -> list[dict]:
        """KEV entries with a due date and a description, in CVE order."""
        return sorted((v for v in self.kev.table.to_dict("records") if v.get("dueDate") and v.get("shortDescription")),
                      key=lambda v: v["cveID"])

    def _technique_passage(self, r: dict) -> Passage:
        return Passage(doc_id=f"ATT-{r['ID']}", source="MITRE ATT&CK", key=r["ID"], name=r["name"],
                       text=self.cleaner(f"MITRE ATT&CK Enterprise technique {r['ID']}: {r['name']}. "
                                         f"{r['description']}"))

    def _kev_passage(self, v: dict) -> Passage:
        return Passage(doc_id=f"KEV-{v['cveID']}", source="CISA KEV", key=v["cveID"],
                       name=f"{v['vendorProject']} {v['product']}", due_date=v["dueDate"], date_added=v["dateAdded"],
                       text=self.cleaner(
                           f"CISA Known Exploited Vulnerabilities catalog entry {v['cveID']}: "
                           f"{v['vulnerabilityName']}. "
                           f"Vendor/project: {v['vendorProject']}. Product: {v['product']}. {v['shortDescription']} "
                           f"Date added to the catalog: {v['dateAdded']}. Remediation due date: {v['dueDate']}. "
                           f"Required action: {v['requiredAction']} "
                           f"Known ransomware campaign use: {v['knownRansomwareCampaignUse']}."))


class ExpandedCorpus(Corpus):
    """The follow-up's knowledge base: the original corpus's 150 passages (unchanged, so the original questions and
    their poisoned passages carry over), plus every other KEV entry and every other top-level ATT&CK technique except
    those in held_back. Held-back techniques have no passage, so their IDs can serve as false answers."""

    def __init__(self, base: Corpus, held_back: Iterable[str]) -> None:
        super().__init__(base.techniques, base.kev, base.n_attack, base.n_kev, base.seed, base.cleaner)
        self.base = base
        self.held_back = frozenset(held_back)

    def _build(self) -> list[Passage]:
        have = {p.key for p in self.base.passages}
        assert not have & self.held_back, "a held-back technique is in the original corpus"
        attack = [self._technique_passage(r) for r in self.eligible_techniques.to_dict("records")
                  if r["ID"] not in have and r["ID"] not in self.held_back]
        kev = [self._kev_passage(v) for v in self._kev_rows() if v["cveID"] not in have]
        return list(self.base.passages) + attack + kev
