"""The two studies: the data each one builds, the systems it compares and the success criteria fixed in advance.

The original study tests the provenance-aware defense against a baseline with embedding search, on 150 legitimate
passages and 40 questions. The follow-up, designed after the original results were known, adds hybrid search and the
reserved-slot defense, on a knowledge base of 1,945 passages and 60 fresh questions. Tiered answering was developed
on those fresh questions, so the final test runs it, unchanged, once on 60 new questions with criteria fixed in
advance.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .analysis import Criteria
from .corpus import Corpus, ExpandedCorpus
from .data_sources import AttackTechniques, KevCatalog
from .experiment import BASELINE, PROVENANCE, PROVENANCE_WIDE, RESERVED, TIERED, TRADEOFF_SYSTEMS
from .questions import FinalQuestionSet, FreshQuestionSet, QuestionSet, held_back_techniques
from .retrieval import CANDIDATES, CONTEXT_SIZE, RESERVED_SLOTS, WIDE_CANDIDATES

RESERVED_NOTE = (f"The reserved-slot defense always gives the model the top {RESERVED_SLOTS} passages from a separate "
                 f"search of verified passages.")
TIERED_NOTE = (f"answers from the top {CONTEXT_SIZE} verified passages alone and falls back to the regular top "
               f"{CONTEXT_SIZE}, flagged as unverified, only if they lack the answer.")
HYBRID_NOTE = "All systems use hybrid retrieval."


def original_data(techniques: AttackTechniques, kev: KevCatalog) -> tuple[Corpus, QuestionSet]:
    """150 legitimate passages (100 ATT&CK, 50 KEV) and 40 questions (10 dev, 30 test)."""
    corpus = Corpus(techniques, kev)
    return corpus, QuestionSet(corpus)


def followup_data(techniques: AttackTechniques, kev: KevCatalog) -> tuple[Corpus, QuestionSet]:
    """The original 150 passages plus every other KEV entry and top-level ATT&CK technique, except 30 techniques held
    back as false answers (1,945 passages in all), and the original 40 questions plus 60 fresh ones."""
    corpus, questions = original_data(techniques, kev)
    expanded = ExpandedCorpus(corpus, held_back_techniques(questions))
    return expanded, FreshQuestionSet(questions, expanded)


def final_data(techniques: AttackTechniques, kev: KevCatalog) -> tuple[Corpus, QuestionSet]:
    """The follow-up's knowledge base and 60 new questions from passages that no earlier question uses."""
    corpus, questions = followup_data(techniques, kev)
    return corpus, FinalQuestionSet(questions, corpus)


@dataclass(frozen=True)
class Study:
    name: str  # as given to --study
    title: str
    build: Callable[[AttackTechniques, KevCatalog], tuple[Corpus, QuestionSet]]
    systems: tuple[str, ...]  # each one answers every question in every condition
    pairs: tuple[tuple[str, str], ...]  # compared with McNemar's exact test
    question_splits: tuple[str, ...]  # the question sets its questions belong to
    splits: tuple[str, ...]  # question sets run by default
    attack: str  # attacks run by default: "both" or one of experiment.ATTACKS
    hybrid: bool  # keyword and embedding search (HybridRetriever) instead of embedding search alone
    criteria: Criteria
    systems_note: str  # shown under the results tables
    tag: str  # suffix of the saved corpus, manifest and questions files
    results_subdir: str  # folder under results/ for the trials and summaries


ORIGINAL = Study(
    name="original", title="the original study", build=original_data,
    systems=(BASELINE, PROVENANCE, PROVENANCE_WIDE),
    pairs=((BASELINE, PROVENANCE), (BASELINE, PROVENANCE_WIDE)),
    question_splits=("dev", "test"),
    splits=("dev", "test"), attack="both", hybrid=False,
    criteria=Criteria(split="test", defense=PROVENANCE, conditions=("S1-5", "S2-5"), slack=3),
    systems_note=f"The top-50 version re-ranks {WIDE_CANDIDATES} candidates instead of {CANDIDATES} (an extra "
                 f"analysis).",
    tag="", results_subdir="")

FOLLOWUP = Study(
    name="followup", title="the follow-up", build=followup_data,
    # then the trade-off analysis (exploratory) and tiered answering (validation), both added after the results
    systems=(BASELINE, PROVENANCE, RESERVED, *TRADEOFF_SYSTEMS, TIERED),
    pairs=((BASELINE, PROVENANCE), (BASELINE, RESERVED), (PROVENANCE, RESERVED), (BASELINE, TIERED),
           (RESERVED, TIERED)),
    question_splits=("dev", "test", "fresh"),
    splits=("fresh",), attack="targeted", hybrid=True,
    criteria=Criteria(split="fresh", defense=RESERVED, conditions=("S1-5", "S1-10", "S2-5", "S2-10"),
                      slack=6),  # the original study's 3 of 22 questions, scaled to 44
    systems_note=f"{RESERVED_NOTE} Tiered answering (validation) {TIERED_NOTE} {HYBRID_NOTE}",
    tag="_followup", results_subdir="followup")

FINAL = Study(
    name="final", title="the final test", build=final_data,
    systems=(BASELINE, RESERVED, TIERED),
    pairs=((BASELINE, RESERVED), (BASELINE, TIERED), (RESERVED, TIERED)),
    question_splits=("final",), splits=("final",), attack="targeted", hybrid=True,
    # fixed before any final-test answers were generated, as in the follow-up
    criteria=Criteria(split="final", defense=TIERED, conditions=("S1-5", "S1-10", "S2-5", "S2-10"),
                      slack=6),
    systems_note=f"{RESERVED_NOTE} Tiered answering, unchanged since the follow-up, {TIERED_NOTE} {HYBRID_NOTE}",
    tag="_final", results_subdir="final")

STUDIES = {study.name: study for study in (ORIGINAL, FOLLOWUP, FINAL)}
