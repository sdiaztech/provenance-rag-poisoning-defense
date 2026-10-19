from pathlib import Path

import pandas as pd

from provenance_rag.console import ConsoleReport
from provenance_rag.corpus import Corpus
from provenance_rag.data_sources import AttackTechniques, KevCatalog
from provenance_rag.manifest import HashManifest
from provenance_rag.questions import QuestionSet

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"  # exactly what MITRE and CISA published
PROCESSED_DIR = ROOT / "data" / "processed"  # real data reshaped by the code
SYNTHETIC_DIR = ROOT / "data" / "synthetic"  # data created for the experiment


def main() -> None:
    report = ConsoleReport()
    techniques, kev = AttackTechniques(RAW_DIR), KevCatalog(RAW_DIR)
    for source in (techniques, kev):
        report.show_source(source)

    corpus = Corpus(techniques, kev)
    passages = corpus.to_frame()
    report.show_table("Trusted corpus", passages.head(5),
                      f"{len(passages)} passages ({len(corpus.by_source('MITRE ATT&CK'))} ATT&CK, "
                      f"{len(corpus.by_source('CISA KEV'))} KEV), {passages.words.mean():.0f} words on average, "
                      f"{passages.words.max()} at most. First 5:")

    manifest = HashManifest.from_corpus(corpus)
    original = corpus.passages[0].text
    checks = pd.DataFrame({
        "passage": ["Trusted passage, unchanged", "Same passage, last character changed",
                    "Fake passage that claims MITRE ATT&CK as its source"],
        "verified": [manifest.is_verified(original), manifest.is_verified(original[:-1] + "!"),
                     manifest.is_verified("MITRE ATT&CK Enterprise technique T0000: Fake. Claims a trusted source.")],
    })
    report.show_table("Hash manifest", checks, f"{len(manifest)} SHA-256 fingerprints. Verification check:")

    question_set = QuestionSet(corpus)
    questions = question_set.to_frame()
    report.show_table("Evaluation questions", questions,
                      f"{len(questions)} questions: {(questions.split == 'dev').sum()} dev, "
                      f"{(questions.split == 'test').sum()} test")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    saved = [corpus.save(PROCESSED_DIR / "trusted_corpus.csv"),
             manifest.save(PROCESSED_DIR / "hash_manifest.json"),
             question_set.save(SYNTHETIC_DIR / "questions.csv")]
    report.show_table("Saved files", pd.DataFrame({"file": [str(p.relative_to(ROOT)) for p in saved],
                                                   "size": [f"{p.stat().st_size:,} bytes" for p in saved]}))


if __name__ == "__main__":
    main()
