"""Hash manifest: SHA-256 fingerprints of the trusted passages, kept apart from the knowledge base."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .corpus import Corpus, sha256


class HashManifest:
    """A passage counts as verified only if its fingerprint is listed here, whatever source it claims."""

    def __init__(self, fingerprints: Iterable[str]) -> None:
        self._fingerprints = frozenset(fingerprints)

    @classmethod
    def from_corpus(cls, corpus: Corpus, held_out: Iterable[str] = ()) -> HashManifest:
        """Fingerprints every corpus passage except those whose doc_id is in held_out.

        Held-out passages are legitimate but unverified, like real content published after the manifest was signed.
        They let the experiment measure what deprioritizing unverified passages costs.
        """
        held_out = set(held_out)
        unknown = held_out - {p.doc_id for p in corpus.passages}
        assert not unknown, f"held-out passages not in the corpus: {sorted(unknown)}"
        return cls(p.sha256 for p in corpus.passages if p.doc_id not in held_out)

    def is_verified(self, text: str) -> bool:
        return sha256(text) in self._fingerprints

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(sorted(self._fingerprints), indent=1))
        return path

    def __len__(self) -> int:
        return len(self._fingerprints)
