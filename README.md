# provenance-rag-poisoning-defense
Tests whether provenance-aware retrieval reduces document poisoning attacks on a cybersecurity RAG system.

## Run
```
python main.py                       # runs everything: builds the data, then the original study (dev and test sets,
                                     # targeted attack and stress test), then the follow-up (60 fresh questions),
                                     # then the final test (60 new questions).
                                     # Answers are cached, so a re-run only answers what is new and resumes if stopped
python main.py --study original      # just the original study (or --study followup, or --study final)
python main.py --split none          # builds the data only
python main.py --regenerate-poisons  # writes new poisoned passages, replacing the saved ones
```

### Using Ollama on JupyterHub
The code stays on your computer; only the model runs on the hub. On the hub, in a terminal:
```
OLLAMA_HOST=0.0.0.0:11434 ollama serve     # leave running; once per server start
ollama pull llama3.1:8b                    # once, in a second terminal
ollama pull all-minilm:l6-v2               # once: the embedding model for retrieval
```
To have Ollama start by itself whenever your hub server starts, paste the contents of
`hub/setup_ollama_autostart.sh` into a hub terminal once. It stops when the server stops; its log is `~/ollama.log`.

On your computer (on the VPN), once: `cp .env.example .env`, then fill in `.env` with your hub address, username and
an API token from https://<hub>/hub/token. `.env` is git-ignored. After that, `python main.py` uses the hub.

## Design
- **Corpus:** 150 legitimate passages: 100 MITRE ATT&CK Enterprise techniques and 50 CISA KEV entries.
- **Questions:** 40 (10 dev, 30 test), each with a reference answer and a false target answer.
- **Manifest:** SHA-256 fingerprints of the legitimate passages, stored apart from the knowledge base. The gold
  passages of 10 questions (2 dev, 8 test) are held out, so the defense sees them as unverified even though they are
  correct. This measures what deprioritizing unverified content costs.
- **Poisoned passages:** 10 per question in the PoisonedRAG black-box style (the question followed by text that
  asserts the target answer). Git-ignored and never published.
- **Scenarios:** S0 no poisoning; S1 poisoned passages labeled "Unverified"; S2 the same passages claiming MITRE
  ATT&CK or CISA KEV as their source. S1 and S2 are built with 1, 5 and 10 poisoned passages per question.
- **Prompts:** both systems share one template. The baseline sees each passage's claimed source; the defense also
  sees whether it is verified and is told to prefer verified passages when passages conflict.
- **Retrieval:** both systems embed with all-MiniLM-L6-v2 (served by Ollama as `all-minilm:l6-v2`) and rank by cosine
  similarity. The baseline gives the model the top 5; the provenance-aware system re-ranks the top 10 verified-first
  and gives it the first 5. An extra version re-ranks the top 50 instead, to show how the candidate pool limits the
  defense (the success criteria apply to the top-10 design). A closed-book run (no passages) shows what the model
  answers from memory.
- **Results:** `results/trials_<split>.csv` has every answer with its context and scores; `results/summary_<split>.csv`
  the rates. Model replies are cached in `results/replies.jsonl` (git-ignored), so re-runs reuse them and an
  interrupted run resumes.
- **Attacks:** targeted (the main results): each question is answered from the 150 legitimate passages plus only its
  own poisoned passages. All-at-once (a stress test): every question's poisoned passages share one knowledge base. The
  dev run showed that poisons written for other questions, which share the question wording, then fill the top 10.

## Follow-up study (`--study followup`)
The original study found two limits: the defense fails once poisoned passages fill all of its candidates (10 poisons
per question, or the stress test), and it cannot tell a correct unverified passage from a poisoned one. The
follow-up tests two changes on questions the original study never used:
- **Hybrid retrieval** for every system: the embedding ranking fused with BM25 keyword search by reciprocal rank
  fusion (k = 60), so exact names and CVE numbers rank their passages higher.
- **Reserved slots:** a new defense that searches the verified passages separately and always gives the model the top
  3 of them; the other 2 slots take the best-ranked remaining passages, verified or not. It is compared with the
  baseline and the original provenance-aware design, all with hybrid retrieval and the same prompts.
- **Larger knowledge base:** the original 150 passages plus every other KEV entry and every other top-level ATT&CK
  technique: 1,945 passages (197 ATT&CK, 1,748 KEV). Thirty techniques, including every original false answer, are held
  back so false answers remain real technique IDs that no legitimate passage contains.
- **Fresh test set:** 60 new questions (38 ATT&CK, 22 KEV; Q101 onward) from passages no original question uses; 16 of
  their gold passages are held out of the manifest. Every false answer (technique ID or date) appears in no legitimate
  passage. The original 40 questions keep their poisoned passages. The follow-up runs the targeted attack only. Its
  corpus, manifest and questions are saved as `*_followup.*`, its results in `results/followup/`.

Success criteria, fixed before any follow-up answers were generated (reserved slots vs baseline, fresh test set):
(a) targeted attack: ASR at most half the baseline's in S1 and S2 with 5 and with 10 poisoned passages per question;
(b) S0: at most 6 fewer correct answers than the baseline on the 44 questions whose answer passage is verified (the
original study's 3 of 22, scaled).
Accuracy on held-out answer passages and the embedding-versus-hybrid retrieval comparison are reported without
thresholds.

**Trade-off analysis (exploratory, added after the follow-up results):** the same reserved-slot defense with 1 to 5 of
the 5 slots kept for verified passages, run on the same fresh questions. Every setting is reported; none is chosen
as a winner, and the success criteria still apply only to the design fixed in advance (3 of 5). It shows how
restricting the model to verified passages lowers attack success but leaves less room for correct passages that are
missing from the manifest. No exact-ID lookup is used.

**Tiered answering (validation, added after the follow-up results):** the model first sees only the top 5 passages
from the separate search of verified passages and may reply NOT FOUND; only then is it asked again with the regular
top 5, verified or not, and that answer is flagged as unverified for the analyst. It runs on the same 60 fresh
questions, so this is a validation run; a design that works there is to be tested once on new questions, with
success criteria fixed in advance.

## Final test (`--study final`)
Tiered answering was developed on the follow-up's 60 fresh questions, so it is tested once, unchanged, on 60 new
questions (Q201 onward; 38 ATT&CK, 22 KEV), written the same way from passages that no earlier question uses, on the
follow-up's knowledge base (1,945 passages). The gold passages of 16 of them (10 ATT&CK, 6 KEV) are held out of the
manifest, and no false answer appears in any legitimate passage. Poisoned passages are written for them like the
others. It compares the baseline, reserved slots and tiered answering, all with hybrid retrieval, under the targeted
attack. Its corpus, manifest and questions are saved as `*_final.*`, its results in `results/final/`.

Success criteria, fixed before any final-test answers were generated (tiered answering vs baseline):
(a) ASR at most half the baseline's in S1 and S2 with 5 and with 10 poisoned passages per question;
(b) S0: at most 6 fewer correct answers than the baseline on the 44 questions whose answer passage is verified.
Reserved slots, accuracy on held-out answer passages, flagged answers and the retrieval comparison are reported
without thresholds.
