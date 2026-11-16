# Follow-up study: progress

Goal: test the two improvements (hybrid retrieval and reserved slots for verified passages) on a larger knowledge
base and fresh questions, with criteria fixed before any follow-up answers. Nothing here changes the original study,
whose results stay reproducible with `python main.py --split all`.

- [x] 1. Larger knowledge base (`ExpandedCorpus`): the original 150 passages plus every other KEV entry and top-level
      ATT&CK technique, except 30 techniques held back as false answers. 1,945 passages (197 ATT&CK, 1,748 KEV).
      Tested offline: original study unchanged; follow-up runs end to end.
- [x] 2. 60 fresh questions (38 ATT&CK, 22 KEV, 16 held out of the manifest); every false answer appears in no
      legitimate passage; criterion (b) scaled to 6 of 44. Original 40 questions unchanged.
- [x] 3. Final offline test and README update: original study identical; follow-up runs end to end (1,320 answers).
- [x] 3c. `python main.py` with no options runs everything: the original study (cached, a minute or two), then the
      follow-up.
- [x] 3b. Ran on the hub (Oct 28): 21.6 min writing poisoned passages, 22.9 min answering. Results in
      results/followup/. Criterion (a) not met (reserved slots: 52% ASR at 5 and 10 poisons vs required <= 50%);
      criterion (b) met (39/44 vs baseline 38/44).
- [x] 3d. Trade-off analysis added (exploratory): 1 to 5 verified slots on the same 60 questions, all settings
      reported. Tested offline; existing answers reused, only the new settings need the model.
- [x] 3e. Ran on the hub (Oct 28; one 408 proxy timeout, now retried automatically). Trade-off: ASR at 5 poisons
      51/50/52/45/0% for 1-5 verified slots; S0 held-out correct 13/13/13/12/1 of 16; S0 verified 39/44 in all.
- [x] 4. Paper updated (Oct 28): Method and Results "Follow Up Study" subsections, Table 4 (attack success and
  accuracy), Table 5 and "Trade Off Between Restriction and Cost", two Discussion paragraphs, Limitations and
  Future Work (tiered answering, untested), references Cormack et al. (2009) and Robertson & Zaragoza (2009).
- [x] 5. Tiered answering added to the follow-up as a validation run (the idea came from the follow-up results, so
      the 60 fresh questions are now validation data): the model first sees only the top 5 verified passages and may
      reply NOT FOUND; only then does it get the regular top 5, verified or not, and that answer is flagged as
      unverified. Tested offline: original study byte-identical; existing follow-up answers unchanged.
- [x] 5b. Ran on the hub (Oct 28, 132 new answers). ASR 25% with 1 poisoned passage, 27% (16/60) with 5 and 10
      (reserved slots 40-52%, baseline 85-100%). S0 correct 46/60 (baseline 51, reserved slots 52): verified answer
      passages 39/44, held out 7/16. 16/60 answers flagged, the same questions in every condition; no unflagged
      answer was attacked (0 of 264). NOT FOUND: answered all 38 questions whose answer was in the top 5 verified
      passages (all correct) and said NOT FOUND for all 6 verified answers the search missed; for the 16 held-out
      questions it said NOT FOUND 10 times and guessed a related technique's ID 6 times (all ATT&CK, all wrong).
- [x] 6. Final test built (Oct 28), tiered answering unchanged: 60 new questions Q201-Q260 (38 ATT&CK, 22 KEV, 16
      held out of the manifest) from passages no earlier question uses; no false answer in any legitimate passage;
      baseline, reserved slots and tiered answering on the follow-up's knowledge base, targeted attack. Criteria fixed
      before any answers (README): (a) ASR at most half the baseline's in S1 and S2 with 5 and 10 poisoned passages per
      question; (b) S0 at most 6 fewer correct than the baseline on the 44 verified answer passages. Tested offline:
      earlier studies byte-identical, poisoned passages for the new questions written, every option checked.
- [x] 6b. Ran on the hub (Oct 28; 21.9 min writing poisoned passages, 24.2 min answering). Both criteria met:
      tiered answering ASR 30% (18/60) at 5 and 10 poisoned passages vs required <= 50% (baseline 100%); S0 verified
      37/44 vs required >= 30 (baseline 36). ASR 23% at 1 poisoned passage; reserved slots 37-43%, lower for tiered in
      every attacked condition (p = .008 to .039). S0 overall 47/60 (baseline 52, reserved slots 53), the difference
      all on held-out questions (10 vs 16). 18/60 answers flagged, none of 252 unflagged answers attacked; guesses on
      8 ATT&CK questions (6 held out, 2 search misses). Paper updated: Results "Final Test" with Table 7, two
      Discussion paragraphs, Limitations and Future Work.
- [ ] Optional: add the 479 ATT&CK sub-techniques to the knowledge base (needs scoring changes).

To resume in a new chat: share this file and ask to continue from the first unchecked step.

## Presentation notes
- Opening slide idea: Google AI Overviews (2024) suggested using glue to make cheese stick to pizza, drawn from a
  sarcastic forum comment. No attacker needed; now imagine someone doing it on purpose to a security assistant.
  Source: Reid, E. (2024, May 30). AI Overviews: About last week. The Keyword.
  https://blog.google/products/search/ai-overviews-update-may-2024/ (already cited in the paper's introduction).
