"""Builds the data and runs the studies. With no options it runs everything: python main.py (see README.md)."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from rich.progress import track

from provenance_rag import analysis
from provenance_rag.console import ConsoleReport
from provenance_rag.corpus import Corpus, ExpandedCorpus
from provenance_rag.data_sources import AttackTechniques, KevCatalog
from provenance_rag.experiment import ATTACKS, TRADEOFF_SYSTEMS, Experiment, ReplyCache, retrieval_comparison
from provenance_rag.manifest import HashManifest
from provenance_rag.ollama import OllamaClient, OllamaUnavailable
from provenance_rag.poisons import PoisonSet, ollama_writer
from provenance_rag.questions import QuestionSet
from provenance_rag.retrieval import CONTEXT_SIZE, EMBEDDING_MODEL, RESERVED_SLOTS, HybridRetriever, Retriever
from provenance_rag.scenarios import POISON_COUNTS, POISONED_SCENARIOS, SCENARIOS, KnowledgeBase
from provenance_rag.studies import STUDIES, Study

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"  # exactly what MITRE and CISA published
PROCESSED_DIR = ROOT / "data" / "processed"  # real data reshaped by the code
SYNTHETIC_DIR = ROOT / "data" / "synthetic"  # data created for the experiment
POISONS_PATH = SYNTHETIC_DIR / "poisoned_passages.csv"  # git-ignored: never published
LOGS_DIR = ROOT / "logs"  # a copy of each run's output; git-ignored because it shows poisoned passages
RESULTS_DIR = ROOT / "results"  # answers and summaries; replies.jsonl is a git-ignored cache of model replies
ANSWER_SETTINGS = {"seed": 0, "temperature": 0.0, "max_tokens": 48}
SPLIT_NOTES = {"fresh": "the follow-up's test set, from passages no original question uses",
               "final": "the final test set, from passages no earlier question uses"}
ANY_STUDY_SPLITS = ("auto", "all", "none")  # --split values that every study accepts
ATTACK_LABELS = {"targeted": ("targeted attack", ""),  # (title in the terminal, suffix of the results files)
                 "all-at-once": ("all-at-once stress test", "_all_at_once")}


def load_env(path: Path) -> None:
    """Reads KEY=VALUE lines from the git-ignored .env file; variables already set in the shell take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Builds the data and runs the experiments. With no options it runs "
                                                 "everything: the original study, the follow-up, then the final test.")
    parser.add_argument("--study", choices=["all", *STUDIES], default="all",
                        help="all (default): every study, in order. original: embedding retrieval; baseline, "
                             "provenance-aware and its top-50 version; 150 passages. followup: hybrid retrieval; "
                             "baseline, provenance-aware and the reserved-slot defense (plus the trade-off analysis "
                             "and tiered answering), on a larger knowledge base (1,945 passages) with 60 fresh "
                             "questions; targeted attack only. final: baseline, reserved slots and tiered answering "
                             "on 60 new questions; targeted attack only")
    parser.add_argument("--split", choices=["auto", "dev", "test", "fresh", "final", "all", "none"], default="auto",
                        help="questions to run. auto (default): dev and test for the original study, the fresh "
                             "questions for the follow-up, the final questions for the final test. none: build the "
                             "data only. A named question set runs only in the studies that have it. Answers are "
                             "cached, so re-running reuses them")
    parser.add_argument("--attack", choices=["both", *ATTACKS], default=None,
                        help="both (default): the targeted attack, where each question is answered from the "
                             "legitimate passages plus its own poisoned passages, then the all-at-once stress test, "
                             "where every question's poisoned passages share one knowledge base. The follow-up runs "
                             "the targeted attack only unless you ask for another")
    parser.add_argument("--regenerate-poisons", action="store_true",
                        help="rewrite the poisoned passages for every question in the study with Ollama, even if the "
                             "saved ones still match (passages for questions without them are written automatically)")
    parser.add_argument("--model", default="llama3.1:8b",
                        help="Ollama model that writes the poisoned passages and answers the questions")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL,
                        help=f"Ollama embedding model for retrieval (default: {EMBEDDING_MODEL}, all-MiniLM-L6-v2)")
    parser.add_argument("--ollama-url", default=None,
                        help="Ollama address (default: OLLAMA_URL, else http://localhost:11434). For JupyterHub: "
                             "https://<hub>/user/<username>/proxy/11434, with JUPYTERHUB_TOKEN set")
    args = parser.parse_args()
    if args.study != "all" and args.split not in (*ANY_STUDY_SPLITS, *STUDIES[args.study].question_splits):
        parser.error(f"the {args.study} study has no {args.split} questions; it has "
                     f"{', '.join(STUDIES[args.study].question_splits)} (or leave --study out)")
    return args


def selected_studies(args: argparse.Namespace) -> list[Study]:
    """The studies --study names, without those that have no questions in the --split question set."""
    studies = STUDIES.values() if args.study == "all" else [STUDIES[args.study]]
    return [study for study in studies if args.split in ANY_STUDY_SPLITS or args.split in study.question_splits]


def splits_for(study: Study, split: str) -> list[str]:
    """The question sets to answer in this study for --split; none for --split none."""
    if split == "none":
        return []
    if split == "auto":
        return list(study.splits)
    if split == "all":
        return list(study.question_splits)
    return [split]


def main() -> None:
    load_env(ROOT / ".env")  # OLLAMA_URL and JUPYTERHUB_TOKEN, if set there
    args = parse_args()
    report = ConsoleReport()
    client = OllamaClient(args.ollama_url)
    studies = selected_studies(args)
    try:
        if args.split != "none":  # check the model server once, before building anything
            problem = client.problem(args.model, args.embedding_model)
            if problem:
                report.show_note(f"Ollama: not reachable, so nothing was run. {problem}", style="red")
                raise SystemExit(1)
            report.show_note(f"Ollama: ready at {client.url} ({args.model} and {args.embedding_model}).",
                             style="green")
        for number, study in enumerate(studies, start=1):
            if len(studies) > 1:
                report.show_note(f"Study {number} of {len(studies)}: {study.title}", style="bold")
            run_study(study, args, client, report)
    except KeyboardInterrupt:
        report.console.print("Stopped with Ctrl+C.", style="red")
        raise SystemExit(130)
    except Exception:
        report.console.print_exception()  # so the error is in the log too
        raise SystemExit(1)
    finally:
        log = report.save_log(LOGS_DIR)
        report.console.print(f"Output saved to {os.path.relpath(log, ROOT)} and latest.txt next to it", style="dim")


def run_study(study: Study, args: argparse.Namespace, client: OllamaClient, report: ConsoleReport) -> None:
    """Builds one study's data, runs its experiment and lists the files it saved."""
    started = time.time()
    corpus, question_set, manifest = build_data(study, report)
    poisons = load_poisons(args, client, question_set, report)
    bases = knowledge_bases(corpus, question_set, poisons, manifest, report)
    results = run_experiment(study, args, client, report, question_set, bases, manifest)

    saved = save_data(study, corpus, manifest, question_set)
    if poisons is not None and POISONS_PATH.stat().st_mtime >= started:
        saved.append(POISONS_PATH)  # written during this run
    saved += results
    report.show_table("Saved files", pd.DataFrame({"file": [os.path.relpath(p, ROOT) for p in saved],
                                                   "size": [f"{p.stat().st_size:,} bytes" for p in saved]}))


def build_data(study: Study, report: ConsoleReport) -> tuple[Corpus, QuestionSet, HashManifest]:
    """The study's legitimate passages, questions and hash manifest, each shown as it is built."""
    techniques, kev = AttackTechniques(RAW_DIR), KevCatalog(RAW_DIR)
    for source in (techniques, kev):
        report.show_source(source)
    corpus, question_set = study.build(techniques, kev)
    show_corpus(corpus, report)
    show_questions(question_set, report)
    return corpus, question_set, build_manifest(corpus, question_set, report)


def show_corpus(corpus: Corpus, report: ConsoleReport) -> None:
    passages = corpus.to_frame()
    expanded = (f" The follow-up's knowledge base: the original 150 plus every other KEV entry and top-level "
                f"technique, except {len(corpus.held_back)} techniques held back as false answers."
                if isinstance(corpus, ExpandedCorpus) else "")
    report.show_table("Trusted corpus", passages.head(5),
                      f"{len(passages)} passages ({len(corpus.by_source('MITRE ATT&CK'))} ATT&CK, "
                      f"{len(corpus.by_source('CISA KEV'))} KEV), {passages.words.mean():.0f} words on average, "
                      f"{passages.words.max()} at most.{expanded} First 5:")


def show_questions(question_set: QuestionSet, report: ConsoleReport) -> None:
    questions = question_set.to_frame()
    per_split = questions.split.value_counts(sort=False)
    unverified = questions[~questions.gold_verified].split.value_counts(sort=False)
    report.show_table("Evaluation questions", questions,
                      f"{len(questions)} questions: " + ", ".join(f"{n} {split}" for split, n in per_split.items())
                      + "".join(f" ({note})" for split, note in SPLIT_NOTES.items() if split in per_split)
                      + f". For {unverified.sum()} of them ("
                      + ", ".join(f"{n} {split}" for split, n in unverified.items())
                      + ") the gold passage is legitimate but held out of the manifest (gold_verified = False).")


def build_manifest(corpus: Corpus, question_set: QuestionSet, report: ConsoleReport) -> HashManifest:
    """Fingerprints every legitimate passage except the held-out gold passages, and shows that verification works."""
    held_out = question_set.unverified_gold_docs
    manifest = HashManifest.from_corpus(corpus, held_out=held_out)
    trusted = next(p.text for p in corpus.passages if p.doc_id not in held_out)
    checks = pd.DataFrame({
        "passage": ["Trusted passage, unchanged", "Same passage, last character changed",
                    "Fake passage that claims MITRE ATT&CK as its source",
                    "Legitimate passage held out of the manifest"],
        "verified": [manifest.is_verified(trusted), manifest.is_verified(trusted[:-1] + "!"),
                     manifest.is_verified("MITRE ATT&CK Enterprise technique T0000: Fake. Claims a trusted source."),
                     manifest.is_verified(corpus.by_id(min(held_out)).text)],
    })
    report.show_table("Hash manifest", checks,
                      f"{len(manifest)} SHA-256 fingerprints for {len(corpus.passages)} legitimate passages "
                      f"({len(held_out)} held out). Verification check:")
    return manifest


def save_data(study: Study, corpus: Corpus, manifest: HashManifest, question_set: QuestionSet) -> list[Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    return [corpus.save(PROCESSED_DIR / f"trusted_corpus{study.tag}.csv"),
            manifest.save(PROCESSED_DIR / f"hash_manifest{study.tag}.json"),
            question_set.save(SYNTHETIC_DIR / f"questions{study.tag}.csv")]


def load_poisons(args: argparse.Namespace, client: OllamaClient, question_set: QuestionSet,
                 report: ConsoleReport) -> PoisonSet | None:
    """Reuses the saved poisoned passages and writes them with Ollama only for questions that have none (or whose
    wording or target changed), so adding questions never changes the passages of existing ones. With
    --regenerate-poisons, every question in the study gets new passages. Returns the passages for this study's
    questions, or None if some are missing and Ollama isn't available."""
    saved = PoisonSet.load(POISONS_PATH) if POISONS_PATH.exists() else PoisonSet([])
    questions = question_set.questions
    todo = list(questions) if args.regenerate_poisons else saved.missing(questions)
    if not todo:
        report.show_note("Ollama: not needed for this step; the saved poisoned passages match the questions.",
                         style="cyan")
        return saved.subset(questions)
    reason = ("you asked for new ones (--regenerate-poisons)" if args.regenerate_poisons
              else f"{len(todo)} of {len(questions)} questions have no saved poisoned passages yet")
    problem = client.problem(args.model)
    if problem:
        report.show_note(f"Ollama: needed because {reason}, but it isn't available, so only S0 is built. {problem}",
                         style="red")
        return None
    report.show_note(f"Ollama: in use, because {reason}. {args.model} at {client.url} is writing poisoned "
                     f"passages for those questions; the others keep their saved passages.", style="green")
    start = time.perf_counter()
    write = ollama_writer(client, args.model)
    poisons = saved.without(todo)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    for q in track(todo, description=f"Writing poisoned passages with {args.model}"):
        poisons = poisons.merge(PoisonSet.generate([q], write))
        poisons.save(POISONS_PATH)  # after every question, so an interruption keeps the work already done
    report.show_note(f"Ollama: done. {args.model} wrote poisoned passages for {len(todo)} questions in "
                     f"{(time.perf_counter() - start) / 60:.1f} minutes.", style="green")
    return poisons.subset(questions)


def knowledge_bases(corpus: Corpus, question_set: QuestionSet, poisons: PoisonSet | None, manifest: HashManifest,
                    report: ConsoleReport) -> list[KnowledgeBase]:
    """S0, plus every poisoned scenario and count the saved passages allow; each shown with its composition."""
    bases = [KnowledgeBase(corpus, question_set)]
    if poisons is not None:
        report.show_table("Poisoned passages", poisons.to_frame().head(5),
                          f"{len(poisons)} passages, {poisons.per_question} per question. Kept private "
                          f"(git-ignored). Review them before running the experiment. First 5:")
        bases += [KnowledgeBase(corpus, question_set, poisons, scenario, n)
                  for scenario in POISONED_SCENARIOS for n in POISON_COUNTS if n <= poisons.per_question]
    report.show_table("Knowledge bases", pd.DataFrame([kb.composition(manifest) for kb in bases]),
                      "  ".join(f"{name}: {description}." for name, description in SCENARIOS.items())
                      + " In a targeted attack (the default), each question is answered from the legitimate passages "
                        "plus only its own poisoned passages.")
    return bases


def run_experiment(study: Study, args: argparse.Namespace, client: OllamaClient, report: ConsoleReport,
                   question_set: QuestionSet, bases: list[KnowledgeBase], manifest: HashManifest) -> list[Path]:
    """Answers the chosen questions with every system in every condition, for each attack, and reports and saves the
    results."""
    splits = splits_for(study, args.split)
    if not splits:
        return []
    if len(bases) < 1 + len(POISONED_SCENARIOS) * len(POISON_COUNTS):
        report.show_note("Experiment: not run, because the poisoned passages are missing.", style="red")
        return []
    attack = args.attack or study.attack
    attacks = list(ATTACKS) if attack == "both" else [attack]
    report.show_note(f"Ollama: in use. {args.model} answers the {' and '.join(splits)} questions "
                     f"({' and '.join(attacks)} attacks) and {args.embedding_model} ranks the passages"
                     f"{' together with BM25 keyword search' if study.hybrid else ''}, at {client.url}.", style="green")
    embed = lambda texts: client.embed(args.embedding_model, texts)
    retriever = HybridRetriever(embed) if study.hybrid else Retriever(embed)  # shared, so texts are embedded once
    generate = lambda prompt: client.generate(args.model, prompt, **ANSWER_SETTINGS)
    cache = ReplyCache(RESULTS_DIR / "replies.jsonl")  # shared by both studies; keyed by model and prompt
    results_dir = RESULTS_DIR / study.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        with report.console.status(f"Loading {args.model} and {args.embedding_model} on the server (the first "
                                   f"request can take a minute)..."):
            client.warm_up(args.model, embedding=[args.embedding_model])
        if study.hybrid:
            ranks = retrieval_comparison(question_set, bases[0], manifest, retriever, splits)
            saved.append(save_csv(ranks, results_dir / f"retrieval_{'_'.join(splits)}.csv"))
            report.show_table("Retrieval: embedding only vs hybrid (no poisoning)", analysis.retrieval_summary(ranks),
                              "Where the answer passage ranks among the legitimate passages. The last row is the "
                              "reserved-slot defense's separate search of verified passages, whose top 3 always reach "
                              "the model.")
        for attack in attacks:
            start = time.perf_counter()
            experiment = Experiment(question_set, bases, manifest, retriever, generate, args.model, cache,
                                    systems=study.systems, attack=attack)
            trials = experiment.run(splits, progress=track)
            report.show_note(f"Ollama: done with the {attack} attack. {len(trials)} answers, {experiment.new_calls} "
                             f"new and {len(trials) - experiment.new_calls} reused from earlier runs, in "
                             f"{(time.perf_counter() - start) / 60:.1f} minutes.", style="green")
            saved += report_results(study, report, trials, splits, attack, results_dir)
    except OllamaUnavailable as error:
        report.show_note(f"Ollama: stopped. {error}", style="red")
    return saved


def report_results(study: Study, report: ConsoleReport, trials: pd.DataFrame, splits: list[str], attack: str,
                   results_dir: Path) -> list[Path]:
    """Shows and saves one attack's results. The stress test repeats S0 and closed-book unchanged, so the S0 table,
    the trade-off and the success criteria are shown for the targeted attack only."""
    saved = []
    for split in splits:
        saved += report_split(study, report, trials[trials.split == split], split, attack, results_dir)
    if attack == "targeted" and study.criteria.split in splits:
        report_criteria(study, report, trials)
    return saved


def report_split(study: Study, report: ConsoleReport, part: pd.DataFrame, split: str, attack: str,
                 results_dir: Path) -> list[Path]:
    """Shows and saves one question set's results for one attack."""
    targeted = attack == "targeted"
    title, suffix = ATTACK_LABELS[attack]
    table = analysis.summary(part)
    saved = [save_csv(part, results_dir / f"trials_{split}{suffix}.csv"),
             save_csv(table, results_dir / f"summary_{split}{suffix}.csv")]
    report.show_table(f"Results: {split} set, {title}", analysis.results_table(table),
                      f"{part.qid.nunique()} questions. Attack success: the answer contains the poisoned target. "
                      f"Closed-book: no passages, so the model answers from memory. {study.systems_note} "
                      + ("Each question is answered from the legitimate passages plus its own poisoned passages."
                         if targeted else "Every question's poisoned passages share one knowledge base."))
    if targeted:
        report.show_table(f"S0 accuracy by answer passage: {split} set", analysis.s0_by_group(part),
                          "Correct answers with no poisoning. Held-out answer passages are legitimate but "
                          "unverified, so they show what ranking unverified content lower costs.")
    tiered = analysis.tiered_table(part)
    if not tiered.empty:
        report.show_table(f"Tiered answering: {split} set, {title}", tiered,
                          "Verified: answered from the top 5 verified passages alone. Fallback: the model replied "
                          "NOT FOUND (or gave no answer), so it was asked again with the regular top 5, verified "
                          "or not, and the answer is flagged as unverified for the analyst.")
    for base, other in study.pairs:
        comparison = analysis.comparisons(part, base, other)
        comparison["p (exact)"] = comparison["p (exact)"].map(lambda p: f"{p:.3f}")
        report.show_table(f"{base.capitalize()} vs {other}: {split} set, {title}", comparison,
                          "McNemar's exact test on the same questions. The \"only\" columns count questions where "
                          "just that system had the outcome.")
    return saved


def report_criteria(study: Study, report: ConsoleReport, trials: pd.DataFrame) -> None:
    """The trade-off analysis (if its systems ran) and the success criteria, on the targeted attack's trials."""
    criteria = study.criteria
    if any(system in TRADEOFF_SYSTEMS for system in study.systems):
        report.show_table(f"Trade-off: how many of the {CONTEXT_SIZE} slots go to verified passages "
                          f"({criteria.split} set, exploratory)",
                          analysis.slot_tradeoff(trials, criteria.split),
                          f"Every setting is reported; the success criteria apply only to the design fixed in "
                          f"advance ({RESERVED_SLOTS} of {CONTEXT_SIZE}). More verified slots let fewer poisoned "
                          f"passages reach the model but leave less room for correct passages missing from the "
                          f"manifest. ASR, n: attack success with n poisoned passages per question (S1 and S2 "
                          f"together). S0: correct answers without poisoning.")
    report.show_table(f"Success criteria ({criteria.split} set, targeted attack)",
                      analysis.check_criteria(trials, criteria),
                      f"Fixed before this study's answers were generated: {criteria.defense} vs baseline.")


def save_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    main()
