"""Build a QASPER-derived corpus + eval set for the RAPTOR benchmark.

Downloads `allenai/qasper` via Hugging Face `datasets`, subsamples papers
stratified by answer type, writes one markdown per paper to
`data/corpus_qasper/`, and writes a YAML eval set to
`data/eval_set_qasper.yaml` matching the schema of the hand-authored
`data/eval_set.yaml`.

This is a one-shot setup script — outputs are committed (or .gitignored)
and the benchmark reads them directly.

Run:
    python scripts/build_qasper_evalset.py
    python scripts/build_qasper_evalset.py --n-papers 30 --questions-per-paper 2
"""

from __future__ import annotations

import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import click
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


CATEGORY_BY_ANSWER_TYPE = {
    "extractive": "extractive_evidence",
    "abstractive": "abstractive_synthesis",
    "yes_no": "yes_no",
    "unanswerable": "unanswerable",
    "boolean": "yes_no",
    "none": "unanswerable",
}

DIFFICULTY_BY_ANSWER_TYPE = {
    "extractive": "medium",
    "abstractive": "hard",
    "yes_no": "easy",
    "unanswerable": "hard",
    "boolean": "easy",
    "none": "hard",
}


def _safe_filename(text: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return cleaned[:max_len] or "untitled"


def _classify_answer(answer: dict) -> tuple[str, str]:
    """Return (answer_text, answer_type) from a QASPER answer dict.

    QASPER answer schema:
      - extractive_spans: list[str]
      - free_form_answer: str
      - yes_no: bool | None
      - unanswerable: bool
    """
    if answer.get("unanswerable"):
        return ("Unanswerable", "unanswerable")
    yn = answer.get("yes_no")
    if yn is not None:
        return ("Yes" if yn else "No", "yes_no")
    free = (answer.get("free_form_answer") or "").strip()
    if free:
        return (free, "abstractive")
    spans = answer.get("extractive_spans") or []
    if spans:
        return (" ".join(s.strip() for s in spans), "extractive")
    return ("Unanswerable", "unanswerable")


def _paper_to_markdown(paper: dict) -> str:
    title = paper.get("title", "Untitled")
    abstract = paper.get("abstract", "")
    out = [f"# {title}\n"]
    if abstract:
        out.append("## Abstract\n")
        out.append(abstract.strip() + "\n")
    full_text = paper.get("full_text", {})
    section_names = full_text.get("section_name") or []
    paragraphs_per_section = full_text.get("paragraphs") or []
    for name, paragraphs in zip(section_names, paragraphs_per_section):
        name = (name or "").strip() or "Section"
        out.append(f"\n## {name}\n")
        for para in paragraphs:
            text = (para or "").strip()
            if text:
                out.append(text + "\n")
    return "\n".join(out)


def _collect_answers(ans_block: dict) -> tuple[str, list[str], str, list[str]] | None:
    """Aggregate one question's per-annotator answers into a gold-answer set.

    Returns (chosen_answer, gold_answers, answer_type, spans), or None if
    `ans_block` has no annotator answer entries at all (question should be
    skipped). `chosen_answer` is the first usable (non-"Unanswerable")
    annotator answer, used for embedding-similarity scoring. `gold_answers`
    collects EVERY usable annotator answer, deduplicated and order-preserved
    — QASPER questions typically have multiple independently-written
    reference answers, and scoring against only the first discards
    legitimate paraphrases that QasperF1Scorer's max-over-references is
    designed to credit. If no annotator answer is usable, both fall back to
    a single "Unanswerable" entry.
    """
    answer_entries = ans_block.get("answer", [])
    if not answer_entries:
        return None

    usable: list[tuple[str, str, list[str]]] = []
    fallback_atype = None
    for ans in answer_entries:
        text, atype = _classify_answer(ans)
        if text and text != "Unanswerable":
            usable.append((text, atype, ans.get("extractive_spans") or []))
        elif fallback_atype is None:
            fallback_atype = atype or "unanswerable"

    if usable:
        seen: set[str] = set()
        gold_answers = []
        for text, _, _ in usable:
            if text not in seen:
                seen.add(text)
                gold_answers.append(text)
        chosen_answer, chosen_atype, spans = usable[0]
        return (chosen_answer, gold_answers, chosen_atype, spans)

    return ("Unanswerable", ["Unanswerable"], fallback_atype or "unanswerable", [])


def _extract_key_facts(answer_text: str, spans: list[str], max_facts: int = 4) -> list[str]:
    """Pull short snippets to use as KeyFactScorer match targets.

    Prefer spans (short, deterministic). Fall back to noun-phrase-ish tokens.
    """
    facts: list[str] = []
    for s in spans:
        s = s.strip()
        if 3 <= len(s) <= 120 and s.lower() not in {f.lower() for f in facts}:
            facts.append(s)
        if len(facts) >= max_facts:
            return facts
    # Fall back: try to pull numbers from answer_text
    nums = re.findall(r"-?\d+(?:[.,]\d+)?(?:%|\b)", answer_text)
    for n in nums:
        if n not in facts:
            facts.append(n)
        if len(facts) >= max_facts:
            break
    return facts


@click.command()
@click.option("--n-papers", default=30, show_default=True,
              help="Number of papers to subsample.")
@click.option("--questions-per-paper", default=2, show_default=True,
              help="Max questions to keep per paper.")
@click.option("--seed", default=42, show_default=True)
@click.option("--corpus-out", default="data/corpus_qasper", show_default=True)
@click.option("--eval-out", default="data/eval_set_qasper.yaml", show_default=True)
@click.option("--split", default="train", show_default=True,
              help="QASPER split: train / validation / test. Train is largest.")
def main(n_papers, questions_per_paper, seed, corpus_out, eval_out, split):
    """Build the QASPER-derived RAPTOR benchmark fixtures."""
    try:
        from datasets import load_dataset
    except ImportError:
        click.echo(
            "Error: `datasets` not installed. Install with: pip install datasets",
            err=True,
        )
        sys.exit(1)

    random.seed(seed)
    click.echo(f"Loading QASPER (split={split}) from Hugging Face...")
    ds = load_dataset("allenai/qasper", split=split)
    click.echo(f"  {len(ds)} papers in split.")

    # ---- Subsample papers stratified by available answer types ----
    candidates: list[tuple[dict, set[str]]] = []
    for paper in ds:
        types_present: set[str] = set()
        for q in (paper.get("qas") or {}).get("question") or []:
            pass  # placeholder; QASPER stores qas as dict-of-lists
        qas = paper.get("qas") or {}
        questions = qas.get("question") or []
        answers_all = qas.get("answers") or []
        for ans_block in answers_all:
            for ans in ans_block.get("answer", []):
                _, atype = _classify_answer(ans)
                types_present.add(atype)
        if questions and types_present:
            candidates.append((paper, types_present))

    click.echo(f"  {len(candidates)} candidate papers with usable QAs.")
    # Prefer papers with broader type coverage
    candidates.sort(key=lambda x: -len(x[1]))
    # Take top N (deterministic given seed via stable sort + later shuffle)
    chosen = candidates[: max(n_papers * 2, n_papers)]
    random.shuffle(chosen)
    chosen = chosen[:n_papers]
    click.echo(f"  Chose {len(chosen)} papers.")

    # ---- Write per-paper markdown + collect QAs ----
    corpus_dir = Path(corpus_out)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale files so re-runs are deterministic
    for old in corpus_dir.glob("*.md"):
        old.unlink()

    eval_records: list[dict] = []
    type_counts: dict[str, int] = defaultdict(int)
    id_counter = 0

    for paper, _types in chosen:
        title = paper.get("title", "Untitled")
        paper_id = paper.get("id") or _safe_filename(title)
        filename = f"{_safe_filename(paper_id)}.md"
        rel_path = filename  # corpus root is corpus_qasper itself
        md = _paper_to_markdown(paper)
        (corpus_dir / filename).write_text(md, encoding="utf-8")

        qas = paper.get("qas") or {}
        questions = qas.get("question") or []
        answers_all = qas.get("answers") or []
        if not questions:
            continue

        # Group answers per question; collect every usable annotator answer.
        per_question_records: list[dict] = []
        for q_text, ans_block in zip(questions, answers_all):
            collected = _collect_answers(ans_block)
            if collected is None:
                continue
            chosen_answer, gold_answers, atype, spans = collected

            id_counter += 1
            prefix = {
                "extractive": "ex",
                "abstractive": "ab",
                "yes_no": "yn",
                "boolean": "yn",
                "unanswerable": "un",
                "none": "un",
            }.get(atype, "qa")
            record = {
                "id": f"qa_{prefix}_{id_counter:04d}",
                "question": q_text.strip(),
                "gold_answer": chosen_answer,
                "gold_answers": gold_answers,
                "gold_source_ids": [rel_path],
                "key_facts": _extract_key_facts(chosen_answer, spans),
                "category": CATEGORY_BY_ANSWER_TYPE.get(atype, "extractive_evidence"),
                "difficulty": DIFFICULTY_BY_ANSWER_TYPE.get(atype, "medium"),
            }
            per_question_records.append(record)
            type_counts[atype] += 1

        random.shuffle(per_question_records)
        eval_records.extend(per_question_records[:questions_per_paper])

    click.echo(f"  Collected {len(eval_records)} questions total.")
    for k, v in sorted(type_counts.items()):
        click.echo(f"    {k}: {v}")

    # ---- Write eval YAML ----
    eval_path = Path(eval_out)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# QASPER-derived eval set (generated by scripts/build_qasper_evalset.py)\n"
        f"# {len(eval_records)} questions over {len(chosen)} papers.\n"
        "# Schema matches data/eval_set.yaml so the existing BenchmarkRunner can consume it.\n\n"
    )
    with eval_path.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(eval_records, f, sort_keys=False, allow_unicode=True)

    click.echo(f"\nWrote corpus to: {corpus_dir}")
    click.echo(f"Wrote eval set to: {eval_path}")
    click.echo(
        "\nNext: run the RAPTOR benchmark over this corpus, e.g.\n"
        f"  python scripts/run_benchmark.py --include-raptor "
        f"--corpus-dir {corpus_out} --eval-set {eval_out}"
    )


if __name__ == "__main__":
    main()
