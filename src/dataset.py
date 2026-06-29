from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

try:
    from src.utils import LABELS, read_jsonl, token_count, write_json, write_jsonl
except ModuleNotFoundError:  # direct execution: python src/dataset.py
    from utils import LABELS, read_jsonl, token_count, write_json, write_jsonl


@dataclass(frozen=True)
class ExtractionConfig:
    min_tokens: int = 30
    max_tokens: int = 300
    max_spans_per_paper_per_label: int = 12
    seed: int = 42


SECTION_PATTERNS = {
    "abstract": (r"abstract",),
    "introduction": (r"intro(?:duction)?", r"background and introduction"),
    "conclusion": (r"conclu", r"discussion and conclusion", r"summary and future"),
    "methods": (r"method", r"approach", r"model", r"architecture", r"implementation", r"algorithm"),
    "results": (r"result", r"experiment", r"analysis", r"finding"),
    "evaluation": (r"evaluat", r"benchmark", r"ablation"),
}


def section_family(name: str) -> str:
    normalized = re.sub(r"[^a-z]+", " ", str(name).lower()).strip()
    for family, patterns in SECTION_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return family
    return "other"


def first_sentence(text: str) -> str:
    pieces = re.split(r"(?<=[.!?])\s+", str(text).strip(), maxsplit=1)
    return pieces[0].strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _candidate(
    paper: dict[str, Any], section: str, paragraph_index: int, label: str,
    text: str, rule: str, family: str,
) -> dict[str, Any]:
    return {
        "paper_id": str(paper["paper_id"]),
        "domain": str(paper.get("domain", "NLP")),
        "section_name": section,
        "section_family": family,
        "paragraph_index": paragraph_index,
        "label": label,
        "label_source_rule": rule,
        "text": str(text).strip(),
        "token_count": token_count(text),
    }


def extract_from_paper(
    paper: dict[str, Any], config: ExtractionConfig = ExtractionConfig()
) -> tuple[list[dict[str, Any]], Counter]:
    """Apply rule-based weak supervision derived solely from document structure."""
    candidates: list[dict[str, Any]] = []
    stats: Counter = Counter()
    if paper.get("title"):
        candidates.append(_candidate(paper, "Title", -1, "macro", paper["title"], "macro_title", "other"))
    if paper.get("abstract"):
        abstract = paper["abstract"]
        if isinstance(abstract, list):
            abstract = " ".join(map(str, abstract))
        candidates.append(_candidate(paper, "Abstract", -1, "macro", abstract, "macro_abstract", "abstract"))

    for section in paper.get("sections", []):
        name = str(section.get("name", "Unknown"))
        family = section_family(name)
        paragraphs = section.get("paragraphs", []) or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        if family == "abstract":
            stats["dropped_duplicate_abstract_section"] += sum(bool(str(p).strip()) for p in paragraphs)
            continue
        for index, paragraph in enumerate(paragraphs):
            paragraph = str(paragraph).strip()
            if not paragraph:
                continue
            if family == "introduction" and index < 2:
                candidates.append(_candidate(paper, name, index, "macro", paragraph, "macro_intro_first2", family))
            elif family == "conclusion":
                candidates.append(_candidate(paper, name, index, "macro", paragraph, "macro_conclusion", family))
            elif family not in {"introduction", "conclusion"} and index == 0:
                candidates.append(_candidate(paper, name, index, "meso", first_sentence(paragraph), "meso_section_lead", family))
            elif family in {"methods", "results", "evaluation"} and index > 0:
                rule_family = "results" if family == "evaluation" else family
                candidates.append(_candidate(paper, name, index, "micro", paragraph, f"micro_{rule_family}_nonlead", family))
            else:
                stats["dropped_no_rule"] += 1

    accepted = []
    for row in candidates:
        if config.min_tokens <= row["token_count"] <= config.max_tokens:
            accepted.append(row)
        else:
            stats["dropped_token_filter"] += 1
    rng = random.Random(config.seed)
    rng.shuffle(accepted)
    per_label: Counter = Counter()
    capped = []
    for row in accepted:
        if per_label[row["label"]] < config.max_spans_per_paper_per_label:
            capped.append(row)
            per_label[row["label"]] += 1
        else:
            stats["dropped_paper_cap"] += 1
    return capped, stats


def normalize_qasper(row: dict[str, Any], domain: str = "NLP") -> dict[str, Any]:
    full_text = row.get("full_text", {}) or {}
    names = _as_list(full_text.get("section_name"))
    paragraphs = _as_list(full_text.get("paragraphs"))
    sections = []
    for name, paras in zip(names, paragraphs):
        sections.append({"name": name, "paragraphs": _as_list(paras)})
    return {
        "paper_id": row.get("id") or row.get("paper_id") or row.get("title"),
        "domain": row.get("domain", domain),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "sections": sections,
    }




def iter_qasper(split: str = "train", domain: str = "NLP") -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to load QASPER") from exc
    dataset = load_dataset("allenai/qasper", split=split)
    for row in dataset:
        yield normalize_qasper(dict(row), domain=domain)


def iter_qasper_parquet(path_or_url: str | Path, domain: str = "NLP") -> Iterator[dict[str, Any]]:
    """Load an official auto-converted QASPER Parquet file without datasets."""
    try:
        frame = pd.read_parquet(path_or_url)
    except ImportError as exc:
        raise RuntimeError("Install pyarrow to load QASPER Parquet files") from exc
    for row in frame.to_dict(orient="records"):
        yield normalize_qasper(row, domain=domain)


def balance_spans(rows: Iterable[dict[str, Any]], max_per_paper_per_label: int = 12, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["paper_id"]), str(row["label"]))].append(dict(row))
    capped = []
    for values in grouped.values():
        rng.shuffle(values)
        capped.extend(values[:max_per_paper_per_label])
    by_label = {label: [r for r in capped if r["label"] == label] for label in LABELS}
    if any(not values for values in by_label.values()):
        missing = [label for label, values in by_label.items() if not values]
        raise ValueError(f"No spans extracted for classes: {missing}")
    target = min(map(len, by_label.values()))
    balanced = []
    for label in LABELS:
        rng.shuffle(by_label[label])
        balanced.extend(by_label[label][:target])
    rng.shuffle(balanced)
    return balanced


def validate_corpus(rows: Iterable[dict[str, Any]], min_per_class: int = 500, min_papers: int = 50, balance_tolerance: float = 0.10) -> dict[str, Any]:
    rows = list(rows)
    counts = Counter(r["label"] for r in rows)
    if any(counts[label] < min_per_class for label in LABELS):
        raise ValueError(f"Need at least {min_per_class} spans per class; got {dict(counts)}")
    papers = {r["paper_id"] for r in rows}
    if len(papers) < min_papers:
        raise ValueError(f"Need at least {min_papers} papers; got {len(papers)}")
    values = [counts[label] for label in LABELS]
    if max(values) > min(values) * (1 + balance_tolerance):
        raise ValueError(f"Class balance exceeds {balance_tolerance:.0%}: {dict(counts)}")
    return {"class_counts": dict(counts), "paper_count": len(papers)}


def make_paper_splits(rows: Iterable[dict[str, Any]], output_path: str | Path | None = None, seed: int = 42) -> dict[str, list[str]]:
    """Make deterministic 70/10/20 paper splits, stratified by domain."""
    domain_papers: dict[str, set[str]] = defaultdict(set)
    paper_domains: dict[str, str] = {}
    for row in rows:
        paper_id = str(row["paper_id"])
        domain = str(row.get("domain", "unknown"))
        if paper_id in paper_domains and paper_domains[paper_id] != domain:
            raise ValueError(f"Paper {paper_id!r} occurs in multiple domains")
        paper_domains[paper_id] = domain
        domain_papers[domain].add(paper_id)
    result = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for domain in sorted(domain_papers):
        papers = sorted(domain_papers[domain])
        rng.shuffle(papers)
        n = len(papers)
        n_train = int(n * 0.70)
        n_val = int(n * 0.10)
        result["train"].extend(papers[:n_train])
        result["validation"].extend(papers[n_train:n_train + n_val])
        result["test"].extend(papers[n_train + n_val:])
    for split in result:
        result[split] = sorted(result[split])
    if output_path:
        write_json(output_path, result)
    return result


def extraction_summary(rows: list[dict[str, Any]], stats: Counter) -> dict[str, Any]:
    return {
        "counts_by_label_source_rule": dict(Counter(r["label_source_rule"] for r in rows)),
        "counts_by_section_family": dict(Counter(r["section_family"] for r in rows)),
        "counts_by_label": dict(Counter(r["label"] for r in rows)),
        "papers_contributing_to_each_class": {
            label: len({r["paper_id"] for r in rows if r["label"] == label}) for label in LABELS
        },
        "dropped_no_rule": stats["dropped_no_rule"],
        "dropped_token_filter": stats["dropped_token_filter"],
        "dropped_paper_cap": stats["dropped_paper_cap"],
        "dropped_class_balance": stats["dropped_class_balance"],
        "dropped_duplicate_abstract_section": stats["dropped_duplicate_abstract_section"],
        "total_spans_saved": len(rows),
        "total_papers": len({r["paper_id"] for r in rows}),
    }


def build_dataset(papers: Iterable[dict[str, Any]], config: ExtractionConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for paper in papers:
        extracted, paper_stats = extract_from_paper(paper, config)
        rows.extend(extracted)
        stats.update(paper_stats)
    balanced = balance_spans(rows, config.max_spans_per_paper_per_label, config.seed)
    stats["dropped_class_balance"] = len(rows) - len(balanced)
    for row_id, row in enumerate(balanced):
        row["row_id"] = row_id
    return balanced, extraction_summary(balanced, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract rule-based weak SLoD labels from structured papers")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--input-jsonl", help="Optional normalized paper JSONL; otherwise downloads QASPER")
    sources.add_argument("--qasper-parquet", help="Local path or URL for official QASPER Parquet")
    parser.add_argument("--qasper-split", default="train")
    parser.add_argument("--domain", default="NLP", help="Domain assigned to QASPER rows lacking domain metadata")
    parser.add_argument("--output", default="data/spans/spans.jsonl")
    parser.add_argument("--summary", default="results/extraction_summary.json")
    parser.add_argument("--splits", default="data/spans/paper_splits.json")
    parser.add_argument("--min-tokens", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-spans-per-paper-per-label", type=int, default=12)
    parser.add_argument("--min-per-class", type=int, default=500)
    parser.add_argument("--min-papers", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-small", action="store_true", help="Development only: skip corpus-size validation")
    args = parser.parse_args()
    config = ExtractionConfig(args.min_tokens, args.max_tokens, args.max_spans_per_paper_per_label, args.seed)
    if args.input_jsonl:
        papers = read_jsonl(args.input_jsonl)
    elif args.qasper_parquet:
        papers = iter_qasper_parquet(args.qasper_parquet, args.domain)
    else:
        papers = iter_qasper(args.qasper_split, args.domain)
    rows, summary = build_dataset(papers, config)
    if not args.allow_small:
        summary["validation"] = validate_corpus(rows, args.min_per_class, args.min_papers)
    write_jsonl(args.output, rows)
    write_json(args.summary, summary)
    make_paper_splits(rows, args.splits, args.seed)
    pd.DataFrame(rows).to_csv(Path(args.output).with_suffix(".csv"), index=False)
    print(f"Saved {len(rows)} balanced spans to {args.output}")


if __name__ == "__main__":
    main()
