from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

try:
    from src.dataset import ExtractionConfig, extract_from_paper, extraction_summary
    from src.utils import LABELS, write_json, write_jsonl
except ModuleNotFoundError:  # direct execution: python src/biomedicine.py
    from dataset import ExtractionConfig, extract_from_paper, extraction_summary
    from utils import LABELS, write_json, write_jsonl


EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
BIOC_PMC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
DEFAULT_QUERY = 'OPEN_ACCESS:Y AND HAS_FT:Y AND SRC:MED AND PUB_TYPE:"research article"'
EXCLUDED_SECTION_TYPES = {"TITLE", "ABSTRACT", "FIG", "TABLE", "REF", "SUPPL", "AUTH_CONT"}
SECTION_TYPE_NAMES = {
    "INTRO": "Introduction",
    "METHODS": "Methods",
    "RESULTS": "Results",
    "DISCUSS": "Discussion",
    "CONCL": "Conclusion",
}


def _get_json(url: str, timeout: int = 60, retries: int = 3) -> Any:
    request = Request(url, headers={"User-Agent": "slod-probe/1.0 (research prototype)"})
    error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # network errors are retried and reported by the caller
            error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {error}")


def normalize_bioc_collection(payload: dict[str, Any] | list[dict[str, Any]], domain: str = "biomedicine") -> dict[str, Any]:
    """Convert one PMC BioC collection to the normalized paper schema used by dataset.py."""
    if isinstance(payload, list):
        if not payload:
            raise ValueError("BioC payload contains no collections")
        payload = payload[0]
    documents = payload.get("documents") or []
    if not documents:
        raise ValueError("BioC payload contains no documents")
    document = documents[0]
    title = ""
    abstract_parts: list[str] = []
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for passage in document.get("passages", []):
        infons = passage.get("infons") or {}
        passage_type = str(infons.get("type", "")).lower()
        section_type = str(infons.get("section_type", "")).upper()
        text = str(passage.get("text") or "").strip()
        if not text:
            continue
        if section_type == "TITLE" and passage_type == "front":
            title = text
        elif section_type == "ABSTRACT" and passage_type == "abstract":
            abstract_parts.append(text)
        elif passage_type.startswith("title") and section_type not in EXCLUDED_SECTION_TYPES:
            current = {"name": text, "paragraphs": [], "bioc_section_type": section_type}
            sections.append(current)
        elif passage_type == "paragraph" and section_type not in EXCLUDED_SECTION_TYPES:
            if current is None or current.get("bioc_section_type") != section_type:
                current = {
                    "name": SECTION_TYPE_NAMES.get(section_type, section_type.title() or "Other"),
                    "paragraphs": [],
                    "bioc_section_type": section_type,
                }
                sections.append(current)
            current["paragraphs"].append(text)

    sections = [{"name": section["name"], "paragraphs": section["paragraphs"]} for section in sections if section["paragraphs"]]
    return {
        "paper_id": str(document.get("id") or ""),
        "domain": domain,
        "title": title,
        "abstract": " ".join(abstract_parts),
        "sections": sections,
    }


def select_external_test_spans(
    rows: Iterable[dict[str, Any]], per_class: int = 500,
    max_per_paper_per_label: int = 5, seed: int = 42,
) -> list[dict[str, Any]]:
    """Create an exactly balanced external test set without fitting on its labels."""
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["paper_id"]), str(row["label"]))].append(dict(row))
    capped: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = grouped[key]
        rng.shuffle(values)
        capped.extend(values[:max_per_paper_per_label])
    selected: list[dict[str, Any]] = []
    for label in LABELS:
        candidates = [row for row in capped if row["label"] == label]
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise ValueError(f"Need {per_class} external {label} spans; got {len(candidates)}")
        selected.extend(candidates[:per_class])
    rng.shuffle(selected)
    for row_id, row in enumerate(selected):
        row["row_id"] = row_id
    return selected


def search_pmcids(query: str = DEFAULT_QUERY, page_size: int = 1000) -> list[str]:
    params = urlencode({"query": query, "format": "json", "pageSize": page_size, "resultType": "core"})
    payload = _get_json(f"{EUROPE_PMC_SEARCH}?{params}")
    return [str(row["pmcid"]) for row in payload.get("resultList", {}).get("result", []) if row.get("pmcid")]


def collect_biomedical_spans(
    pmcids: Iterable[str], per_class: int = 500, min_papers: int = 100,
    config: ExtractionConfig = ExtractionConfig(max_spans_per_paper_per_label=5),
    request_delay: float = 0.34,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    failures: list[dict[str, str]] = []
    fetched = 0
    for pmcid in pmcids:
        try:
            payload = _get_json(BIOC_PMC.format(pmcid=pmcid))
            paper = normalize_bioc_collection(payload)
            extracted, paper_stats = extract_from_paper(paper, config)
            rows.extend(extracted)
            stats.update(paper_stats)
            fetched += 1
        except Exception as exc:
            failures.append({"paper_id": str(pmcid), "error": str(exc)})
        counts = Counter(row["label"] for row in rows)
        contributors = {row["paper_id"] for row in rows}
        if len(contributors) >= min_papers and all(counts[label] >= per_class for label in LABELS):
            break
        if request_delay:
            time.sleep(request_delay)

    selected = select_external_test_spans(rows, per_class, config.max_spans_per_paper_per_label, config.seed)
    summary = extraction_summary(selected, stats)
    summary.update({
        "source": "NCBI PMC Open Access BioC API",
        "domain": "biomedicine",
        "purpose": "external_cross_domain_test_only",
        "access_date": date.today().isoformat(),
        "papers_fetched_successfully": fetched,
        "failed_papers": failures,
        "target_per_class": per_class,
        "max_spans_per_paper_per_label": config.max_spans_per_paper_per_label,
    })
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a separate PMC BioC biomedical external test set")
    parser.add_argument("--output", default="data/spans/pmc_biomedicine_spans.jsonl")
    parser.add_argument("--summary", default="results/pmc_biomedicine_extraction_summary.json")
    parser.add_argument("--manifest", default="data/spans/pmc_biomedicine_manifest.json")
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--min-papers", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--max-spans-per-paper-per-label", type=int, default=5)
    parser.add_argument("--min-tokens", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--request-delay", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pmcids = search_pmcids(page_size=args.candidate_limit)
    config = ExtractionConfig(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_spans_per_paper_per_label=args.max_spans_per_paper_per_label,
        seed=args.seed,
    )
    rows, summary = collect_biomedical_spans(
        pmcids, args.per_class, args.min_papers, config, args.request_delay,
    )
    write_jsonl(args.output, rows)
    pd.DataFrame(rows).to_csv(Path(args.output).with_suffix(".csv"), index=False)
    write_json(args.summary, summary)
    write_json(args.manifest, {
        "source": "NCBI PMC Open Access BioC API",
        "search_endpoint": EUROPE_PMC_SEARCH,
        "query": DEFAULT_QUERY,
        "access_date": summary["access_date"],
        "selected_paper_ids": sorted({row["paper_id"] for row in rows}),
    })
    print(f"Saved {len(rows)} biomedical external-test spans from {summary['total_papers']} papers")


if __name__ == "__main__":
    main()
