import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.controls import make_length_controlled
from src.dataset import (
    ExtractionConfig,
    balance_spans,
    extract_from_paper,
    make_paper_splits,
    validate_corpus,
)
from src.probe import evaluate_predictions, run_section_name_baseline


def words(n: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def sample_paper(paper_id="p1", domain="NLP"):
    return {
        "paper_id": paper_id,
        "domain": domain,
        "title": words(35, "title"),
        "abstract": words(50, "abstract"),
        "sections": [
            {"name": "Introduction", "paragraphs": [words(45, "intro"), words(46, "intro"), words(40)]},
            {"name": "Methods", "paragraphs": [words(45, "lead") + ". " + words(35), words(60, "method")]},
            {"name": "Results and Evaluation", "paragraphs": [words(40, "lead") + ". " + words(35), words(55, "result")]},
            {"name": "Related Work", "paragraphs": [words(40, "related") + ". " + words(35)]},
            {"name": "Conclusion", "paragraphs": [words(48, "conclusion")]},
        ],
    }


def test_weak_rules_emit_metadata_and_count_drops():
    spans, stats = extract_from_paper(sample_paper(), ExtractionConfig(min_tokens=30, max_tokens=300))
    rules = {s["label_source_rule"] for s in spans}
    assert {"macro_title", "macro_abstract", "macro_intro_first2", "macro_conclusion"} <= rules
    assert {"meso_section_lead", "micro_methods_nonlead", "micro_results_nonlead"} <= rules
    assert all({"paper_id", "domain", "section_name", "section_family", "paragraph_index", "label", "text", "token_count", "label_source_rule"} <= s.keys() for s in spans)
    assert stats["dropped_no_rule"] >= 1


def test_full_text_abstract_is_not_relabelled_as_meso():
    paper = sample_paper()
    paper["sections"].insert(0, {"name": "Abstract", "paragraphs": [words(45, "duplicate")]})
    spans, stats = extract_from_paper(paper, ExtractionConfig(min_tokens=30, max_tokens=300))
    assert not any(span["section_family"] == "abstract" and span["label"] == "meso" for span in spans)
    assert stats["dropped_duplicate_abstract_section"] == 1


def test_per_paper_cap_and_balancing():
    rows = []
    for label, n in (("macro", 12), ("meso", 11), ("micro", 10)):
        rows.extend({"paper_id": f"p{i % 3}", "label": label, "text": str(i)} for i in range(n))
    balanced = balance_spans(rows, max_per_paper_per_label=2, seed=7)
    counts = pd.DataFrame(balanced).groupby("label").size()
    assert counts.nunique() == 1
    assert pd.DataFrame(balanced).groupby(["paper_id", "label"]).size().max() <= 2


def test_paper_split_is_disjoint_reproducible_and_saved(tmp_path: Path):
    rows = [{"paper_id": f"p{i}", "domain": "NLP", "label": "macro"} for i in range(20)]
    out = tmp_path / "splits.json"
    first = make_paper_splits(rows, output_path=out, seed=11)
    second = make_paper_splits(rows, seed=11)
    assert first == second == json.loads(out.read_text())
    sets = [set(first[k]) for k in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert [len(x) for x in sets] == [14, 2, 4]


def test_length_control_is_separate_and_fixed_range():
    original = pd.DataFrame({"text": [words(80), words(140)], "token_count": [80, 140], "label": ["macro", "micro"]})
    controlled = make_length_controlled(original, min_tokens=60, max_tokens=70, target_tokens=65, seed=1)
    assert original.token_count.tolist() == [80, 140]
    assert controlled.token_count.tolist() == [65, 65]
    assert controlled.controlled_original_token_count.tolist() == [80, 140]


def test_metrics_and_section_baseline_share_expected_schema():
    labels = ["macro", "meso", "micro"]
    metrics = evaluate_predictions(labels, labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 3
    train = pd.DataFrame({"section_name": ["Abstract", "Methods", "Results"] * 3, "label": labels * 3})
    test = train.copy()
    baseline = run_section_name_baseline(train, test, labels)
    assert baseline["accuracy"] == 1.0
    assert set(baseline) >= {"accuracy", "macro_f1", "per_class", "confusion_matrix"}


def test_validation_rejects_small_or_unbalanced_corpus():
    rows = [{"paper_id": "p1", "label": "macro"}] * 3
    with pytest.raises(ValueError, match="at least"):
        validate_corpus(rows, min_per_class=2, min_papers=2, balance_tolerance=0.10)
