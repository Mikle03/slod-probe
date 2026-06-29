"""Dependency-light smoke test for constrained/offline build environments."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from src.controls import make_length_controlled
from src.dataset import ExtractionConfig, balance_spans, extract_from_paper, make_paper_splits


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


paper = {
    "paper_id": "p1", "domain": "NLP", "title": words(35), "abstract": words(50),
    "sections": [
        {"name": "Introduction", "paragraphs": [words(40), words(41), words(42)]},
        {"name": "Methods", "paragraphs": [words(35) + ". " + words(35), words(60)]},
        {"name": "Results", "paragraphs": [words(35) + ". " + words(35), words(60)]},
        {"name": "Conclusion", "paragraphs": [words(45)]},
    ],
}
rows, stats = extract_from_paper(paper, ExtractionConfig())
rules = {row["label_source_rule"] for row in rows}
assert {"macro_abstract", "macro_intro_first2", "meso_section_lead", "micro_methods_nonlead", "micro_results_nonlead"} <= rules
assert stats["dropped_no_rule"] == 1

balanced_input = []
for label in ("macro", "meso", "micro"):
    balanced_input += [{"paper_id": f"p{i % 3}", "label": label} for i in range(9)]
balanced = balance_spans(balanced_input, max_per_paper_per_label=2)
assert len(balanced) == 18

with TemporaryDirectory() as directory:
    split_rows = [{"paper_id": f"p{i}", "domain": "NLP", "label": "macro"} for i in range(20)]
    splits = make_paper_splits(split_rows, Path(directory) / "splits.json", seed=11)
    assert [len(splits[key]) for key in ("train", "validation", "test")] == [14, 2, 4]

source = pd.DataFrame({"text": [words(80), words(140)], "token_count": [80, 140], "label": ["macro", "micro"]})
controlled = make_length_controlled(source, min_tokens=60, max_tokens=70, target_tokens=65)
assert source.token_count.tolist() == [80, 140]
assert controlled.token_count.tolist() == [65, 65]

print("offline smoke: PASS")
