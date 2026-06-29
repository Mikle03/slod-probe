import json
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src import dataset
from src.dataset import ExtractionConfig
from src.utils import read_jsonl, seed_everything, write_jsonl


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def paper(paper_id="p1", domain="NLP"):
    return {
        "paper_id": paper_id,
        "domain": domain,
        "title": words(35, "title"),
        "abstract": words(45, "abstract"),
        "sections": [
            {"name": "Introduction", "paragraphs": [words(40), words(41)]},
            {"name": "Proposed Method", "paragraphs": [words(35) + ". rest", words(55)]},
            {"name": "Evaluation", "paragraphs": [words(35) + ". rest", words(55)]},
            {"name": "Conclusion", "paragraphs": [words(42)]},
        ],
    }


def test_section_helpers_and_qasper_normalization():
    assert dataset.section_family("Ablation Evaluation") == "evaluation"
    assert dataset.section_family("Prior Literature") == "other"
    assert dataset.first_sentence("First sentence. Second sentence.") == "First sentence."
    normalized = dataset.normalize_qasper({
        "id": "x", "title": "t", "abstract": ["a", "b"],
        "full_text": {"section_name": ["Methods"], "paragraphs": [["one", "two"]]},
    })
    assert normalized["paper_id"] == "x"
    assert normalized["sections"][0]["paragraphs"] == ["one", "two"]


def test_build_summary_validation_and_jsonl_roundtrip(tmp_path):
    rows, summary = dataset.build_dataset([paper(f"p{i}") for i in range(3)], ExtractionConfig())
    assert set(pd.DataFrame(rows).label) == {"macro", "meso", "micro"}
    assert summary["dropped_class_balance"] >= 0
    assert dataset.validate_corpus(rows, min_per_class=1, min_papers=1)["paper_count"] == 3
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    seed_everything(2)


def test_qasper_iterator_uses_loader(monkeypatch):
    fake = ModuleType("datasets")
    fake.load_dataset = lambda name, split: [{
        "id": "q1", "title": "Title", "abstract": "Abstract",
        "full_text": {"section_name": [], "paragraphs": []},
    }]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    assert list(dataset.iter_qasper())[0]["paper_id"] == "q1"


def test_qasper_parquet_iterator_normalizes_rows(monkeypatch):
    frame = pd.DataFrame([{
        "id": "pq1", "title": "Title", "abstract": "Abstract",
        "full_text": {
            "section_name": np.array(["Methods"], dtype=object),
            "paragraphs": np.array([np.array(["paragraph"], dtype=object)], dtype=object),
        },
    }])
    monkeypatch.setattr(pd, "read_parquet", lambda _: frame)
    rows = list(dataset.iter_qasper_parquet("qasper.parquet", domain="NLP"))
    assert rows[0]["paper_id"] == "pq1"
    assert rows[0]["sections"][0]["name"] == "Methods"


def test_qasper_normalization_accepts_arrow_style_ndarrays():
    row = {
        "id": "arrow1", "title": "Title", "abstract": "Abstract",
        "full_text": {
            "section_name": np.array(["Methods", "Results"], dtype=object),
            "paragraphs": np.array([
                np.array(["method paragraph"], dtype=object),
                np.array(["result paragraph"], dtype=object),
            ], dtype=object),
        },
    }
    normalized = dataset.normalize_qasper(row)
    assert [section["name"] for section in normalized["sections"]] == ["Methods", "Results"]


def test_dataset_cli_writes_spans_summary_and_splits(tmp_path, monkeypatch):
    source = tmp_path / "papers.jsonl"
    write_jsonl(source, [paper(f"p{i}") for i in range(3)])
    output = tmp_path / "spans.jsonl"
    summary = tmp_path / "summary.json"
    splits = tmp_path / "splits.json"
    monkeypatch.setattr(sys, "argv", [
        "dataset.py", "--input-jsonl", str(source), "--output", str(output),
        "--summary", str(summary), "--splits", str(splits), "--allow-small",
    ])
    dataset.main()
    assert output.exists() and output.with_suffix(".csv").exists()
    assert json.loads(summary.read_text())["total_spans_saved"] > 0
    assert json.loads(splits.read_text())["test"]


def test_duplicate_paper_domain_is_rejected():
    with pytest.raises(ValueError, match="multiple domains"):
        dataset.make_paper_splits([
            {"paper_id": "same", "domain": "NLP"},
            {"paper_id": "same", "domain": "CV"},
        ])

