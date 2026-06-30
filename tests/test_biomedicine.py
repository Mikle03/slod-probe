import json
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src import biomedicine, cross_domain
from src.biomedicine import normalize_bioc_collection, select_external_test_spans
from src.cross_domain import run_external_domain_experiment, select_qualitative_examples
from src.dataset import ExtractionConfig


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def bioc_fixture() -> dict:
    return {
        "documents": [{
            "id": "PMC123",
            "passages": [
                {"infons": {"type": "front", "section_type": "TITLE"}, "text": "A biomedical study"},
                {"infons": {"type": "abstract_title_1", "section_type": "ABSTRACT"}, "text": "Background"},
                {"infons": {"type": "abstract", "section_type": "ABSTRACT"}, "text": words(35, "abstract")},
                {"infons": {"type": "title_1", "section_type": "INTRO"}, "text": "Introduction"},
                {"infons": {"type": "paragraph", "section_type": "INTRO"}, "text": words(40, "intro")},
                {"infons": {"type": "paragraph", "section_type": "INTRO"}, "text": words(41, "intro")},
                {"infons": {"type": "title_1", "section_type": "METHODS"}, "text": "Materials and Methods"},
                {"infons": {"type": "paragraph", "section_type": "METHODS"}, "text": words(35, "methodlead") + ". " + words(10)},
                {"infons": {"type": "paragraph", "section_type": "METHODS"}, "text": words(50, "method")},
                {"infons": {"type": "title_1", "section_type": "RESULTS"}, "text": "Results"},
                {"infons": {"type": "paragraph", "section_type": "RESULTS"}, "text": words(35, "resultlead") + ". " + words(10)},
                {"infons": {"type": "paragraph", "section_type": "RESULTS"}, "text": words(50, "result")},
                {"infons": {"type": "title_1", "section_type": "DISCUSS"}, "text": "Discussion and Conclusion"},
                {"infons": {"type": "paragraph", "section_type": "DISCUSS"}, "text": words(45, "conclusion")},
                {"infons": {"type": "paragraph", "section_type": "FIG"}, "text": words(50, "caption")},
                {"infons": {"type": "paragraph", "section_type": "REF"}, "text": words(50, "reference")},
            ],
        }]
    }


def test_bioc_normalization_preserves_sections_and_excludes_non_body_content():
    paper = normalize_bioc_collection(bioc_fixture())
    assert paper["paper_id"] == "PMC123"
    assert paper["domain"] == "biomedicine"
    assert paper["title"] == "A biomedical study"
    assert paper["abstract"].startswith("abstract0")
    assert [section["name"] for section in paper["sections"]] == [
        "Introduction", "Materials and Methods", "Results", "Discussion and Conclusion"
    ]
    all_text = " ".join(p for s in paper["sections"] for p in s["paragraphs"])
    assert "caption" not in all_text
    assert "reference" not in all_text


def test_bioc_normalization_accepts_real_api_collection_list_shape():
    paper = normalize_bioc_collection([bioc_fixture()])
    assert paper["paper_id"] == "PMC123"
    assert len(paper["sections"]) == 4


def test_bioc_normalization_and_selection_reject_invalid_input():
    with pytest.raises(ValueError, match="no documents"):
        normalize_bioc_collection({"documents": []})
    with pytest.raises(ValueError, match="Need 2 external macro"):
        select_external_test_spans([], per_class=2)


def test_external_selection_is_balanced_deterministic_and_paper_capped():
    rows = []
    for paper_index in range(120):
        for label in ("macro", "meso", "micro"):
            for span_index in range(6):
                rows.append({
                    "paper_id": f"PMC{paper_index}", "domain": "biomedicine", "label": label,
                    "text": words(35), "section_name": label, "row_id": len(rows),
                })
    selected = select_external_test_spans(rows, per_class=500, max_per_paper_per_label=5, seed=9)
    frame = pd.DataFrame(selected)
    assert frame.label.value_counts().to_dict() == {"macro": 500, "meso": 500, "micro": 500}
    assert frame.groupby(["paper_id", "label"]).size().max() <= 5
    assert frame.paper_id.nunique() >= 100
    assert selected == select_external_test_spans(rows, per_class=500, max_per_paper_per_label=5, seed=9)


def test_external_experiment_trains_only_on_source_domain():
    train = pd.DataFrame({
        "paper_id": [f"q{i}" for i in range(9)],
        "row_id": list(range(9)),
        "label": ["macro", "meso", "micro"] * 3,
        "section_name": ["Abstract", "Methods", "Results"] * 3,
    })
    external = pd.DataFrame({
        "paper_id": [f"PMC{i}" for i in range(6)],
        "row_id": list(range(100, 106)),
        "label": ["macro", "meso", "micro"] * 2,
        "section_name": ["Abstract", "Methods", "Results"] * 2,
    })
    train_vectors = np.eye(3, dtype="float32")[[0, 1, 2] * 3]
    external_vectors = np.eye(3, dtype="float32")[[0, 1, 2] * 2]
    result = run_external_domain_experiment(
        train, train_vectors, {i: i for i in range(9)},
        external, external_vectors, {100 + i: i for i in range(6)}, seed=4,
    )
    assert result["embedding_probe"]["accuracy"] == 1.0
    assert result["n_train"] == 9 and result["n_test"] == 6
    assert result["train_domain"] == "NLP"
    assert result["test_domain"] == "biomedicine"
    assert result["paper_overlap"] == []


def test_search_and_collection_use_bioc_api_without_training_on_external_labels(monkeypatch):
    def fake_get_json(url, **_kwargs):
        if "europepmc" in url:
            return {"resultList": {"result": [{"pmcid": "PMC1"}, {"pmcid": "PMC2"}, {}]}}
        pmcid = "PMC1" if "PMC1" in url else "PMC2"
        payload = deepcopy(bioc_fixture())
        payload["documents"][0]["id"] = pmcid
        return [payload]

    monkeypatch.setattr(biomedicine, "_get_json", fake_get_json)
    assert biomedicine.search_pmcids(page_size=3) == ["PMC1", "PMC2"]
    rows, summary = biomedicine.collect_biomedical_spans(
        ["PMC1", "PMC2"], per_class=2, min_papers=2,
        config=ExtractionConfig(max_spans_per_paper_per_label=5), request_delay=0,
    )
    assert len(rows) == 6
    assert summary["purpose"] == "external_cross_domain_test_only"
    assert summary["papers_fetched_successfully"] == 2


def test_biomedicine_cli_writes_only_separate_artifacts(tmp_path, monkeypatch):
    rows = []
    for i, label in enumerate(("macro", "meso", "micro")):
        rows.append({
            "paper_id": f"PMC{i}", "domain": "biomedicine", "section_name": label,
            "section_family": label, "paragraph_index": 0, "label": label,
            "label_source_rule": f"test_{label}", "text": words(35), "token_count": 35,
            "row_id": i,
        })
    summary = {
        "access_date": "2026-06-30", "total_papers": 3, "total_spans_saved": 3,
    }
    monkeypatch.setattr(biomedicine, "search_pmcids", lambda **_kwargs: ["PMC0", "PMC1", "PMC2"])
    monkeypatch.setattr(biomedicine, "collect_biomedical_spans", lambda *_args, **_kwargs: (rows, summary))
    output = tmp_path / "pmc.jsonl"
    summary_path = tmp_path / "summary.json"
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(sys, "argv", [
        "biomedicine.py", "--output", str(output), "--summary", str(summary_path),
        "--manifest", str(manifest), "--per-class", "1", "--min-papers", "1",
    ])
    biomedicine.main()
    assert output.exists() and output.with_suffix(".csv").exists()
    assert json.loads(manifest.read_text())["selected_paper_ids"] == ["PMC0", "PMC1", "PMC2"]


def test_qualitative_selection_and_cross_domain_cli(tmp_path, monkeypatch):
    train = pd.DataFrame({
        "paper_id": [f"q{i}" for i in range(9)], "row_id": range(9),
        "label": ["macro", "meso", "micro"] * 3,
        "section_name": ["Abstract", "Methods", "Results"] * 3,
    })
    external = pd.DataFrame({
        "paper_id": [f"PMC{i}" for i in range(6)], "row_id": range(6),
        "label": ["macro", "meso", "micro"] * 2,
        "section_name": ["Abstract", "Methods", "Results"] * 2,
        "text": [words(35)] * 6,
    })
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    external.to_csv(test_path, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": train.paper_id.tolist(), "validation": [], "test": []}))
    train_embeddings, test_embeddings = tmp_path / "train.npz", tmp_path / "test.npz"
    np.savez(train_embeddings, embeddings=np.eye(3, dtype="float32")[[0, 1, 2] * 3], row_ids=np.arange(9))
    np.savez(test_embeddings, embeddings=np.eye(3, dtype="float32")[[0, 1, 2] * 2], row_ids=np.arange(6))
    metrics, predictions, examples = tmp_path / "metrics.json", tmp_path / "predictions.csv", tmp_path / "examples.csv"
    monkeypatch.setattr(sys, "argv", [
        "cross_domain.py", "--train-spans", str(train_path), "--train-splits", str(splits),
        "--train-embeddings", str(train_embeddings), "--test-spans", str(test_path),
        "--test-embeddings", str(test_embeddings), "--output", str(metrics),
        "--predictions", str(predictions), "--examples", str(examples),
    ])
    cross_domain.main()
    assert json.loads(metrics.read_text())["embedding_probe"]["accuracy"] == 1.0
    assert predictions.exists() and examples.exists()
    prediction_frame = pd.read_csv(predictions)
    qualitative = select_qualitative_examples(prediction_frame, per_class=1)
    assert set(qualitative.example_type) == {"high_confidence_correct"}
