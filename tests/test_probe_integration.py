import json
import sys

import numpy as np
import pandas as pd
import pytest

from src import probe


def experiment_frame():
    rows = []
    for paper_index in range(6):
        for label_index, (label, section) in enumerate((
            ("macro", "Abstract"), ("meso", "Related Work"), ("micro", "Methods")
        )):
            rows.append({
                "row_id": len(rows), "paper_id": f"p{paper_index}", "domain": "NLP",
                "label": label, "section_name": section, "label_index": label_index,
            })
    return pd.DataFrame(rows)


def test_condition_selection_and_errors():
    frame = experiment_frame()
    splits = {"train": ["p0", "p1", "p2", "p3"], "validation": [], "test": ["p4", "p5"]}
    train, test = probe.select_condition(frame, splits, "in_domain", "NLP", None, None)
    assert train.paper_id.nunique() == 4 and test.paper_id.nunique() == 2
    with pytest.raises(ValueError, match="two genuine"):
        probe.select_condition(frame, splits, "cross_domain", None, None, None)
    with pytest.raises(ValueError, match="must differ"):
        probe.select_condition(frame, splits, "cross_domain", None, "NLP", "NLP")
    with pytest.raises(ValueError, match="Unknown"):
        probe.select_condition(frame, splits, "bad", None, None, None)


def test_probe_cli_runs_all_baselines_and_saves_confusions(tmp_path, monkeypatch):
    frame = experiment_frame()
    spans = tmp_path / "spans.csv"
    embeddings = tmp_path / "vectors.npz"
    splits_path = tmp_path / "splits.json"
    output = tmp_path / "metrics.json"
    frame.drop(columns="label_index").to_csv(spans, index=False)
    vectors = np.eye(3, dtype="float32")[frame.label_index.to_numpy()]
    np.savez_compressed(embeddings, embeddings=vectors, row_ids=frame.row_id.to_numpy())
    splits_path.write_text(json.dumps({
        "train": ["p0", "p1", "p2", "p3"], "validation": [], "test": ["p4", "p5"]
    }))
    monkeypatch.setattr(sys, "argv", [
        "probe.py", "--train", "--eval", "--condition", "in_domain", "--domain", "NLP",
        "--spans", str(spans), "--embeddings", str(embeddings),
        "--splits", str(splits_path), "--output", str(output),
    ])
    probe.main()
    result = json.loads(output.read_text())
    assert result["embedding_probe"]["accuracy"] == 1.0
    assert result["paper_overlap"] == []
    assert len(list(tmp_path.glob("*confusion.csv"))) == 3


def test_majority_baseline_records_predicted_class():
    train = pd.DataFrame({"label": ["macro", "macro", "micro"]})
    test = pd.DataFrame({"label": ["macro", "meso"]})
    assert probe.run_majority_baseline(train, test)["predicted_class"] == "macro"
