import json
import sys

import pandas as pd

from src import controls
from src.controls import compare_length_results, make_length_controlled


def test_length_comparison_reports_drop_and_delta(tmp_path):
    normal = {"embedding_probe": {"macro_f1": 0.72}}
    controlled = {"embedding_probe": {"macro_f1": 0.61}}
    normal_path = tmp_path / "normal.json"
    controlled_path = tmp_path / "controlled.json"
    output_path = tmp_path / "comparison.json"
    normal_path.write_text(json.dumps(normal))
    controlled_path.write_text(json.dumps(controlled))

    result = compare_length_results(normal_path, controlled_path, output_path, tolerance=0.01)

    assert result["interpretation"] == "drops"
    assert result["macro_f1_delta"] == -0.11
    assert json.loads(output_path.read_text()) == result


def test_length_comparison_uses_tolerance_for_stable_result(tmp_path):
    for name, score in (("normal", 0.700), ("controlled", 0.705)):
        (tmp_path / f"{name}.json").write_text(json.dumps({"embedding_probe": {"macro_f1": score}}))

    result = compare_length_results(tmp_path / "normal.json", tmp_path / "controlled.json", tolerance=0.01)

    assert result["interpretation"] == "stays_same"


def test_default_length_control_uses_common_qasper_support():
    frame = pd.DataFrame({
        "text": [" ".join(["x"] * 35), " ".join(["y"] * 120)],
        "token_count": [35, 120],
        "label": ["meso", "macro"],
    })
    controlled = make_length_controlled(frame)
    assert len(controlled) == len(frame)
    assert controlled.token_count.tolist() == [30, 30]


def test_controls_cli_preserves_arxiv_paper_id_strings(tmp_path, monkeypatch):
    source = tmp_path / "spans.csv"
    output = tmp_path / "controlled.csv"
    pd.DataFrame({
        "paper_id": ["1609.07020"],
        "text": [" ".join(["x"] * 35)],
        "token_count": [35],
        "label": ["meso"],
    }).to_csv(source, index=False)
    monkeypatch.setattr(sys, "argv", [
        "controls.py", "--input", str(source), "--output", str(output),
    ])
    controls.main()
    saved = pd.read_csv(output, dtype={"paper_id": str})
    assert saved.paper_id.tolist() == ["1609.07020"]
