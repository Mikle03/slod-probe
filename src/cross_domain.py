from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from src.probe import evaluate_predictions, load_vectors, run_majority_baseline, run_section_name_baseline, save_confusion_csvs
    from src.utils import write_json
except ModuleNotFoundError:  # direct execution: python src/cross_domain.py
    from probe import evaluate_predictions, load_vectors, run_majority_baseline, run_section_name_baseline, save_confusion_csvs
    from utils import write_json


def run_external_domain_experiment(
    train: pd.DataFrame, train_vectors: np.ndarray, train_index: dict[int, int],
    external: pd.DataFrame, external_vectors: np.ndarray, external_index: dict[int, int],
    seed: int = 42, predictions_output: str | Path | None = None,
) -> dict[str, Any]:
    train_positions = [train_index[int(row_id)] for row_id in train.row_id]
    test_positions = [external_index[int(row_id)] for row_id in external.row_id]
    classifier = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    classifier.fit(train_vectors[train_positions], train.label)
    probabilities = classifier.predict_proba(external_vectors[test_positions])
    predictions = classifier.classes_[probabilities.argmax(axis=1)]
    result = {
        "embedding_probe": evaluate_predictions(external.label.to_numpy(), predictions),
        "majority_baseline": run_majority_baseline(train, external),
        "section_name_baseline": run_section_name_baseline(train, external, seed=seed),
        "n_train": len(train),
        "n_test": len(external),
        "train_papers": int(train.paper_id.nunique()),
        "test_papers": int(external.paper_id.nunique()),
        "paper_overlap": sorted(set(train.paper_id.astype(str)) & set(external.paper_id.astype(str))),
        "condition": "cross_domain_external",
        "train_domain": "NLP",
        "test_domain": "biomedicine",
        "source_corpus_shift": "QASPER to PMC BioC",
    }
    if predictions_output:
        prediction_frame = external.copy()
        prediction_frame["predicted_label"] = predictions
        prediction_frame["confidence"] = probabilities.max(axis=1)
        prediction_frame["correct"] = prediction_frame.label == prediction_frame.predicted_label
        for class_index, label in enumerate(classifier.classes_):
            prediction_frame[f"probability_{label}"] = probabilities[:, class_index]
        target = Path(predictions_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(target, index=False)
    return result


def select_qualitative_examples(predictions: pd.DataFrame, per_class: int = 3) -> pd.DataFrame:
    examples = []
    for label in ("macro", "meso", "micro"):
        subset = predictions[predictions.label == label]
        correct = subset[subset.correct.astype(bool)].sort_values("confidence", ascending=False).head(per_class).copy()
        correct["example_type"] = "high_confidence_correct"
        failed = subset[~subset.correct.astype(bool)].sort_values("confidence", ascending=False).head(per_class).copy()
        failed["example_type"] = "high_confidence_failure"
        examples.extend([correct, failed])
    return pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a QASPER-trained probe on a separate PMC BioC test set")
    parser.add_argument("--train-spans", default="data/spans/spans.csv")
    parser.add_argument("--train-splits", default="data/spans/paper_splits.json")
    parser.add_argument("--train-embeddings", default="embeddings/scibert_spans.npz")
    parser.add_argument("--test-spans", default="data/spans/pmc_biomedicine_spans.csv")
    parser.add_argument("--test-embeddings", default="embeddings/scibert_pmc_biomedicine.npz")
    parser.add_argument("--output", default="results/cross_domain_biomedicine_metrics.json")
    parser.add_argument("--predictions", default="results/analysis/cross_domain_biomedicine_predictions.csv")
    parser.add_argument("--examples", default="results/analysis/cross_domain_biomedicine_examples.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = pd.read_csv(args.train_spans, dtype={"paper_id": str})
    splits = json.loads(Path(args.train_splits).read_text(encoding="utf-8"))
    train = source[source.paper_id.astype(str).isin(splits["train"])].copy()
    external = pd.read_csv(args.test_spans, dtype={"paper_id": str})
    source_vectors, source_index = load_vectors(args.train_embeddings)
    external_vectors, external_index = load_vectors(args.test_embeddings)
    result = run_external_domain_experiment(
        train, source_vectors, source_index, external, external_vectors, external_index,
        args.seed, args.predictions,
    )
    output = Path(args.output)
    write_json(output, result)
    save_confusion_csvs(result, output)
    predictions = pd.read_csv(args.predictions)
    examples = select_qualitative_examples(predictions)
    Path(args.examples).parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(args.examples, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
