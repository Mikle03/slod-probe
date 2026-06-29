from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

try:
    from src.utils import LABELS, write_json
except ModuleNotFoundError:
    from utils import LABELS, write_json


def evaluate_predictions(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray, labels: list[str] = LABELS) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "label_order": labels,
    }


def run_majority_baseline(train: pd.DataFrame, test: pd.DataFrame, labels: list[str] = LABELS) -> dict[str, Any]:
    majority = str(train.label.value_counts().index[0])
    result = evaluate_predictions(test.label.to_numpy(), np.repeat(majority, len(test)), labels)
    result["predicted_class"] = majority
    return result


def run_section_name_baseline(train: pd.DataFrame, test: pd.DataFrame, labels: list[str] = LABELS, seed: int = 42) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5))
    x_train = vectorizer.fit_transform(train.section_name.fillna("unknown").astype(str))
    x_test = vectorizer.transform(test.section_name.fillna("unknown").astype(str))
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    classifier.fit(x_train, train.label)
    return evaluate_predictions(test.label.to_numpy(), classifier.predict(x_test), labels)


def select_condition(frame: pd.DataFrame, splits: dict[str, list[str]], condition: str, domain: str | None, train_domain: str | None, test_domain: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    paper_ids = frame.paper_id.astype(str)
    if condition in {"in_domain", "length_controlled"}:
        chosen = domain or sorted(frame.domain.astype(str).unique())[0]
        train = frame[(frame.domain.astype(str) == chosen) & paper_ids.isin(splits["train"])]
        test = frame[(frame.domain.astype(str) == chosen) & paper_ids.isin(splits["test"])]
    elif condition == "cross_domain":
        domains = sorted(frame.domain.astype(str).unique())
        if train_domain is None or test_domain is None:
            if len(domains) < 2:
                raise ValueError("cross_domain requires two genuine domain values; QASPER alone is normally NLP-only")
            train_domain, test_domain = domains[:2]
        if train_domain == test_domain:
            raise ValueError("train-domain and test-domain must differ")
        train = frame[(frame.domain.astype(str) == train_domain) & paper_ids.isin(splits["train"])]
        test = frame[(frame.domain.astype(str) == test_domain) & paper_ids.isin(splits["test"])]
    else:
        raise ValueError(f"Unknown condition: {condition}")
    if train.empty or test.empty:
        raise ValueError(f"Condition {condition} produced an empty train or test set")
    missing = set(test.label.unique()) - set(train.label.unique())
    if missing:
        raise ValueError(f"Training split lacks test classes: {sorted(missing)}")
    return train, test


def load_vectors(path: str | Path) -> tuple[np.ndarray, dict[int, int]]:
    payload = np.load(path, allow_pickle=False)
    return payload["embeddings"], {int(row_id): i for i, row_id in enumerate(payload["row_ids"])}


def run_experiment(frame: pd.DataFrame, vectors: np.ndarray, vector_index: dict[int, int], train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    train_positions = [vector_index[int(i)] for i in train.row_id]
    test_positions = [vector_index[int(i)] for i in test.row_id]
    classifier = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    classifier.fit(vectors[train_positions], train.label)
    result = {
        "embedding_probe": evaluate_predictions(test.label.to_numpy(), classifier.predict(vectors[test_positions])),
        "majority_baseline": run_majority_baseline(train, test),
        "section_name_baseline": run_section_name_baseline(train, test, seed=seed),
        "n_train": len(train),
        "n_test": len(test),
        "train_papers": int(train.paper_id.nunique()),
        "test_papers": int(test.paper_id.nunique()),
        "paper_overlap": sorted(set(train.paper_id.astype(str)) & set(test.paper_id.astype(str))),
    }
    return result


def save_confusion_csvs(result: dict[str, Any], output_json: Path) -> None:
    for name in ("embedding_probe", "majority_baseline", "section_name_baseline"):
        metrics = result[name]
        pd.DataFrame(metrics["confusion_matrix"], index=metrics["label_order"], columns=metrics["label_order"]).to_csv(
            output_json.with_name(f"{output_json.stem}_{name}_confusion.csv")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a linear SLoD probe and diagnostic baselines")
    parser.add_argument("--train", action="store_true", help="Accepted for requested CLI compatibility")
    parser.add_argument("--eval", action="store_true", help="Accepted for requested CLI compatibility")
    parser.add_argument("--condition", choices=["in_domain", "cross_domain", "length_controlled"], default="in_domain")
    parser.add_argument("--spans", default="data/spans/spans.csv")
    parser.add_argument("--embeddings", default="embeddings/scibert_spans.npz")
    parser.add_argument("--splits", default="data/spans/paper_splits.json")
    parser.add_argument("--domain")
    parser.add_argument("--train-domain")
    parser.add_argument("--test-domain")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.spans, dtype={"paper_id": str})
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    train, test = select_condition(frame, splits, args.condition, args.domain, args.train_domain, args.test_domain)
    vectors, vector_index = load_vectors(args.embeddings)
    result = run_experiment(frame, vectors, vector_index, train, test, args.seed)
    result["condition"] = args.condition
    result["domain"] = args.domain
    result["train_domain"] = args.train_domain
    result["test_domain"] = args.test_domain
    output = Path(args.output or f"results/{args.condition}_metrics.json")
    write_json(output, result)
    save_confusion_csvs(result, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

