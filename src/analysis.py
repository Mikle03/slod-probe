from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

try:
    from src.probe import load_vectors, select_condition
    from src.utils import LABELS, write_json
except ModuleNotFoundError:
    from probe import load_vectors, select_condition
    from utils import LABELS, write_json


def select_qualitative_examples(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: list[str] = LABELS,
    n: int = 3,
) -> pd.DataFrame:
    frame = test.reset_index(drop=True).copy()
    frame["true_label"] = frame["label"]
    frame["predicted_label"] = predictions
    frame["confidence"] = probabilities.max(axis=1)
    frame["outcome"] = np.where(
        frame.true_label == frame.predicted_label,
        "correct_high_confidence",
        "failed_high_confidence",
    )
    selected = []
    for label in labels:
        for outcome in ("correct_high_confidence", "failed_high_confidence"):
            candidates = frame[(frame.true_label == label) & (frame.outcome == outcome)]
            selected.append(candidates.sort_values("confidence", ascending=False).head(n))
    result = pd.concat(selected, ignore_index=True)
    preferred = [
        "outcome", "true_label", "predicted_label", "confidence", "paper_id",
        "section_name", "section_family", "label_source_rule", "token_count", "text", "row_id",
    ]
    return result[[column for column in preferred if column in result.columns]]


def confusion_pairs(matrix: list[list[int]], labels: list[str] = LABELS) -> list[dict[str, Any]]:
    pairs = []
    for true_index, true_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            if true_index != predicted_index:
                pairs.append({
                    "true": true_label,
                    "predicted": predicted_label,
                    "count": int(matrix[true_index][predicted_index]),
                })
    return sorted(pairs, key=lambda item: item["count"], reverse=True)


def make_tsne(
    frame: pd.DataFrame,
    vectors: np.ndarray,
    vector_index: dict[int, int],
    max_per_class: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    sampled = pd.concat([
        group.sample(min(len(group), max_per_class), random_state=seed)
        for _, group in frame.groupby("label", sort=True)
    ], ignore_index=True)
    positions = [vector_index[int(row_id)] for row_id in sampled.row_id]
    values = vectors[positions]
    dimensions = min(50, values.shape[0] - 1, values.shape[1])
    reduced = PCA(n_components=dimensions, random_state=seed).fit_transform(values)
    coordinates = TSNE(
        n_components=2, perplexity=min(30, max(5, (len(sampled) - 1) // 3)),
        init="pca", learning_rate="auto", random_state=seed,
    ).fit_transform(reduced)
    sampled["tsne_x"] = coordinates[:, 0]
    sampled["tsne_y"] = coordinates[:, 1]
    return sampled


def save_tsne_plot(coordinates: pd.DataFrame, output: str | Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"macro": "#2864dc", "meso": "#e59b20", "micro": "#cf3f5b"}
    figure, axis = plt.subplots(figsize=(7.2, 5.2), dpi=160)
    for label in LABELS:
        subset = coordinates[coordinates.label == label]
        axis.scatter(subset.tsne_x, subset.tsne_y, s=12, alpha=0.62, label=label, c=colors[label], edgecolors="none")
    axis.set(title="Frozen SciBERT embedding space (t-SNE)", xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative and embedding-space SLoD analyses")
    parser.add_argument("--spans", default="data/spans/spans.csv")
    parser.add_argument("--embeddings", default="embeddings/scibert_spans.npz")
    parser.add_argument("--splits", default="data/spans/paper_splits.json")
    parser.add_argument("--metrics", default="results/in_domain_metrics.json")
    parser.add_argument("--domain", default="NLP")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.spans, dtype={"paper_id": str})
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    train, test = select_condition(frame, splits, "in_domain", args.domain, None, None)
    vectors, vector_index = load_vectors(args.embeddings)
    train_positions = [vector_index[int(row_id)] for row_id in train.row_id]
    test_positions = [vector_index[int(row_id)] for row_id in test.row_id]
    classifier = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed)
    classifier.fit(vectors[train_positions], train.label)
    probabilities = classifier.predict_proba(vectors[test_positions])
    predictions = classifier.predict(vectors[test_positions])

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = select_qualitative_examples(test, probabilities, predictions, list(classifier.classes_))
    examples.to_csv(output / "qualitative_examples.csv", index=False)
    examples.to_json(output / "qualitative_examples.json", orient="records", indent=2)

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    write_json(output / "confusion_pairs.json", confusion_pairs(
        metrics["embedding_probe"]["confusion_matrix"], metrics["embedding_probe"]["label_order"]
    ))
    coordinates = make_tsne(test, vectors, vector_index, seed=args.seed)
    coordinates.to_csv(output / "tsne_coordinates.csv", index=False)
    save_tsne_plot(coordinates, output / "tsne_embeddings.png")
    print(f"Saved analysis artifacts to {output}")


if __name__ == "__main__":
    main()
