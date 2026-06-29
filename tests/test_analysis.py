import numpy as np
import pandas as pd

from src.analysis import confusion_pairs, make_tsne, select_qualitative_examples


LABELS = ["macro", "meso", "micro"]


def test_select_examples_returns_three_correct_and_failures_per_class():
    rows = []
    probabilities = []
    predictions = []
    for label_index, label in enumerate(LABELS):
        for index in range(6):
            rows.append({"label": label, "text": f"{label} example {index}", "row_id": len(rows)})
            if index < 3:
                probability = np.full(3, 0.025)
                probability[label_index] = 0.95
                predicted = label
            else:
                wrong_index = (label_index + 1) % 3
                probability = np.full(3, 0.05)
                probability[wrong_index] = 0.90
                predicted = LABELS[wrong_index]
            probabilities.append(probability)
            predictions.append(predicted)

    examples = select_qualitative_examples(
        pd.DataFrame(rows), np.asarray(probabilities), np.asarray(predictions), LABELS, n=3,
    )

    assert len(examples) == 18
    assert set(examples.groupby(["true_label", "outcome"]).size()) == {3}
    assert {"confidence", "predicted_label", "text"} <= set(examples.columns)


def test_confusion_pairs_orders_non_diagonal_errors():
    matrix = [[8, 2, 0], [1, 7, 2], [3, 0, 7]]
    pairs = confusion_pairs(matrix, LABELS)
    assert pairs[0] == {"true": "micro", "predicted": "macro", "count": 3}
    assert all(pair["true"] != pair["predicted"] for pair in pairs)


def test_tsne_sample_retains_label_column():
    frame = pd.DataFrame({
        "row_id": np.arange(18),
        "label": np.repeat(LABELS, 6),
        "text": [f"example {index}" for index in range(18)],
    })
    vectors = np.random.default_rng(4).normal(size=(18, 8)).astype("float32")
    coordinates = make_tsne(frame, vectors, {index: index for index in range(18)}, max_per_class=2)
    assert set(coordinates.label) == set(LABELS)
    assert {"tsne_x", "tsne_y"} <= set(coordinates.columns)
