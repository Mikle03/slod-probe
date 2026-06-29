from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def compare_length_results(
    normal_metrics: str | Path,
    controlled_metrics: str | Path,
    output_path: str | Path | None = None,
    tolerance: float = 0.01,
) -> dict:
    """Compare probe macro-F1 and classify the change with an explicit tolerance."""
    normal = json.loads(Path(normal_metrics).read_text(encoding="utf-8"))
    controlled = json.loads(Path(controlled_metrics).read_text(encoding="utf-8"))
    normal_f1 = float(normal["embedding_probe"]["macro_f1"])
    controlled_f1 = float(controlled["embedding_probe"]["macro_f1"])
    delta = round(controlled_f1 - normal_f1, 10)
    interpretation = "drops" if delta < -tolerance else "improves" if delta > tolerance else "stays_same"
    result = {
        "normal_macro_f1": normal_f1,
        "length_controlled_macro_f1": controlled_f1,
        "macro_f1_delta": delta,
        "tolerance": tolerance,
        "interpretation": interpretation,
    }
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def make_length_controlled(
    frame: pd.DataFrame, min_tokens: int = 30, max_tokens: int = 30,
    target_tokens: int = 30, seed: int = 42,
) -> pd.DataFrame:
    if not min_tokens <= target_tokens <= max_tokens:
        raise ValueError("target_tokens must lie inside [min_tokens, max_tokens]")
    rng = random.Random(seed)
    rows = []
    for _, source in frame.iterrows():
        words = str(source["text"]).split()
        if len(words) < min_tokens:
            continue
        take = min(target_tokens, max_tokens, len(words))
        start = rng.randint(0, len(words) - take) if len(words) > take else 0
        row = source.copy()
        row["controlled_original_token_count"] = int(source["token_count"])
        row["text"] = " ".join(words[start:start + take])
        row["token_count"] = take
        rows.append(row)
    return pd.DataFrame(rows, columns=list(frame.columns) + ["controlled_original_token_count"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a separate length-controlled SLoD dataset")
    parser.add_argument("--input", default="data/spans/spans.csv")
    parser.add_argument("--output", default="data/spans/spans_length_controlled.csv")
    parser.add_argument("--min-tokens", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--target-tokens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-metrics", help="Normal in-domain metrics JSON to compare")
    parser.add_argument("--controlled-metrics", help="Length-controlled metrics JSON to compare")
    parser.add_argument("--comparison-output", default="results/length_control_comparison.json")
    parser.add_argument("--comparison-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if args.normal_metrics or args.controlled_metrics:
        if not (args.normal_metrics and args.controlled_metrics):
            parser.error("--normal-metrics and --controlled-metrics must be provided together")
        result = compare_length_results(
            args.normal_metrics, args.controlled_metrics,
            args.comparison_output, args.comparison_tolerance,
        )
        print(json.dumps(result, indent=2))
        return
    controlled = make_length_controlled(
        pd.read_csv(args.input, dtype={"paper_id": str}),
        args.min_tokens, args.max_tokens, args.target_tokens, args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    controlled.to_csv(args.output, index=False)
    print(f"Saved {len(controlled)} controlled spans to {args.output}")


if __name__ == "__main__":
    main()
