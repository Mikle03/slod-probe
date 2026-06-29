from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_MODEL = "allenai/scibert_scivocab_uncased"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def extract_embeddings(
    texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 16,
    max_length: int = 512, device: str | None = None,
) -> np.ndarray:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before embedding") from exc
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    model.requires_grad_(False)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            vectors.append(mean_pool(output.last_hidden_state, batch["attention_mask"]).cpu().numpy())
    return np.concatenate(vectors).astype("float32") if vectors else np.empty((0, model.config.hidden_size), dtype="float32")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen mean-pooled scientific text embeddings")
    parser.add_argument("--input", default="data/spans/spans.csv")
    parser.add_argument("--output", default="embeddings/scibert_spans.npz")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    vectors = extract_embeddings(frame.text.fillna("").astype(str).tolist(), args.model, args.batch_size, args.max_length, args.device)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    row_ids = frame["row_id"].to_numpy() if "row_id" in frame else np.arange(len(frame))
    np.savez_compressed(target, embeddings=vectors, row_ids=row_ids, model=np.array(args.model))
    print(f"Saved {vectors.shape} frozen embeddings to {target}")


if __name__ == "__main__":
    main()

