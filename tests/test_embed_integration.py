import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src import embed


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        size = len(texts)
        return {
            "input_ids": torch.tensor([[1, 3, 9]] * size),
            "attention_mask": torch.tensor([[1, 1, 0]] * size),
        }


class FakeModel(torch.nn.Module):
    config = SimpleNamespace(hidden_size=2)

    def forward(self, input_ids, attention_mask):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return SimpleNamespace(last_hidden_state=hidden)


def test_extract_embeddings_with_fake_frozen_encoder(monkeypatch):
    fake = ModuleType("transformers")
    fake.AutoTokenizer = SimpleNamespace(from_pretrained=lambda _: FakeTokenizer())
    fake.AutoModel = SimpleNamespace(from_pretrained=lambda _: FakeModel())
    monkeypatch.setitem(sys.modules, "transformers", fake)
    vectors = embed.extract_embeddings(["a", "b"], model_name="fake", batch_size=1, device="cpu")
    np.testing.assert_allclose(vectors, [[2.0, 2.0], [2.0, 2.0]])
    assert vectors.dtype == np.float32


def test_embed_cli_saves_aligned_row_ids(tmp_path, monkeypatch):
    source = tmp_path / "spans.csv"
    output = tmp_path / "vectors.npz"
    pd.DataFrame({"row_id": [8, 9], "text": ["a", "b"]}).to_csv(source, index=False)
    monkeypatch.setattr(embed, "extract_embeddings", lambda *args, **kwargs: np.ones((2, 3), dtype="float32"))
    monkeypatch.setattr(sys, "argv", ["embed.py", "--input", str(source), "--output", str(output)])
    embed.main()
    saved = np.load(output)
    assert saved["embeddings"].shape == (2, 3)
    assert saved["row_ids"].tolist() == [8, 9]

