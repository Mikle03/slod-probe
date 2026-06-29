import numpy as np
import torch

from src.embed import mean_pool


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    np.testing.assert_allclose(pooled.numpy(), [[2.0, 4.0]])

