import pytest
import torch
import numpy as np

from softargmax import softargmax1d, softargmax2d


def test_softargmax1d():
    x = torch.tensor([1, 2, 3, 4, 3, 2, 1], dtype=torch.float32)
    argmax = softargmax1d(x)

    assert argmax == 3


def test_softargmax2d():
    x = torch.tensor(gaussian_2d(shape=(20, 30), center=(4, 13)))
    argmax = softargmax2d(x).numpy()
    assert np.all(argmax[0, :] == [4, 13])


def gaussian_2d(shape, center, sigma=1.0):
    xs, ys = np.meshgrid(
        np.arange(0, shape[1], step=1.0, dtype=np.float32),
        np.arange(0, shape[0], step=1.0, dtype=np.float32))

    alpha = -0.5 / (sigma ** 2)
    res = np.exp(alpha * ((xs - center[1]) ** 2 + (ys - center[0]) ** 2))
    return res
