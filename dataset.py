"""
Fashion-MNIST loading, the fixed train/val/test split, and deterministic
batch ordering. Loads the dataset at import time — import this module
exactly once (main.py does it first), everything else imports the
already-loaded arrays from here.
"""

import random

import numpy as np
import mlx.core as mx

from torchvision.datasets import FashionMNIST

import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def to_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def prepare_dataset(dataset):
    images = to_np(dataset.data).astype(np.float32) / 255.0
    labels = to_np(dataset.targets).astype(np.int32)
    images = (images - 0.2860) / 0.3530
    return images[..., None], labels  # [N, H, W, C] for MLX


print("Loading Fashion-MNIST...")
_train = FashionMNIST(root="./data", train=True, download=True)
_test = FashionMNIST(root="./data", train=False, download=True)

full_train_images, full_train_labels = prepare_dataset(_train)
test_images, test_labels = prepare_dataset(_test)

_perm = np.random.default_rng(0).permutation(len(full_train_images))
_val_idx, _train_idx = _perm[: config.VAL_SIZE], _perm[config.VAL_SIZE :]

val_images, val_labels = full_train_images[_val_idx], full_train_labels[_val_idx]
train_images, train_labels = full_train_images[_train_idx], full_train_labels[_train_idx]

print("Train:", train_images.shape, "Val:", val_images.shape, "Test:", test_images.shape)

SPLITS = {"val": (val_images, val_labels), "test": (test_images, test_labels)}


def epoch_batches(n, seed, epoch):
    """Deterministic shuffled batch indices. Same (seed, epoch) -> same order."""
    idx = np.random.default_rng(seed + epoch).permutation(n)
    for start in range(0, n, config.BATCH_SIZE):
        yield idx[start : start + config.BATCH_SIZE]