"""
Unit tests for the KD math itself — no dataset, no models. Run this
before any training so a broken loss fails in seconds, not after a
multi-hour run.
"""

import numpy as np
import mlx.core as mx

from losses import make_log_target, kd_kl_loss
from metrics import np_softmax


def sanity_checks():
    rng = np.random.default_rng(0)
    z1 = (rng.normal(size=(64, 10)) * 3).astype(np.float32)
    z2 = (rng.normal(size=(64, 10)) * 3).astype(np.float32)
    s = (rng.normal(size=(64, 10)) * 3).astype(np.float32)
    mz1, mz2, ms = mx.array(z1), mx.array(z2), mx.array(s)

    gaps = {}
    for T in (1.0, 4.0, 16.0):
        lA = np.array(make_log_target("kd_A", mz1, mz2, T))
        lB = np.array(make_log_target("kd_B", mz1, mz2, T))

        assert np.allclose(np.exp(lA).sum(-1), 1.0, atol=1e-5)
        assert np.allclose(np.exp(lB).sum(-1), 1.0, atol=1e-5)

        pA = np_softmax((z1 + z2) / 2, T)
        pB = (np_softmax(z1, T) + np_softmax(z2, T)) / 2
        assert np.allclose(np.exp(lA), pA, atol=1e-5)
        assert np.allclose(np.exp(lB), pB, atol=1e-5)

        q = np_softmax(s, T)
        ref_kl = np.mean(np.sum(pB * (np.log(pB) - np.log(q)), axis=1))
        got_kl = float(kd_kl_loss(mx.array(lB), ms, T))
        assert abs(ref_kl - got_kl) < 1e-4, (ref_kl, got_kl)
        assert got_kl >= 0
        self_kl = float(kd_kl_loss(mx.array(lB), mx.array(T * lB), T))
        assert abs(self_kl) < 1e-4

        gaps[T] = float(np.mean(np.abs(np.exp(lA) - np.exp(lB))))

    same = np.abs(
        np.array(make_log_target("kd_A", mz1, mz1, 4.0))
        - np.array(make_log_target("kd_B", mz1, mz1, 4.0))
    ).max()
    assert same < 1e-5

    assert gaps[1.0] > gaps[4.0] > gaps[16.0], gaps

    norms = {}
    for T in (1.0, 4.0, 16.0):
        lt = make_log_target("kd_B", mz1, mz2, T)
        g = mx.grad(lambda z: (T**2) * kd_kl_loss(lt, z, T))(ms)
        norms[T] = float(mx.sqrt(mx.sum(g * g)))
    print("A/B mean |p_A - p_B| by T:", {k: round(v, 5) for k, v in gaps.items()})
    print("T^2*KL grad norm by T:    ", {k: round(v, 4) for k, v in norms.items()})
    print("Sanity checks passed.")