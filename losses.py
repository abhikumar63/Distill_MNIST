"""
CE / KD losses and the two teacher-ensemble target constructions.
Pure MLX + math — no project imports, so sanity.py can test this in
isolation before any data or model code runs.

  A: log softmax( mean(z1, z2) / T )              (logit average)
  B: log( (softmax(z1/T) + softmax(z2/T)) / 2 )    (probability mixture)
"""

import math

import mlx.core as mx


def log_softmax(x):
    """Version-proof log-softmax over the last axis."""
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def cross_entropy(logits, labels):
    """CE on raw logits. NO temperature."""
    log_probs = log_softmax(logits)
    picked = log_probs[mx.arange(labels.shape[0]), labels]
    return -mx.mean(picked)


def compute_accuracy(logits, labels):
    return mx.mean(mx.argmax(logits, axis=-1) == labels)


def log_target_A(z1, z2, T):
    return log_softmax(((z1 + z2) / 2.0) / T)


def log_target_B(z1, z2, T):
    lp1 = log_softmax(z1 / T)
    lp2 = log_softmax(z2 / T)
    return mx.logaddexp(lp1, lp2) - math.log(2.0)


def log_target_single(z, T):
    return log_softmax(z / T)


def make_log_target(method, z1, z2, T):
    if method == "kd_A":
        return log_target_A(z1, z2, T)
    if method == "kd_B":
        return log_target_B(z1, z2, T)
    if method == "kd_T1":
        return log_target_single(z1, T)
    if method == "kd_T2":
        return log_target_single(z2, T)
    raise ValueError(method)


def kd_kl_loss(teacher_log_probs, student_logits, T):
    """KL(teacher_T || student_T), summed over classes, mean over batch."""
    student_log_probs = log_softmax(student_logits / T)
    teacher_probs = mx.exp(teacher_log_probs)
    kl = teacher_probs * (teacher_log_probs - student_log_probs)
    return mx.mean(mx.sum(kl, axis=-1))
